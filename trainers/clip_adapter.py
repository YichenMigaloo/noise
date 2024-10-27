import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import os
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torchvision.transforms as transforms

_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a {} image.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',

    'biggan': 'a {} photo.',
    'cyclegan': 'a {} photo.',
    'dalle2': 'a {} photo.',
    'deepfake': 'a {} photo.',
    'eg3d': 'a {} photo.',
    'gaugan': 'a {} photo.',
    'glide_50_27': 'a {} photo.',
    'glide_100_10': 'a {} photo.',
    'glide_100_27': 'a {} photo.',
    'guided': 'a {} photo.',
    'ldm_100': 'a {} photo.',
    'ldm_200': 'a {} photo.',
    'ldm_200_cfg': 'a {} photo.',
    'progan': 'a {} photo.',
    'sd_512x512': 'a {} photo.',
    'sdxl': 'a {} photo.',
    'stargan': 'a {} photo.',
    'stylegan': 'a {} photo.',
    'stylegan2': 'a {} photo.',
    'stylegan3': 'a {} photo.',
    'taming': 'a {} photo.',
    'firefly': 'a {} photo.',
    'midjourney_v5': 'a {} photo.',
    'dalle3': 'a {} photo.',
    'faceswap': 'a {} photo.',
    'progan_train': 'a {} photo.',
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            
            nn.Linear(768, 384, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(384, 768, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(1024, 4).to(clip_model.dtype)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)
        ratio = 0.4
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    


def encode_output_path(image_path):
        directory, filename = os.path.split(image_path)
        new_directory = directory.replace('/images', '/noiseprint')
        output_filename = filename + ".npz"
        output_path = os.path.join(new_directory, output_filename)
        return output_path
def load_noiseprint(npz_path):
        output_path = encode_output_path(npz_path)
        data = np.load(output_path)
        map_data = data['map']
        conf_data = data['conf']
        
        # Convert numpy arrays to torch tensors
        map_tensor = torch.tensor(map_data)
        conf_tensor = torch.tensor(conf_data)
        
        return map_tensor, conf_tensor

def prepare_custom_map(map, conf):
    # 确保 map 和 conf 是 2D 的 [H, W]
    if len(map.shape) == 2:
        map = map.unsqueeze(0)  # 添加通道维度，变为 [1, H, W]
    if len(conf.shape) == 2:
        conf = conf.unsqueeze(0)  # 添加通道维度，变为 [1, H, W]
    target_size = (224,224)
    transform = transforms.CenterCrop(target_size)
    map = transform(map)
    conf = transform(conf)
    #blank = torch.zeros_like(map)
    combined = torch.cat((map, conf), dim=0)
    
    return combined



def modify_first_conv_layer(model, new_in_channels=5):
    old_conv = model.visual.conv1
    
    # 创建一个新的卷积层，修改输入通道数为5
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,  # 修改输入通道数为5
        out_channels=old_conv.out_channels,  # 保持输出通道数不变
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None  # 保留是否有bias
    )
    
    # 初始化新的卷积层
    with torch.no_grad():
        # 将原始3通道的卷积权重复制到新卷积层的前3个通道
        new_conv.weight[:, :3, :, :] = old_conv.weight  # 保持前3个通道的权重
        # 对新增的两个通道 (map 和 conf) 进行随机初始化
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        if old_conv.bias is not None:
            new_conv.bias = old_conv.bias
    
    # 替换模型中的第一层卷积层
    model.visual.conv1 = new_conv

    return model


@TRAINER_REGISTRY.register()
class CLIP_Adapter(TrainerX):
    """ CLIP-Adapter """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model = modify_first_conv_layer(clip_model, new_in_channels=5)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        #print('Trainable Parameters: ', str(params))

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        self.register_model('clip_adapter', self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        '''input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label'''
        input = batch['img']  # 这是原始的 RGB 图像 (3 通道)
    
        # 获取每个图像对应的 noiseprint 数据
        impaths = batch['impath']
        maps = []
        for path in impaths:
            map_tensor, conf_tensor = load_noiseprint(path)  # 分别加载 map 和 conf tensor
            # 将 map_tensor 和 conf_tensor 处理为所需尺寸
            temp_map = prepare_custom_map(map_tensor, conf_tensor)  # 返回 map_tensor 和 conf_tensor 的组合
            
            maps.append(temp_map)

        # 将所有 noiseprint 数据组合成 batch
        maps_batch = torch.stack(maps)
        
        # 将 RGB 图像和 noiseprint 的 map 和 conf 合并为 5 通道输入
        input = input.to(self.device)
        maps_batch = maps_batch.to(self.device)
        
        # 将 RGB 图像 (3 通道) 和 noiseprint map + conf (2 通道) 合并为 5 通道输入
        combined_input = torch.cat((input, maps_batch), dim=1)  # 在通道维度拼接，最终形成 5 通道张量

        label = batch['label'].to(self.device)

        return combined_input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return
        # print(
        #         'Note that load_model() is skipped as no pretrained model is given'
        #     )
        # return
        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # this is modified for weight ensemble, comment this for clip adapter
            # checkpoint = load_checkpoint(model_path)
            # state_dict = checkpoint
            # epoch = 1
            
            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

