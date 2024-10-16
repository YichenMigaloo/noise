import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Tokenization function with exception handling
def tokenize_prompts(classnames):
    try:
        return torch.cat([clip.tokenize(c) for c in classnames])
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return None

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def load_vit_without_last_layer(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location = 'cpu').eval()
        state_dict =None
        model = torch.jit._unwrap_optional(torch.jit._recursive.wrap_cpp_module(model._c))  # unwrap model

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict())
    original_forward = model.visual.forward

    def forward_without_proj(x):
        x = original_forward(x)
        if hasattr(model.visual, 'proj'):
            x = x  
        return x

    model.visual.forward = forward_without_proj
    return model

# Adapter from the first model
class Adapter(nn.Module):
    #Linear Adapter
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 384, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(384, 768, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.to(self.fc[0].weight.dtype)
        x = self.fc(x)
        return x

# Load the Images and Extract Noise Print
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def extract_noise_print(image):
    # Dummy noise print extraction (replace with actual noise extraction method)
    noise = torch.randn_like(image) * 0.1  # Adding random noise as a placeholder
    return noise

# Merge Function into Existing Code
def extract_and_fuse_embeddings(model, image_path):
    # Load the RGB image and the noise print
    rgb_image = load_image(image_path)
    noise_print = extract_noise_print(rgb_image)
    
    # Combine both images into a batch
    images = torch.cat((rgb_image, noise_print), dim=0)
    
    # Pass through the network to get embeddings
    embeddings = model(images)
    
    # Combine the Embeddings
    combined_embedding = torch.cat((embeddings[0], embeddings[1]), dim=0)  # Concatenate embeddings
    
    return combined_embedding

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.attn_mask = None  # Store the attention mask for the transformer

    def forward(self, prompts, tokenized_prompts):
        # Adjust positional embeddings to match the sequence length of prompts
        seq_length = prompts.shape[1]  # Get the sequence length of prompts

        # Extend or slice the positional embeddings to match the prompt sequence length
        if seq_length > self.positional_embedding.shape[0]:
            positional_embedding = self._extend_positional_embeddings(seq_length).type(self.dtype)
        else:
            positional_embedding = self.positional_embedding[:seq_length, :].type(self.dtype)

        # Ensure shapes are compatible for addition
        if positional_embedding.shape[0] != prompts.shape[1]:
            raise ValueError(f"Positional embedding shape {positional_embedding.shape} does not match prompt shape {prompts.shape}")

        # Add positional embedding to prompts
        x = prompts + positional_embedding
        
        # Cast tensors to ensure consistent data types (to avoid Float/Half precision mismatch)
        x = x.to(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND for transformer

        # Update attention mask to match the sequence length
        self._update_attention_mask(seq_length)

        # Use autocast for mixed precision training to ensure the right precision
        with torch.cuda.amp.autocast():
            x = self.transformer(x)  # Pass through transformer

        x = x.permute(1, 0, 2)  # LND -> NLD after transformer
        x = self.ln_final(x).type(self.dtype)

        # Take features from the end-of-token (eot) embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    def _extend_positional_embeddings(self, target_length):
        """Extend the positional embeddings to match the required sequence length."""
        current_length = self.positional_embedding.shape[0]
        if target_length > current_length:
            repeat_factor = (target_length // current_length) + 1
            extended_positional_embedding = self.positional_embedding.repeat(repeat_factor, 1)[:target_length, :]
            return extended_positional_embedding
        else:
            return self.positional_embedding

    def _update_attention_mask(self, seq_length):
        """Update the attention mask to match the sequence length."""
        for layer in self.transformer.resblocks:
            layer.attn_mask = torch.full(
                (seq_length, seq_length), float("-inf")
            ).triu(1).to(layer.attn_mask.device)  # Upper triangular mask for attention

# PromptLearner from the second model
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)  # Get number of classes
        self.n_cls = n_cls  # Store the number of classes as a class attribute
        n_ctx = cfg.TRAINER.COOP.N_CTX  # Number of context vectors
        ctx_dim = clip_model.ln_final.weight.shape[0]  # Dimension of the context

        # Initialize context vectors
        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # Token prefix and suffix (non-trainable parts)
        # We need to ensure that the prefix and suffix are initialized for each class
        classnames = [name.replace("_", " ") for name in classnames]
        tokenized_classnames = clip.tokenize(classnames)
        self.token_prefix = clip_model.token_embedding(tokenized_classnames[:, :1]).type(clip_model.dtype)  # First token (CLS)
        self.token_suffix = clip_model.token_embedding(tokenized_classnames[:, 1:]).type(clip_model.dtype)  # Remaining tokens (CLS removed)

    def forward(self):
        # Ensure everything is on the same device
        device = self.ctx.device
        prefix = self.token_prefix.to(device)
        suffix = self.token_suffix.to(device)
        ctx = self.ctx.to(device)
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Ensure all components (prefix, ctx, suffix) have compatible shapes before concatenation
        #print(f"Prefix shape: {prefix.shape}, Context shape: {ctx.shape}, Suffix shape: {suffix.shape}")

        # List to hold all prompts
        prompts = []  # Initialize as an empty list

        # For each class, concatenate prefix, context, and suffix
        for i in range(self.n_cls):
            prefix_i = prefix[i : i + 1, :, :]  # Correctly slice the prefix for class i
            suffix_i = suffix[i : i + 1, :, :]  # Correctly slice the suffix for class i
            ctx_i = ctx[i : i + 1, :, :]  # Correctly slice the context for class i
            
            # Ensure that prefix, ctx, and suffix are compatible for concatenation
            #print(f"Prefix_i shape: {prefix_i.shape}, Context_i shape: {ctx_i.shape}, Suffix_i shape: {suffix_i.shape}")
            
            # Check if dimensions match (except for dimension 1, which can vary)
            if prefix_i.shape[2] != ctx_i.shape[2] or ctx_i.shape[2] != suffix_i.shape[2]:
                raise ValueError(f"Dimension mismatch: Prefix_i shape: {prefix_i.shape}, Context_i shape: {ctx_i.shape}, Suffix_i shape: {suffix_i.shape}")

            prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
            prompts.append(prompt)  # Append each prompt to the list

        # After the loop, concatenate all prompts in the list into a single tensor
        prompts = torch.cat(prompts, dim=0)  # Convert list of tensors into a single tensor
        
        return prompts

# CustomCLIP integrating both Adapter and PromptLearner
class AdapterPrompt(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual  # Image encoder from CLIP
        self.text_encoder = TextEncoder(clip_model)  # Use the modified TextEncoder with attention mask
        # Integrating both trainable parts: Adapter and PromptLearner
        self.adapter = Adapter(1024, 4)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, classnames):
        prompts = self.prompt_learner()
        tokenized_prompts = tokenize_prompts(classnames)
        if tokenized_prompts is None:
            return None
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = self.image_encoder(image.type(self.dtype))

        adapted_image_features = self.adapter(image_features.to(self.adapter.fc[0].weight.dtype))
        #adapted_image_features = self.adapter(image_features.to(self.adapter.conv[0].weight.dtype))
        #adapted_image_features = self.adapter(image_features.to(self.adapter.query.weight.dtype))
        #adapted_image_features = self.adapter(image_features.to(self.adapter.mlp[0].weight.dtype))

        image_features = adapted_image_features / adapted_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_features = text_features.to(image_features.dtype)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

# Trainer class combining both models and integrating training for Adapter and PromptLearner
@TRAINER_REGISTRY.register()
class UnifiedTrainer(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Classnames:{classnames}")
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        #clip_model = load_clip_to_cpu(cfg)
        clip_model = load_vit_without_last_layer(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building unified CLIP model")
        self.model = AdapterPrompt(cfg, classnames, clip_model)
        self.model.to(self.device)
        print("Turning off gradients in both the image and text encoder (except trainable parts)")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "adapter" not in name:
                param.requires_grad_(False)

        # Ensure optimizer is initialized with trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # Manual registration of model and optimizer
        self._models["unifiedtrainer"] = self.model
        self._optims["unifiedtrainer"] = self.optim
        self._scheds["unifiedtrainer"] = self.sched

        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected ({torch.cuda.device_count()} GPUs), using DataParallel")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        if self.cfg.TRAINER.COOP.PREC == "amp":
            with autocast():
                output = self.model(image, self.dm.dataset.classnames)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()
        else:
            output = self.model(image, self.dm.dataset.classnames)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
