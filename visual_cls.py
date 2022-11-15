import torch.nn as nn
import torch
from clip.model import VisionTransformer


class Vit_cls(torch.nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        if arch=='vit_base_patch32_224':
            vision_width, image_resolution = 768, 224
            vision_patch_size, vision_layers, embed_dim = 32, 12, 768
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
        self.head = torch.nn.Linear(embed_dim, num_classes)
        del self.visual.proj
        
    def forward(self, images):
        image_embed = self.visual(images.type(self.dtype))
        logits = self.head(image_embed[:,0,:])
        return logits