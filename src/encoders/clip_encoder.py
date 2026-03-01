import torch
import numpy as np
from torchvision import transforms
from .base import BaseEncoder, EncoderOutput


class CLIPEncoder(BaseEncoder):
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str = "cuda", image_size: int = 224):
        super().__init__(model_id=model_id, device=device, image_size=image_size)

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def spatial_token_dim(self) -> int:
        return 1024

    @property
    def encoder_version(self) -> str:
        return "clip-vit-large-patch14-v1"

    def _build(self):
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.processor = CLIPProcessor.from_pretrained(self.model_id)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def _extract(self, pixel_values: torch.Tensor) -> EncoderOutput:
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        cls_token = vision_outputs.last_hidden_state[:, 0, :]
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]

        global_embedding = cls_token.cpu().numpy().squeeze(0)
        spatial_features = patch_tokens.cpu().numpy().squeeze(0)

        norm = np.linalg.norm(global_embedding)
        if norm > 0:
            global_embedding = global_embedding / norm

        return EncoderOutput(
            global_embedding=global_embedding,
            spatial_features=spatial_features,
        )
