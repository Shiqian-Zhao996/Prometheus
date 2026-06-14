import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

class IncontextMatcher:
    def __init__(self, device='cuda', clip_model_name="ViT-L-14/openai", text_batch_size=32):
        self.device = device
        self.text_encoder = None
        self.image_encoder = None
        self.text_batch_size = text_batch_size
        self.norm = True

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.clip_model = self.clip_model.to(self.device)

        self.clip_model.eval()

    def match_clip(self, text, image, base_prompt=None):
        if len(text) == 0:
            return np.array([])

        with torch.no_grad():

            embeds = []
            images = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.clip_model.get_image_features(**images)
            if self.norm:
                image_features /= image_features.norm(dim=-1, keepdim=True)

            if base_prompt is not None:
                text = [base_prompt + ", " + modifier for modifier in text]


            chunks = np.array_split(text, max(1, len(text)/self.text_batch_size))
            for chunk in chunks:
                chunk = list(chunk)
                text_tokens = self.clip_processor(text=chunk, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
                text_features = self.clip_model.get_text_features(**text_tokens)

                if self.norm:
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                for text_feature in text_features:
                    embeds.append(text_feature)

            embeds = torch.stack(embeds)

            clip_similarity = (image_features @ embeds.T).squeeze().cpu().numpy()

            return clip_similarity
