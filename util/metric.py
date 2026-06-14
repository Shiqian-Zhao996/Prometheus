import lpips
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms


class CLIPEvaluator(object):
    def __init__(self, device='cuda', clip_model='openai/clip-vit-base-patch32') -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained(clip_model).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2").to(device)

        self.model.eval()
        self.lpips_fn.eval()
        self.sbert.eval()

    @torch.no_grad()
    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:
        
        # set the maximum length of the input text to 77
        text_tokens = self.clip_processor(text=text, return_tensors="pt", padding=False, truncation=True, max_length=77).to(self.device)
        text_features = self.model.get_text_features(**text_tokens)

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor, norm: bool = True) -> torch.Tensor:
        images = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**images)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features
    
    @torch.no_grad()
    def CLIP_II(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        similarity = (src_img_features @ gen_img_features.T).mean()
        similarity = round(similarity.item(), 4)

        return similarity

    @torch.no_grad()
    def CLIP_TI(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        similarity = (text_features @ gen_img_features.T).mean()
        similarity = round(similarity.item(), 4)

        return similarity
    
    @torch.no_grad()
    def LPIPS(self, src_images, generated_images):
        # convert the images to tensor
        to_tensor = transforms.ToTensor()
        src_images = [to_tensor(src_images)]
        generated_images = [to_tensor(generated_images)]

        # convert the images to tensor and convert the range from 0-255 to -1 to 1
        src_images = torch.stack(src_images).to(self.device)
        generated_images = torch.stack(generated_images).to(self.device)

        src_images = src_images * 2 - 1
        generated_images = generated_images * 2 - 1

        similarity = self.lpips_fn(src_images, generated_images).mean().item()
        similarity = round(similarity, 4)
        return similarity
    

    
    def SBERT_TT(self, text1, text2):
        embeddings = self.sbert.encode([text1, text2])
        similarity = self.sbert.similarity(embeddings[0], embeddings[1])

        similarity = round(similarity.item(), 4)

        similarity = (similarity + 1) / 2

        return similarity
