import torch
from diffusers import DiffusionPipeline


class OnlineModel:
    def __init__(self, model_name=None, model_path=None, image_size=512, device="cuda"):
        self.model_name = model_name
        self.model_path = model_path
        self.image_size = image_size
        self.device = device
        self.random_seed = 0

        if model_name is not None:
            self.load_model()
            

    def load_model(self):
        if self.model_name == "SD3.5":
            pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.float16)
            pipe.safety_checker = None
            self.model = pipe
            self.model = self.model.to(self.device)
            self.generator = torch.Generator(device=self.device).manual_seed(self.random_seed)
        elif self.model_name == "FLUX":
            pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
            self.model = pipe
            self.model = self.model.to(self.device)
            self.generator = torch.Generator(device=self.device).manual_seed(self.random_seed)
        elif self.model_name == "ShuttleDiffusion":
            pipe = DiffusionPipeline.from_pretrained("shuttleai/shuttle-3-diffusion", torch_dtype=torch.float16)
            self.model = pipe
            self.model = self.model.to(self.device)
            self.generator = torch.Generator(device=self.device).manual_seed(self.random_seed)

        else:
            raise ValueError(f"Model {self.model_name} not found")

    def query(self, prompt):
        if self.model_name == "SD3.5":
            self.model.set_progress_bar_config(disable=False)
            self.model.safety_checker = None
            self.generator = torch.Generator(device=self.device).manual_seed(self.random_seed)
            response = self.model(prompt, guidance_scale=0.0, num_inference_steps=4, generator=self.generator).images[0]

        elif self.model_name == "FLUX":
            self.model.set_progress_bar_config(disable=False)
            self.generator = torch.Generator(device=self.device).manual_seed(self.random_seed)
            response = self.model(prompt, height=512, width=512, guidance_scale=0.0, num_inference_steps=4, generator=self.generator).images[0]
        
        elif self.model_name == "ShuttleDiffusion":
            self.model.set_progress_bar_config(disable=False)
            self.generator = torch.Generator(device=self.device).manual_seed(self.random_seed)
            response = self.model(prompt, height=512, width=512, guidance_scale=3.5, num_inference_steps=4, generator=self.generator).images[0]
        else:
            raise ValueError(f"Model {self.model_name} not found")
        
        response = response.resize((self.image_size, self.image_size))
        return response
