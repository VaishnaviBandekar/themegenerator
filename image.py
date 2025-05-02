from diffusers import StableDiffusionPipeline
import torch
import os

# Get the Hugging Face token from the environment
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load the Stable Diffusion model using the token
model_name = "black-forest-labs/FLUX.1-dev"
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    use_auth_token=HF_TOKEN
)

pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")

def generate_image_from_theme_description(theme_description):
    image = pipe(theme_description, num_inference_steps=10, guidance_scale=7.5).images[0]
    output_path = "static/event_theme_image.png"
    image.save(output_path)
    return output_path
