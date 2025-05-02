from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model and pipeline
model_name = "black-forest-labs/FLUX.1-dev"  # Using Stable Diffusion 2.1 for public access
pipe = StableDiffusionPipeline.from_pretrained(model_name)
pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")  # Use GPU if available

# Function to generate an event theme image based on theme description
def generate_image_from_theme_description(theme_description):
    # Generate the image based on the theme description from LLM
    image = pipe(theme_description, num_inference_steps=10, guidance_scale=7.5).images[0]

    # Save the image
    output_path = "static/event_theme_image.png"  # Save in static folder for web access
    image.save(output_path)
    return output_path
