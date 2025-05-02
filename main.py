from flask import Flask, render_template, request, jsonify, send_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from image import generate_image_from_theme_description
from huggingface_hub import login
import os

# Initialize Flask
app = Flask(__name__)

# Authenticate with Hugging Face using environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)

# Load the LLaMA model and tokenizer
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

def generate_event_theme(event_type, audience, season, venue):
    prompt = (
        f"Generate a creative theme for a {event_type} event. "
        f"The theme should reflect the {season} season with colors. "
        f"The venue is a {venue}, and the audience consists of {audience}. "
        f"Provide a unique theme name and a concise theme description."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=250,
            num_beams=7,
            no_repeat_ngram_size=3,
            temperature=0.7,
            early_stopping=True
        )

    theme_description = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if theme_description.startswith(prompt):
        theme_description = theme_description[len(prompt):].strip()
    return theme_description

@app.route('/')
def home():
    return "Theme Generator API is running!"

@app.route('/generate-theme', methods=['POST'])
def generate_theme():
    data = request.json
    event_type = data['event_type']
    audience = data['audience']
    season = data['season']
    venue = data['venue']

    theme_description = generate_event_theme(event_type, audience, season, venue)
    image_path = generate_image_from_theme_description(theme_description)
    return jsonify({'theme_description': theme_description, 'image_path': image_path})

@app.route('/download-image', methods=['GET'])
def download_image():
    file_path = request.args.get('file_path')
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002)

