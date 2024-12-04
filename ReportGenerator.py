from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
import torch
from PIL import Image
import os

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the GPT-2 model and tokenizer for generating descriptive text
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Define a function to process the image and generate descriptive text
def generate_detailed_description(image_path):
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None
    
    # Step 1: Load and process the image
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    # Step 2: Extract features using CLIP
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # Step 3: Decode features to text prompt
    image_features /= image_features.norm(dim=-1, keepdim=True)
    prompt = "This image shows a scene where "
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(gpt2_model.device)

    # Step 4: Generate descriptive text using GPT-2
    gpt2_model.eval()
    with torch.no_grad():
        output = gpt2_model.generate(
            input_ids,
            max_length=100,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    description = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return description
