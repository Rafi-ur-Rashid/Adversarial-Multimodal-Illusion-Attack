from PIL import Image
import requests

from transformers import AutoProcessor, AutoTokenizer, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["a big horn goat"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)
path = "test_img/perturbed_dog.jpg"
image = Image.open(path)
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)

# Normalize the features
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# Calculate the similarity
similarity = (image_features @ text_features.T).item()
print(f"Similarity: {similarity:.4f}")
