from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Load RGB images
def load_image(path):
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")['pixel_values']
    return inputs

# Load x and y images
x_original= load_image("test_img/img_goat.jpg")
x=x_original.clone().detach().requires_grad_(True)  # Perturbable image
y = load_image("test_img/img_dog.jpg")


# Set PGD parameters
epsilon = 0.1  # Perturbation size
alpha = 0.01   # Step size
num_steps = 100

# Get the target image features
with torch.no_grad():
    y_features = model.get_image_features(y).detach()
    y_features = y_features / y_features.norm(p=2, dim=-1, keepdim=True)

# PGD optimization loop
for step in range(num_steps):
    # Get image features for x
    x_features = model.get_image_features(x)
    x_features = x_features / x_features.norm(p=2, dim=-1, keepdim=True)
    
    # Compute cosine similarity between x_features and y_features
    cosine_sim = torch.nn.functional.cosine_similarity(x_features, y_features)
    
    # Minimize cosine similarity by maximizing the negative similarity
    loss = 1-cosine_sim
    loss.backward()

    # Apply the PGD update
    with torch.no_grad():
        x = x - alpha * x.grad.sign()  # Update x in the direction to minimize similarity
        perturbation = torch.clamp(x - x_original, -epsilon, epsilon)  # Clip perturbation
        x = torch.clamp(x_original + perturbation, 0, 1).detach().requires_grad_(True)  # Apply perturbation

    # Optional: Print loss at each step
    if step % 10 == 0:


      
        print(f"Step {step+1}/{num_steps}, Loss: {loss.item()}")



# Import necessary library for visualization

# Convert original and perturbed images to PIL format for display
original_image = ToPILImage()(x_original.squeeze().detach().cpu())
perturbed_image = ToPILImage()(x.squeeze().detach().cpu())


# Original Image
Image.open("test_img/img_goat.jpg").show()

# Perturbed Image
plt.figure(figsize=(4, 4))
plt.imshow(perturbed_image)
plt.axis("off")
plt.show()

# Save the image with high resolution (e.g., 300 dpi)
perturbed_image.save("test_img/perturbed_goat.jpg", dpi=(300, 300))


# Calculate cosine similarity between perturbed x and y
with torch.no_grad():
    x_features_final = model.get_image_features(x)
    y_features_final = model.get_image_features(y)

    # Normalize the features
    x_features_final = x_features_final / x_features_final.norm(p=2, dim=-1, keepdim=True)
    y_features_final = y_features_final / y_features_final.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    final_cosine_similarity = torch.nn.functional.cosine_similarity(x_features_final, y_features_final)
    print(f"Cosine Similarity between perturbed x and y: {final_cosine_similarity.item():.4f}")
    
    x_features_original = model.get_image_features(x_original)
    y_features_final = model.get_image_features(y)

    # Normalize the features
    x_features_original = x_features_original / x_features_original.norm(p=2, dim=-1, keepdim=True)
    y_features_final = y_features_final / y_features_final.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    orig_cosine_similarity = torch.nn.functional.cosine_similarity(x_features_original, y_features_final)
    print(f"Cosine Similarity between original x and y: {orig_cosine_similarity.item():.4f}")
