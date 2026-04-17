"""
Version: v1

Description:
This script performs image classification and explainability analysis using XAI techniques.

Includes:
- Pretrained ResNet18 model for image classification
- Confidence-based uncertainty (1 - probability)
- Top-3 prediction analysis
- Noise-based stability (robustness check)
- LIME (Local Interpretable Model-Agnostic Explanations) for visual explanation

Outputs:
- Predicted class and confidence
- Uncertainty score
- Prediction consistency under noise
- LIME explanation image

GitHub & Integration: Anand Narayan
Coding and Implementation:
- Ansh Batra
- Kushal Gupta
"""
import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
# import os
# from openai import OpenAI

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load model
model = models.resnet18(weights="DEFAULT")
model.eval()

def predict_fn(images):
    images = torch.tensor(images).permute(0,3,1,2).float() / 255.0
    
    transform_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    for i in range(images.shape[0]):
        images[i] = transform_norm(images[i])
    
    outputs = model(images)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.detach().numpy()


# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
img_path = input("Enter image path: ")
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Prediction
output = model(input_tensor)

# Labels
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Softmax (confidence)
prob = torch.nn.functional.softmax(output, dim=1)
confidence, pred_class = torch.max(prob, 1)
# -------------------------
# Uncertainty (Top-3 Predictions)
# -------------------------
top_probs, top_classes = torch.topk(prob, 3)

print("\nTop 3 Predictions:")
for i in range(3):
    print(f"{labels[top_classes[0][i]]}: {top_probs[0][i]*100:.2f}%")

# -------------------------
# Uncertainty Score
# -------------------------
uncertainty = 1 - confidence.item()
print(f"Uncertainty Score: {uncertainty:.2f}")

# -------------------------
# Advanced Uncertainty (Noise Stability)
# -------------------------
noisy_preds = []

for _ in range(5):
    noise = torch.randn_like(input_tensor) * 0.01
    noisy_output = model(input_tensor + noise)
    noisy_prob = torch.nn.functional.softmax(noisy_output, dim=1)
    noisy_preds.append(torch.max(noisy_prob, 1)[1].item())

print("Prediction consistency:", noisy_preds)
    
# Final Output
print("\n===== RESULT =====")
print(f"Prediction: {labels[pred_class.item()]}")
print(f"Confidence: {confidence.item()*100:.2f}%")


print("\nGenerating LIME explanation...")
explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    np.array(image),
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=2000
)

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=False,
    num_features=20,
    hide_rest=False
)

# -------------------------
# LIME Region Info
# -------------------------
num_regions = len(np.unique(mask))
print(f"\nLIME identified {num_regions} important regions influencing the prediction.")

lime_image_output = mark_boundaries(temp, mask)

# -------------------------
# AI Explanation (API)
# -------------------------

# label = labels[pred_class.item()]
# conf = confidence.item()*100

# prompt = f"""
# The model predicted '{label}' with {conf:.2f}% confidence.
# LIME highlighted important regions in the image.

# Explain in simple English why the model made this prediction.
# Focus on visual features like shape, texture, or object parts.
# """
# print("\nProcessing AI explanation...")
# try:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     ai_explanation = response.choices[0].message.content

#     print("\n===== AI EXPLANATION =====")
#     print(ai_explanation)

# except Exception as e:
#     print("\nAPI Error:", e)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(lime_image_output)
plt.title("LIME Explanation")
plt.axis('off')
plt.imsave("results/lime_output.png", lime_image_output)
plt.show()