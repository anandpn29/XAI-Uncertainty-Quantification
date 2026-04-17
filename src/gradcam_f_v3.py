"""
Version: v3 

Description:
This script performs image classification and explainability analysis using advanced XAI techniques.

It improves upon earlier LIME-based implementation by using Grad-CAM for faster and more stable visual explanations.
Additionally, it includes entropy-based uncertainty estimation and activation distribution analysis.

Includes:
- Pretrained ResNet50 model for image classification
- Softmax-based confidence scoring
- Entropy-based uncertainty measurement
- Grad-CAM heatmap visualization
- Activation histogram analysis for interpretability
- Basic explanation generation based on model outputs

Outputs:
- Top-3 predicted classes with confidence scores
- Prediction confidence and uncertainty score
- Entropy of prediction distribution
- Grad-CAM heatmap visualization
- Activation intensity histogram
- Saved heatmap image for reporting

Evolution Note:
- v1: LIME-based explanation (slow, high computation cost)
- v2: Grad-CAM introduced for improved efficiency and stability
- v3: Enhanced Grad-CAM with entropy + histogram analysis for better interpretability

GitHub & Integration: Anand Narayan
Coding and Implementation:
- Ansh Batra
- Kushal Gupta
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# 1. INPUT
# ---------------------------
img_path = input("Enter image path: ").strip()

# ---------------------------
# 2. LOAD MODEL
# ---------------------------
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model.eval()

# ---------------------------
# 3. TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# ---------------------------
# 4. LOAD LABELS
# ---------------------------
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# ---------------------------
# 5. PREDICTION
# ---------------------------
with torch.no_grad():
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1).cpu().numpy()

top3_idx = probs[0].argsort()[-3:][::-1]

print("\nTop 3 Predictions:")
for i in top3_idx:
    print(f"{classes[i]}: {probs[0][i]*100:.2f}%")

pred_class = classes[top3_idx[0]]
confidence = probs[0][top3_idx[0]]

# ---------------------------
# 6. P-VALUE & UNCERTAINTY
# ---------------------------
p_value = confidence
entropy = -np.sum(probs[0] * np.log(probs[0] + 1e-10))
uncertainty = 1 - confidence

print(f"\nPredicted: {pred_class}")
print(f"Confidence: {confidence*100:.2f}%")
print(f"P-value (approx): {p_value:.4f}")
print(f"Uncertainty Score: {uncertainty*100:.2f}%")
print(f"Entropy: {entropy:.4f}")

# ---------------------------
# 7. GRAD-CAM
# ---------------------------
gradients = None
activations = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer = model.layer4[-1]

fh = target_layer.register_forward_hook(forward_hook)
bh = target_layer.register_full_backward_hook(backward_hook)

# Forward
output = model(input_tensor)
class_idx = output.argmax()

# Backward
model.zero_grad()
output[0, class_idx].backward()

fh.remove()
bh.remove()

# Build CAM
grads = gradients.detach()
acts = activations.detach()

weights_cam = grads.mean(dim=(2, 3), keepdim=True)
cam = (weights_cam * acts).sum(dim=1).squeeze()

cam = torch.relu(cam)
cam = cam / (cam.max() + 1e-8)
cam = cam.cpu().numpy()

# ---------------------------
# 🔥 NEW: HEATMAP HISTOGRAM
# ---------------------------
plt.figure()
plt.hist(cam.flatten(), bins=50)
plt.title("Heatmap Intensity Distribution")
plt.xlabel("Activation Intensity")
plt.ylabel("Frequency")
plt.savefig("results/histogram.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------
# 8. HEATMAP OVERLAY
# ---------------------------
heatmap = cv2.resize(cam, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

orig = cv2.imread(img_path)
orig = cv2.resize(orig, (224, 224))

overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

cv2.imwrite("results/gradcam_output.jpg", overlay)

# Display
plt.figure()
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.show()

print("\n✅ Heatmap and histogram displayed and saved as gradcam_output.jpg")

# ---------------------------
# 9. EXPLANATION
# ---------------------------
def explain(pred_class, confidence, uncertainty, entropy):
    if confidence > 0.8:
        conf_text = "very confident"
    elif confidence > 0.5:
        conf_text = "moderately confident"
    else:
        conf_text = "less confident"

    return f"""
--- AI Explanation ---

The model predicts the image contains a '{pred_class}'.

Confidence: {confidence*100:.2f}%  
P-value (approx): {confidence:.4f}  
Entropy: {entropy:.4f}  

Lower entropy means higher certainty.

Histogram shows how concentrated the model's attention is.
"""

print(explain(pred_class, confidence, uncertainty, entropy))