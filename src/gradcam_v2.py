"""
Version: v2

Description:
This script improves upon the initial LIME-based explainability approach by introducing Grad-CAM for faster and more stable visual explanations in CNN-based image classification.

It focuses on improving computational efficiency and better spatial interpretability of model decisions compared to LIME.

Includes:
- Pretrained ResNet50 model for image classification
- Softmax-based confidence scoring
- Monte Carlo Dropout-based uncertainty estimation
- Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations
- Top-3 prediction analysis

Outputs:
- Top-3 predicted classes with confidence scores
- Prediction confidence and uncertainty score
- Grad-CAM heatmap visualization
- Saved heatmap image for analysis

Evolution Note:
- v1: LIME-based explanations (slow and computationally expensive)
- v2: Grad-CAM introduced for efficient and stable explanations

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

# -------------------------
# Model (Grad-CAM)
# -------------------------
model = models.resnet18(weights="DEFAULT")
model.eval()

# -------------------------
# Dropout model (uncertainty)
# -------------------------
class ResNetDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights="DEFAULT")
        self.features = torch.nn.Sequential(*(list(base.children())[:-1]))
        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc = torch.nn.Linear(base.fc.in_features, 1000)
        self.fc.load_state_dict(base.fc.state_dict())

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

uncertainty_model = ResNetDropout()

# -------------------------
# Hooks
# -------------------------
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -------------------------
# Labels
# -------------------------
with open("imagenet_classes.txt") as f:
    labels = [l.strip() for l in f.readlines()]

# -------------------------
# Input
# -------------------------
img_path = input("Enter image path: ")
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# -------------------------
# Prediction
# -------------------------
output = model(input_tensor)
prob = F.softmax(output, dim=1)

top3_prob, top3_class = torch.topk(prob, 3)

print("\nTop 3 Predictions:")
for i in range(3):
    print(f"{labels[top3_class[0][i]]}: {top3_prob[0][i]*100:.2f}%")

pred_class = top3_class[0][0]

# -------------------------
# Uncertainty
# -------------------------
uncertainty_model.train()
mc_samples = 30
preds = []

for _ in range(mc_samples):
    out = uncertainty_model(input_tensor)
    preds.append(F.softmax(out, dim=1).detach().numpy())

uncertainty_model.eval()

preds = np.stack(preds)
mean_prob = preds.mean(axis=0)
std_prob = preds.std(axis=0)

uncertainty = float(std_prob[0, pred_class] * 100)

print(f"\nPredicted: {labels[pred_class.item()]}")
print(f"Confidence: {mean_prob[0, pred_class]*100:.2f}%")
print(f"Uncertainty Score: {uncertainty:.2f}%")

# -------------------------
# Grad-CAM
# -------------------------
model.zero_grad()
output[0, pred_class].backward()

acts = activations[-1].detach().cpu()
grads = gradients[-1].detach().cpu()

weights = torch.mean(grads, dim=(2,3), keepdim=True)
cam = torch.sum(weights * acts, dim=1)

cam = F.relu(cam)
cam = cam.squeeze().detach().cpu().numpy().astype(np.float32)

cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)
cam = np.power(cam, 2.0)

cam = cv2.resize(cam, (224,224))

# Heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

img_cv = np.array(image)
img_cv = cv2.resize(img_cv, (224,224))

overlay = cv2.addWeighted(img_cv, 0.4, heatmap, 0.8, 0)

cv2.imwrite("results/gradcam_output.png", overlay)
print("Grad-CAM saved in results/gradcam_output.png")

