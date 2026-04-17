"""
Version: v4 (FINAL SUBMISSION VERSION)

Description:
This is the final optimized explainable AI pipeline for image classification using ResNet-50.

It improves interpretability and uncertainty estimation by combining:
- Grad-CAM for spatial explanation
- Statistical p-value significance testing on logits
- Entropy-based uncertainty measurement
- Confidence scoring using softmax probabilities
- Activation distribution analysis (histogram)

This version represents the final evolution of the project:
LIME (baseline) → Grad-CAM (optimization) → Enhanced statistical + interpretability model (final)

Key Features:
- Pretrained ResNet50 (ImageNet)
- Grad-CAM heatmap visualization
- P-value based significance testing (z-score analysis)
- Entropy-based uncertainty quantification
- Attention distribution histogram
- Combined visualization dashboard

Outputs:
- Top-5 predictions with confidence & statistical significance
- Grad-CAM heatmap (saved)
- Combined analysis plot (saved)
- Detailed explanation report

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
from scipy import stats

# ── Load model ──────────────────────────────────────────
weights = models.ResNet50_Weights.IMAGENET1K_V1
model   = models.resnet50(weights=weights)
model.eval()
classes = weights.meta["categories"]

# ── Load & transform image ───────────────────────────────
img_path = input("Enter image path: ").strip()
image    = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tensor = transform(image).unsqueeze(0)

# ── Forward pass ─────────────────────────────────────────
with torch.no_grad():
    logits = model(tensor).cpu().numpy()[0]

probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
top5  = probs.argsort()[-5:][::-1]

# ── Per-prediction p-values (z-test vs background logits) ─
z_scores = (logits - logits.mean()) / (logits.std() + 1e-8)
p_values = stats.norm.sf(z_scores)   # one-tailed, right tail

# ── Print results ─────────────────────────────────────────
print(f"\n{'Rank':<4} {'Label':<35} {'Conf':>7} {'P-value':>9} {'Sig':>5}")
print("─" * 62)
for rank, i in enumerate(top5, 1):
    sig = "✓" if p_values[i] < 0.05 else "✗"
    print(f"{rank:<4} {classes[i][:34]:<35} {probs[i]*100:>6.2f}% {p_values[i]:>9.4f}  {sig}")

# ── Summary stats ─────────────────────────────────────────
best       = top5[0]
confidence = probs[best]
entropy    = -np.sum(probs * np.log(probs + 1e-10))
norm_ent   = entropy / np.log(len(classes))

# ── Grad-CAM ─────────────────────────────────────────────
store = {}
model.layer4[-1].register_forward_hook(lambda m, i, o: store.update({"act": o}))
model.layer4[-1].register_full_backward_hook(lambda m, i, o: store.update({"grad": o[0]}))

model(tensor)[0, best].backward()

cam = (store["grad"].mean(dim=(2, 3), keepdim=True) * store["act"]).sum(1).squeeze()
cam = torch.relu(cam)
cam = (cam / (cam.max() + 1e-8)).detach().numpy()

heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (224, 224))), cv2.COLORMAP_JET)
orig    = cv2.resize(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), (224, 224))
overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
cv2.imwrite("results/gradcam_output.jpg", overlay)

# ── Plot: Grad-CAM + bar chart + histogram ───────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"ResNet-50 — {classes[best]}", fontweight="bold")

ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
ax1.set_title("Grad-CAM Heatmap")
ax1.axis("off")

labels = [classes[i][:28] for i in top5]
confs  = [probs[i] * 100  for i in top5]
colors = ["#2ecc71" if p_values[i] < 0.05 else "#e74c3c" for i in top5]
ax2.barh(labels[::-1], confs[::-1], color=colors[::-1])
ax2.set_xlabel("Confidence (%)")
ax2.set_title("Top-5  (green = p<0.05)")
for j, (c, i) in enumerate(zip(confs[::-1], top5[::-1])):
    ax2.text(c + 0.2, j, f"{c:.1f}%  p={p_values[i]:.3f}", va="center", fontsize=8)

ax3.hist(cam.flatten(), bins=40, color="#26c6da", edgecolor="none")
ax3.axvline(0.5, color="#ff7043", linewidth=1.2, linestyle="--", label="0.5 threshold")
ax3.set_xlabel("Activation Intensity")
ax3.set_ylabel("Pixel Count")
ax3.set_title("Heatmap Distribution")
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/gradcam_analysis.png", dpi=120, bbox_inches="tight")
plt.show()

# ── Explanation ───────────────────────────────────────────
def explain(classes, best, confidence, p_values, entropy, norm_ent, cam):
    conf_word = "very high" if confidence > .9 else "high" if confidence > .7 else "moderate" if confidence > .5 else "low"
    ent_word  = "certain"   if norm_ent   < .1 else "confident" if norm_ent < .25 else "uncertain"
    focus_pct = (cam > 0.5).mean() * 100
    sig       = "significant ✓" if p_values[best] < 0.05 else "not significant ✗"

    print(f"""
─── Explanation ───────────────────────────────────────────
Predicted : {classes[best]}
Confidence: {confidence*100:.1f}%  → {conf_word}
P-value   : {p_values[best]:.4f}  → {sig}
Entropy   : {entropy:.3f}  (normalised {norm_ent:.2f}) → model is {ent_word}
Attention : {focus_pct:.1f}% of image is 'hot' in Grad-CAM

• Confidence — how strongly softmax scores this class.
• P-value    — z-test vs all 1000 logits; p<0.05 = meaningful.
• Entropy    — low = model is sure; high = image is ambiguous.
• Grad-CAM   — warm areas drove the prediction most.
• Histogram  — spike near 0 means focused attention on few regions.
───────────────────────────────────────────────────────────
Saved: gradcam_output.jpg  |  gradcam_analysis.png
""")

explain(classes, best, confidence, p_values, entropy, norm_ent, cam)

print("Saved: results/gradcam_output.jpg | results/gradcam_analysis.png")