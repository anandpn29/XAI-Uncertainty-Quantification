# XAI-Uncertainty-Quantification
Group implementation project for AI111 course

The aim of this project is to understand how explainable AI methods work and how reliable those explanations are.

We used a pretrained ResNet18 model for image classification and applied LIME to see which parts of the image are important for prediction. Along with this, we tried to measure uncertainty in the model’s output.

---

## What we did

- Used ResNet18 for prediction
- Checked top-3 predicted classes
- Calculated uncertainty using confidence score (1 - probability)
- Added small noise to input and observed prediction changes
- Generated explanation using LIME

---

## Outputs

- Predicted class and confidence
- Top-3 predictions
- Uncertainty score
- Prediction consistency after noise
- LIME output image (saved in results folder)

---

## Project Structure

- src → contains code  
- results → contains output images  
- docs → report / ppt  
- imagenet_classes.txt → class labels  

---

## How to run

Install required libraries:

pip install torch torchvision matplotlib numpy pillow lime scikit-image

Run:

python src/lime_uncertainty_v1.py
---

## Team
- Anand Narayan (GitHub, integration)
- Ansh Batra (coding)
- Kushal Gupta (coding)
- Harshit Mishra(Presentation and report)
- Pushkar Aggarwal(Literature review)

## Some observations
We noticed that even when prediction confidence is high, explanation can change with small noise. This shows that explanations are not always stable.

Also, it is difficult to clearly define how much uncertainty is acceptable in explanations.
