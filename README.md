# XAI-Uncertainty-Quantification

Group implementation project for AI111 course

The aim of this project is to understand how explainable AI methods work using Grad-CAM and how reliable model predictions are under uncertainty.

We used a pretrained ResNet18 model for image classification and applied Grad-CAM to visualize which regions of the image influence the model’s prediction. Along with this, we estimated uncertainty using Monte Carlo Dropout to understand prediction stability.

---

## What we did

- Used ResNet18 for image classification  
- Extracted top-3 predicted classes  
- Computed prediction confidence using softmax  
- Estimated uncertainty using Monte Carlo Dropout sampling  
- Generated Grad-CAM heatmaps for visual explanations  
- Saved final explanation output as image  

---

## Outputs

- Predicted class and confidence  
- Top-3 predictions  
- Uncertainty score (%)  
- Grad-CAM heatmap visualization (saved in results folder)  

---

## Project Structure

- src → contains main code (Grad-CAM implementation)  
- images → input test images  
- results → output heatmaps  
- imagenet_classes.txt → class labels  

---

## How to run

Install required libraries:

pip install torch torchvision numpy matplotlib pillow opencv-python

Run:

python src/gradcam_main.py

Enter image name (from images folder) when prompted.

---

## Team

- Anand Narayan (GitHub, integration, version control)  
- Ansh Batra (coding)  
- Kushal Gupta (coding)  
- Harshit Mishra (Presentation and report)  
- Pushkar Aggarwal (Literature review)  

---

## My Contribution (GitHub Role)

- Managed and structured the GitHub repository  
- Integrated final Grad-CAM implementation into clean project structure  
- Maintained reproducible execution workflow  
- Organized dataset, code, and output files  
- Ensured submission-ready project formatting and documentation  

---

## Mathematical Concepts Used

- Probability (softmax output)  
- Monte Carlo Dropout for uncertainty estimation  
- Gradient-based activation mapping (Grad-CAM)  
- Feature importance visualization  

---

## Limitations

- Grad-CAM resolution is coarse and approximate  
- Monte Carlo Dropout provides only an estimate of uncertainty  
- Results depend on pretrained ResNet18 model  
- Explanations are not fully deterministic  

---

## Observations

We observed that Grad-CAM provides better spatial consistency compared to earlier explainability approaches. However, uncertainty estimation still varies with different stochastic forward passes, indicating that model confidence is not always reliable.

This shows that explainability and uncertainty are still open research problems in deep learning.