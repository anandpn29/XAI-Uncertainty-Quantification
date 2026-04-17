# XAI-Uncertainty-Quantification

Group implementation project for AI111 course

The aim of this project is to understand how explainable AI methods work using Grad-CAM and how reliable model predictions are under uncertainty.

This final version enhances the previous Grad-CAM implementation by adding entropy-based uncertainty estimation and activation distribution analysis (histogram) to better understand model confidence and interpretability.

We used a pretrained ResNet50 model for image classification and applied Grad-CAM to visualize which regions of the image influence the model’s prediction. Along with this, we estimated uncertainty using Monte Carlo Dropout and entropy to analyze prediction stability.

---

## What we did

- Used ResNet50 for image classification  
- Extracted top-3 predicted classes  
- Computed prediction confidence using softmax  
- Estimated uncertainty using Monte Carlo Dropout sampling  
- Calculated entropy of prediction distribution  
- Generated Grad-CAM heatmaps for visual explanations  
- Plotted activation intensity histogram for interpretability  
- Saved final outputs as images for analysis  

---

## Outputs

- Predicted class and confidence  
- Top-3 predictions  
- Uncertainty score (%)  
- Entropy value of prediction distribution  
- Grad-CAM heatmap visualization (saved in results folder)  
- Activation histogram plot (saved in results folder)  

---

## Project Structure

- src → contains main code (Grad-CAM final implementation)  
- images → input test images  
- results → output heatmaps and graphs  
- imagenet_classes.txt → class labels  

---

## How to run

Install required libraries:

pip install torch torchvision numpy matplotlib pillow opencv-python

Run:

python src/gradcam_v3.py

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

- Managed full repository evolution across all versions (LIME → Grad-CAM → Final enhanced model)  
- Integrated final Grad-CAM + entropy + histogram pipeline  
- Ensured reproducible execution and clean project structure  
- Organized dataset, codebase, and output results  
- Maintained version control and submission-ready GitHub repository  

---

## Mathematical Concepts Used

- Probability (softmax output)  
- Monte Carlo Dropout for uncertainty estimation  
- Entropy for measuring prediction uncertainty  
- Gradient-based activation mapping (Grad-CAM)  
- Statistical distribution analysis (histogram of activations)  

---

## Limitations

- Grad-CAM resolution remains coarse and approximate  
- Monte Carlo Dropout provides only an approximate uncertainty estimate  
- Entropy depends on model probability calibration  
- Results depend on pretrained ResNet50 model  
- Explanations are not fully deterministic  

---

## Observations

We observed that Grad-CAM provides strong spatial interpretability, while entropy and Monte Carlo Dropout help quantify uncertainty more robustly.

The histogram of activations further shows how concentrated or distributed the model’s attention is on image regions.

This final version improves both interpretability and uncertainty estimation compared to previous versions, showing a clear evolution from LIME → Grad-CAM → enhanced Grad-CAM pipeline.