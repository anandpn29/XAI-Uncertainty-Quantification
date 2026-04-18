# XAI-Uncertainty-Quantification

Group implementation project for AI111 course

The aim of this project is to understand how explainable AI methods work using Grad-CAM and how reliable model predictions are under uncertainty using statistical and information-theoretic measures.

This final version introduces a fully enhanced explainability pipeline combining Grad-CAM, entropy-based uncertainty, and statistical significance testing using p-values for logits.

We used a pretrained ResNet50 model for image classification and applied Grad-CAM to visualize important image regions influencing predictions. Additionally, we introduced entropy and p-value analysis to quantify uncertainty and statistical significance of predictions.

---

## What we did

- Used ResNet50 for image classification  
- Extracted top-5 predicted classes  
- Computed softmax-based confidence scores  
- Performed z-score based p-value significance testing  
- Estimated uncertainty using entropy of probability distribution  
- Generated Grad-CAM heatmaps for visual explanations  
- Plotted activation distribution histogram  
- Combined all outputs into a unified visualization dashboard  

---

## Outputs

- Top-5 predicted classes with confidence scores  
- P-value significance for predictions  
- Entropy-based uncertainty score  
- Grad-CAM heatmap visualization (saved in results folder)  
- Combined analysis plot (saved in results folder)  
- Final explanation report printed in terminal  

---

## Project Structure

- src → contains final implementation code  
- images → input test images  
- results → output heatmaps and analysis plots  
- imagenet_classes.txt → class labels  

---

## How to run

Install required libraries:

pip install torch torchvision numpy matplotlib pillow opencv-python scipy

'or'

Clone the repository:
git clone <repo-link>

Install dependencies:
pip install -r requirements.txt

Run:

python src/gradcam_final.py

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

- Managed full project evolution across all versions (LIME → Grad-CAM → final statistical model)  
- Integrated final pipeline into a clean and reproducible GitHub structure  
- Ensured proper version control and structured repository organization  
- Maintained documentation and submission-ready formatting  
- Coordinated integration of all coding contributions  

---

## Mathematical Concepts Used

- Softmax probability estimation  
- Z-score based statistical testing  
- P-value significance testing  
- Entropy for uncertainty measurement  
- Gradient-based activation mapping (Grad-CAM)  
- Histogram-based distribution analysis  

---

## Limitations

- P-values are approximate and based on logit assumptions  
- Grad-CAM resolution is coarse  
- Entropy depends on model calibration  
- Monte Carlo effects are not fully Bayesian  
- Results depend on pretrained ResNet50 model  

---

## Observations

We observed that combining statistical significance (p-values) with entropy-based uncertainty provides a more robust interpretation of model confidence.

Grad-CAM highlights spatial reasoning, while histogram analysis shows how concentrated the model’s attention is.

This final version provides a comprehensive explainability pipeline combining visual, statistical, and probabilistic interpretability techniques.