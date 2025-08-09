# Brain_Tumor_Detection_CNN
<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/5b2dd6e5-3df5-427f-97f6-3208b69b1a2e" />


### This project was a great learning experience to explore the world of **medical imaging** using CNN which is a deep learning model used to analyze images to detect abnormalities and ,in my case, tumors in MRI brain images. 


# Dataset
Data: [Brain MRI Images for Brain Tumor Detection — Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
, due to the small amount of data - tumorous (156) and non tumorous (98)- and to avoid overfitting I used Data augmentation with Total number of samples: 2065, Positive samples: 1085 (52.54%), Negative samples: 980 (47.46%), to create synthetic images with 
-Rotation
-Horizontal/vertical flips
-Width/height shifts
-Brightness adjustments

# Loading images 
Images are read in grayscale using OpenCV (cv2).

Resized to 150×150 pixels.

Normalized to a range of 0–1 for faster model convergence.

# Model architecture
Two approaches are implemented:

1. I used a custom CNN model Conv2D + MaxPooling2D layers:
32 filters → 64 filters → 128 filters.
Dense Layers:
128 neurons (ReLU activation) + Dropout (0.5) to prevent overfitting.
Final layer: 1 neuron (sigmoid) for binary classification.
Loss Function: Binary Crossentropy.
Optimizer: Adam.
However, the Final Training Accuracy was 80%
So to increase the accuracy I used
2. Transfer Learning with VGG16 in which:
Base model: VGG16 (pre-trained on ImageNet).
Converts grayscale MRI to RGB for VGG16 input.
Freezes convolutional layers to preserve pre-trained weights.
Adds a Flatten → Dense(128) → Dropout → Dense(1, sigmoid) head.
Loss Function: Binary Crossentropy.
Optimizer: Adam with learning rate 1e-4.

# Results
After using VGG16

Final Training Accuracy: 96.04%

Final Validation Accuracy: 84.31%

<img width="971" height="404" alt="image" src="https://github.com/user-attachments/assets/5a65e744-ddab-465d-96fe-8fe9f57baa65" />



<img width="969" height="364" alt="image" src="https://github.com/user-attachments/assets/262980c2-0ac2-4737-8269-08f0089afd1e" />

# Summary 
-The CNN successfully distinguishes between tumorous and non-tumorous MRI scans.

-Transfer learning with VGG16 boosted performance compared to training from scratch.

-Data augmentation improved robustness by making the model less sensitive to MRI variations.



