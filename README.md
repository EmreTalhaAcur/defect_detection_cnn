# Manufacturing Defect Detection with CNN

This project introduces a **Convolutional Neural Network (CNN)** pipeline built with PyTorch to analyze RGB images from assembly line cameras and automatically classify components as Healthy (0) or Defective (1).

## ⚙️ Convolutional Layer Mechanics
The architecture processes visual features without losing spatial pixel relationships:
* **Convolutional Layers (`Conv2d`):** Moving kernel filters sweep across image matrices to extract edges, textures, and contrast gradients.
* **Max Pooling (`MaxPool2d`):** Downsamples feature maps by extracting the most dominant pixel values, drastically reducing computational overhead while retaining critical markers.
* **Backpropagation:** Propagates the computed error from the `BCEWithLogitsLoss` function backwards through the network via the Chain Rule to optimize convolutional kernels.

## 🦾 Robust Hardened Dataset Simulation
To push the model beyond simple position memorization and ensure deep visual understanding:
1. **Randomized Spatial Positioning:** Fault artifacts (defects) are dynamically placed in random quadrants of the 32x32 image grid.
2. **Dynamic Scaling:** Structural dimensions of defects vary fluidly across samples.
3. **Low-Contrast Blending:** The defect intensity was tuned down to blend smoothly into the background noise, forcing the CNN filters to undergo a visible learning curve (Convergence) starting around Epoch 4.

## 🛠️ Tech Stack
* Python 3
* PyTorch (Conv2d, MaxPool2d, Flatten)
* NumPy
