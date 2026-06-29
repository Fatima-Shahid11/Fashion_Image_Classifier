# Fashion-MNIST CNN Classifier

A Convolutional Neural Network built from scratch in PyTorch to classify images of clothing items into 10 categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

This project demonstrates the core CNN concepts, convolutional layers, pooling, fully connected layers, training loops, and evaluation — applied end-to-end on a real dataset.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | ~93.5% |
| Training epochs | 5 |
| Total parameters | 421,642 |

## Classification

<img width="1173" height="497" alt="image" src="https://github.com/user-attachments/assets/27162278-67e1-42bf-846e-dd1b5f55985d" />



## Confusion Matrix

<img width="785" height="624" alt="image" src="https://github.com/user-attachments/assets/0a230c6b-d404-41b5-85c1-44e3b1e019f6" />

The model converges quickly and achieves strong performance on a standard CV benchmark, with most remaining errors coming from visually similar classes (e.g., shirt vs T-shirt vs pullover).

---

## Architecture

**Design choices:**
- 3×3 kernels with `padding=1` to preserve spatial dimensions in convolutions
- MaxPool layers handle downsampling (28 → 14 → 7)
- Channel count grows (1 → 32 → 64) as spatial size shrinks
- Two conv blocks are enough for 28×28 inputs

---

## Concepts demonstrated

- 2D convolutions and feature maps
- ReLU activation and non-linearity
- Max pooling for downsampling and translation tolerance
- Flatten + fully connected layers for classification
- Cross-entropy loss for multi-class classification
- Adam optimizer with backpropagation
- Train/test split and the difference between training and evaluation modes
- Confusion matrix for error analysis

---

## Tech stack

- **Python**, **PyTorch**
- **matplotlib**, **seaborn** for visualization
- **scikit-learn** for confusion matrix and metrics
- **Google Colab** for training environment

---

## How to run

Open the notebook in Colab and run all cells:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Hlvb40nin4WPuQhgSKVTVF66gLniCt_a?usp=sharing)

Or locally:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn
jupyter notebook notebook.ipynb
```

---

## What's next

This project is the foundation for more complex applications:
- Apply the same architecture to industrial sensor data (vibration signals → spectrograms → CNN) for predictive maintenance
- Replace the custom CNN with a pretrained model (transfer learning with ResNet) for higher accuracy
- Deploy the model as a FastAPI service for live inference
