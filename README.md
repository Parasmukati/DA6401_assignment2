# DA6401 Assignment 2: CNNs from Scratch & Transfer Learning with iNaturalist

This repository contains the solution to **Assignment 2** for the course **DA6401**, focused on training convolutional neural networks from scratch and fine-tuning pre-trained models using the [iNaturalist dataset](https://www.kaggle.com/competitions/inaturalist-2021-fgvc8/data).

## 📁 Repository Structure

```
.
├── PartA.ipynb            # CNN model trained from scratch + W&B sweeps + visualizations
├── PartB.ipynb            # Fine-tuning of a pre-trained model on iNaturalist
├── data/                  # iNaturalist train/test folders (not included in repo)
├── wandb/                 # W&B local logs
├── README.md              
```

---

## 🚀 Assignment Overview

### 📌 **Part A: CNN from Scratch**

#### ✅ Tasks Completed:
- Built a configurable CNN with 5 Conv-Activation-Pool blocks.
- Dense layer + output layer with 101 classes.
- Made kernel sizes, activation functions, and dense layer size configurable.
- Computed total parameters and computations for custom configurations.
- Split train/validation data with class balance.
- Implemented extensive hyperparameter tuning via `wandb` sweeps.
- Evaluated best model on test data.
- Visualized:
  - Model predictions on test set (10×3 grid)
  - First-layer filters (optional)
  - Guided backpropagation (optional)

#### 🔍 Key Hyperparameters Explored:
- Number of filters: 32, 64, 128
- Activation functions: ReLU, GELU, SiLU
- Filter scaling: constant, doubling, halving
- Dropout rates: 0.2, 0.3
- Batch normalization: enabled/disabled
- Data augmentation: enabled/disabled

#### 📊 W&B Visualizations:
- Accuracy vs Created plot
- Parallel coordinates plot
- Correlation summary

#### 📎 Report Links:
- 🧠 Part A W&B Report: [W&B Project](https://api.wandb.ai/links/na21b051-indian-institute-of-technology-madras/f78cdtsl)
- 🧾 GitHub Code: [Part A Notebook](https://github.com/Parasmukati/DA6401_assignment2/blob/main/PartA.ipynb)

---

### 📌 **Part B: Transfer Learning**

#### ✅ Tasks Completed:
- Loaded pre-trained model (e.g., ResNet50 from `torchvision.models`)
- Resized iNaturalist images to match ImageNet input dimensions.
- Replaced final classification layer with 101-class output layer.
- Explored fine-tuning strategies:
  - Freezing all layers except last
  - Freezing up to k layers
  - Full fine-tuning

#### 🧪 Strategies Tried:
- Freeze all but last linear layer
- Freeze all conv layers and retrain classifier head
- Freeze first N blocks and fine-tune deeper layers

#### 📌 Observations:
- Fine-tuning resulted in faster convergence and higher validation accuracy
- Smaller training time compared to training from scratch
- Transfer learning helped generalize better with limited data

#### 📎 Report Links:
- 🧠 Part B W&B Report: [W&B Project](https://api.wandb.ai/links/na21b051-indian-institute-of-technology-madras/f78cdtsl)
- 🧾 GitHub Code: [Part B Notebook](https://github.com/Parasmukati/DA6401_assignment2/blob/main/PartB.ipynb)

---

## ⚙️ How to Run

### 🔧 Requirements
- Python 3.8+
- PyTorch
- torchvision
- wandb

### 🏃 Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/da6401_assignment2.git
cd da6401_assignment2

# Install dependencies
pip install torch torchvision wandb
```

### 📟 Running the notebooks
I used Google Colab with GPU enabled.

---
