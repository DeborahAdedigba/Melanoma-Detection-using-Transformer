# Multi-Class Melanoma Classification with Transformer-CNN Ensembles

This repository contains Jupyter notebooks for training and evaluating deep learning models on the **PH² dataset** for **multi-class melanoma classification**.  
The project evaluates **transformer-based** and **CNN-based** architectures and boosts performance through **ensemble learning** with a **Base Model (ResNet-50)**.

---

## 📌 Project Overview

Melanoma is a severe type of skin cancer that benefits greatly from early detection. This project explores multiple state-of-the-art models:

- **Vision Transformer (ViT)**
- **ConvNeXt**
- **Swin Transformer**
- **MaxViT**
- **Base Model (ResNet-50)**

Each transformer-based model is trained individually and then combined with the **Base Model** using an **ensemble approach** (probability averaging). The goal is to assess performance gains from combining architectures with complementary strengths.

---

## 🗂 Repository Structure

```

.
├── New\_MMelanoma\_full.ipynb         # Contains ViT, ConvNeXt, Swin, MaxViT training & ensemble evaluation
├── Base\_Model.ipynb                 # ResNet-50 training & evaluation (baseline model)
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies

````

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
````

**Main libraries:**

* TensorFlow / Keras
* timm (PyTorch Image Models, for transformers)
* NumPy, Pandas
* OpenCV
* Matplotlib, Seaborn
* scikit-learn

---

## 📂 Dataset

* **Dataset**: PH² Dataset ([http://www.fc.up.pt/addi/ph2%20database.html](http://www.fc.up.pt/addi/ph2%20database.html))
* Contains dermoscopic images with metadata for multiple lesion classes.
* Store dataset in Google Drive and update file paths in the notebooks.

---

## 🚀 Usage

### 1️⃣ Train Baseline CNN (Base Model - ResNet-50)

```bash
jupyter notebook Base_Model.ipynb
```

### 2️⃣ Train Transformer Models

```bash
jupyter notebook New_MMelanoma_full.ipynb
```

This notebook trains:

* Vision Transformer (ViT)
* ConvNeXt
* Swin Transformer
* MaxViT

It also evaluates **ensembles** of each transformer with the Base Model (ResNet-50).

---

## 📊 Results

### **Individual Models**

| Model                  | Test Accuracy | Precision | Recall | F1-Score |  AUC-ROC | Parameters (M) |
| ---------------------- | ------------: | --------: | -----: | -------: | -------: | -------------: |
| Vision Transformer     |        0.8000 |    0.7900 | 0.7600 |   0.7700 |        - |             86 |
| ConvNeXt               |        0.8087 |    0.8100 | 0.7700 |   0.7800 |        - |             89 |
| Swin Transformer       |        0.6754 |    0.6637 | 0.6359 |   0.6289 | 0.9007\* |             88 |
| MaxViT                 |        0.7478 |    0.7025 | 0.7352 |   0.7161 |   0.9045 |            119 |
| Base Model (ResNet-50) |        0.8107 |    0.8603 | 0.7712 |   0.8123 |   0.9729 |             27 |

\* AUC-ROC values for ViT and ConvNeXt were not computed in this run.

---

### **Ensemble Combinations with Base Model**

| Ensemble              | Avg Accuracy | Precision | Recall | F1-Score | Best Fold Accuracy | Std Dev |
| --------------------- | -----------: | --------: | -----: | -------: | -----------------: | ------: |
| ViT + Base Model      |   **0.9508** |    0.9500 | 0.9400 |   0.9500 |             0.9609 |  0.0098 |
| ConvNeXt + Base Model |       0.9404 |    0.9300 | 0.9200 |   0.9300 |             0.9457 |  0.0102 |
| MaxViT + Base Model   |       0.9165 |    0.9200 | 0.8800 |   0.9000 |             0.9239 |  0.0127 |
| Swin + Base Model     |       0.9017 |    0.8900 | 0.8700 |   0.8800 |             0.9109 |  0.0058 |

**Key Finding:** All ensembles outperformed the best single model, with **ViT + Base Model** achieving the highest accuracy.

---

## ⚠️ Notes for Reuse

* **Google Drive Paths**: The notebooks use the author’s Google Drive structure (`/content/drive/MyDrive/...`). Update these paths for your environment.
* **Colab GPU Runtime**: Recommended for training speed.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Deborah Adedigba**
Applied AI and Data Science
