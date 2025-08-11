# **Melanoma Detection using Transformer-CNN Ensemble Framework**

## **Overview**

This repository contains the implementation and evaluation of an **ensemble learning framework** that integrates multiple **transformer-based architectures** (ViT, Swin Transformer, MaxViT, ConvNeXt) with a **CNN baseline (ResNet-50)** for **multi-class melanoma detection** from dermoscopic images.

The project was developed as part of research addressing the **urgent need for accurate, interpretable, and clinically deployable AI-based melanoma detection systems**, and forms the experimental backbone for the research paper:

---

## **Background & Motivation**

Melanoma is the deadliest form of skin cancer, responsible for **325,000 new cases and 57,000 deaths annually** worldwide. Early detection is critical — 5-year survival drops from **99%** (localized) to **27%** (metastatic). Current diagnostic practice relies heavily on **dermatologist visual assessment**, which suffers from **inter-observer variability (65–85% agreement)** and **specialist shortages** in many regions.

Deep learning methods have achieved dermatologist-level accuracy in binary melanoma classification, but face three key challenges:

1. **CNNs’ locality bias** limits capturing of global contextual lesion patterns.
2. **Single-model solutions** fail to leverage complementary strengths across architectures.
3. **Poor interpretability** reduces clinical trust and adoption.

This research bridges these gaps by:

* Combining CNN and Transformer models to exploit **both local and global feature learning**.
* Using **meta-learning ensembles** for optimal prediction fusion.
* Integrating **explainable AI** tailored for transformer architectures.

---

## **Research Objectives**

1. **Evaluate Transformer-based models** (ViT, Swin, MaxViT, ConvNeXt) and compare them against CNN baselines for multi-class melanoma classification.
2. **Develop a two-tier meta-learning ensemble framework** combining transformer and CNN architectures.
3. **Incorporate explainable AI** techniques (Grad-CAM for attention, LIME) for clinically interpretable predictions.
4. **Shift from binary to 6-class classification** to reflect real-world diagnostic workflows.

---

## **Novelty & Contributions**

1. **First ensemble framework** systematically merging multiple transformer models with CNNs for melanoma detection.
2. **Advanced meta-learning** using Random Forest as a meta-classifier, outperforming naïve voting/averaging.
3. **Transformer-specific explainability** methods mapping AI decisions to dermatological criteria.
4. **Clinically realistic multi-class setup**, improving real-world applicability.

---

## **Dataset**

We used the **ISIC 2020/2021 Challenge dataset**, consisting of dermoscopic images across six lesion categories (malignant & benign).

* **Source:** [ISIC Archive](https://www.isic-archive.com/)
* Preprocessing: resizing, normalization, stratified patient-level split, and augmentation.

---

## **Architectures Evaluated**

| Model         | Test Accuracy | Precision | Recall | F1-Score | AUC-ROC  | Parameters (M) |
| ------------- | ------------- | --------- | ------ | -------- | -------- | -------------- |
| **ViT**       | 0.8000        | 0.7900    | 0.7600 | 0.7700   | -        | 86             |
| **ConvNeXt**  | 0.8087        | 0.8100    | 0.7700 | 0.7800   | -        | 89             |
| **Swin**      | 0.6754        | 0.6637    | 0.6359 | 0.6289   | 0.9007\* | 88             |
| **MaxViT**    | 0.7478        | 0.7025    | 0.7352 | 0.7161   | 0.9045   | 119            |
| **ResNet-50** | 0.8107        | 0.8603    | 0.7712 | 0.8123   | 0.9729   | 27             |

---

## **Ensemble Results (ResNet-50 + Transformers)**

| Combination          | Avg Accuracy | Precision | Recall | F1-Score | Best Fold | Std Dev |
| -------------------- | ------------ | --------- | ------ | -------- | --------- | ------- |
| **ViT + ResNet-50**  | **0.9508**   | 0.9500    | 0.9400 | 0.9500   | 0.9609    | 0.0098  |
| ConvNeXt + ResNet-50 | 0.9404       | 0.9300    | 0.9200 | 0.9300   | 0.9457    | 0.0102  |
| MaxViT + ResNet-50   | 0.9165       | 0.9200    | 0.8800 | 0.9000   | 0.9239    | 0.0127  |
| Swin + ResNet-50     | 0.9017       | 0.8900    | 0.8700 | 0.8800   | 0.9109    | 0.0058  |

---

## **Folder Structure**

```
📂 melanoma-detection
 ├── 📓 Base_Model.ipynb          # ResNet-50 training & evaluation
 ├── 📓 New_MMelanoma_full.ipynb  # Ensemble framework & results
 ├── 📄 requirements.txt          # Dependencies
 └── 📄 README.md                 # Project documentation
```

---

## **Usage Instructions**

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/melanoma-detection.git
cd melanoma-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Dataset Access

* Update Google Drive paths in the notebooks to your own Drive mount point.

### 4️⃣ Train Base Model (ResNet-50)

```python
# Run all cells in Base_Model.ipynb
```

### 5️⃣ Train Ensemble Models

```python
# Run all cells in New_MMelanoma_full.ipynb
```

---

## **Reproducibility Notes**

* All experiments were conducted in **Google Colab Pro** with **NVIDIA Tesla V100 GPU**.
* **Patient-level K-fold CV** was used to avoid data leakage.

---

## **License**

This project is licensed under the MIT License.

---

