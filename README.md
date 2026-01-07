# Multimodal" Price Prediction

This project demonstrates a **multimodal machine learning pipeline** that combines **tabular data** and **satellite/aerial images** to predict a continuous target variable (`price`). The workflow is implemented in a single Jupyter notebook and includes data loading, preprocessing, baseline modeling, and multimodal modeling.

---

## 📌 Project Overview

The goal of this project is to improve price prediction performance by leveraging:

* **Structured/tabular features** (CSV data)
* **Unstructured image data** (TIFF images)

A tabular-only baseline is first established, followed by a multimodal approach that incorporates image features extracted using deep learning.

---

## 🗂 Project Structure

```
.
├── multimodalcdc.ipynb   # Main notebook (end-to-end pipeline)
├── train.csv             # Training tabular data (not included)
├── test.csv              # Test tabular data (not included)
├── images_cdc/            # Training images (.tif)
└── images_cdc_test/       # Test images (.tif)
```

> **Note:** Dataset files and image directories are not included in this repository and must be provided separately.

---

## 🧪 Methodology

### 1. Data Loading

* Loads training and test CSV files
* Inspects columns and data types
* Associates each row with a corresponding image using a shared `id`

### 2. Image Handling

* Image paths are constructed from the `id` column
* Images are loaded from disk for feature extraction

### 3. Feature Engineering

* Drops non-feature columns such as `id`, `date`, and `image_path`
* Applies log transformation to the target variable (`price`)

### 4. Baseline Model (Tabular Only)

* Uses **XGBoost Regressor**
* Train/validation split
* Evaluation with RMSE and R²

### 5. Multimodal Modeling (Tabular + Images)

* Image features extracted using a CNN (e.g., pretrained backbone)
* Image embeddings concatenated with tabular features
* Final regression model trained on combined feature space

---

## 📊 Evaluation Metrics

* **Root Mean Squared Error (RMSE)**
* **R² Score**

Target variable is modeled in log-space using `log1p(price)` for stability.

---

## 🛠 Dependencies

Key libraries used in this project:

* Python 3.8+
* pandas
* numpy
* scikit-learn
* xgboost
* torch / torchvision (for image modeling)
* PIL / OpenCV (image loading)
* matplotlib / seaborn (optional visualization)

Example installation:

```bash
pip install pandas numpy scikit-learn xgboost torch torchvision pillow
```

---

## 🚀 How to Run

1. Clone this repository
2. Place `train.csv` and `test.csv` in the appropriate paths
3. Ensure image directories are correctly set in the notebook:

   ```python
   TRAIN_IMAGE_DIR = "path/to/images_cdc"
   TEST_IMAGE_DIR  = "path/to/images_cdc_test"
   ```
4. Open and run `multimodalcdc.ipynb` from top to bottom

---

## 📈 Results

* Tabular-only baseline provides a strong starting point
* Multimodal model aims to improve predictive performance by incorporating visual context
* Performance gains depend on image quality and alignment with tabular data

---

## 🔮 Future Improvements

* Hyperparameter tuning
* More advanced CNN backbones (EfficientNet, ResNet)
* Data augmentation for images
* Cross-validation
* Model ensembling

---

## 📄 License

This project is provided for educational and research purposes. Please ensure you have the right to use the dataset and images before training models.

---

## 🙌 Acknowledgements

* CDC dataset (if applicable)
* Open-source ML libraries and frameworks
