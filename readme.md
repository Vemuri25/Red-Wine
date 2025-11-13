# 🍷 Wine Quality ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP%20%2F%20DL-ee4c2c?logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![ONNX](https://img.shields.io/badge/ONNX-Model%20Export-purple)

A complete **regression + classification** project built using **PyTorch**, **XGBoost**, and **ONNX**, applied to the **Red Wine Quality Dataset**.

This repository demonstrates:

* 🔹 Data preprocessing & normalization
* 🔹 Exploratory Data Analysis (EDA)
* 🔹 Neural network classification models (MLP + Deep Learning)
* 🔹 Regression models (Linear, Deep, XGBoost)
* 🔹 ONNX export for deployment


## 📑 Table of Contents

* [1. Installation](#1-installation)
* [2. Dataset Summary](#2-dataset-summary)
* [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
* [4. Preprocessing](#4-preprocessing)
* [5. Classification Models](#5-classification-models)
* [6. Regression Models](#6-regression-models)
* [7. ONNX Export](#7-onnx-export)
* [8. Results](#8-results)
* [9. Conclusions](#9-conclusions)
* [10. Author](#10-author)

---

## 1. Installation

Install all required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn torch mlxtend xgboost onnxmltools onnxruntime skl2onnx
```

---

## 2. Dataset Summary

Source: **winequality-red.csv** (1599 samples, 12 columns)

**Feature Types:**

* Acidity levels
* Sulfur dioxide levels
* Density & pH
* Alcohol percentage
* **Quality score (target)**

Load the dataset:

```python
path_data = 'winequality-red.csv'
wine_df = pd.read_csv(path_data)
```

---

## 3. Exploratory Data Analysis

### 🔸 Correlation Heatmap

```python
cm = np.corrcoef(wine_df.values.T)
hm = heatmap(cm, row_names=wine_df.columns, column_names=wine_df.columns, figsize=(20,10))
plt.show()
```

### 🔸 Quality Distribution

```python
plt.hist(wine_df['quality'], bins='auto')
plt.title("Distribution of Wine Quality")
plt.show()
```

---

## 4. Preprocessing

### Steps Included:

✔ Train-test split (80/20)
✔ Normalization using training statistics
✔ Type conversion (`float32`, `int64`)
✔ Label mapping for classification

Normalization:

```python
x_means = X_train_tr.mean(0, keepdim=True)
x_stds = X_train_tr.std(0, keepdim=True) + 1e-4
```

---

## 5. Classification Models

Two PyTorch models are implemented.

### ### 🔹 **Model 1 — MLP_Net**

* Layers: `11 → 5 → 7`
* Activation: ReLU + Softmax
* Optimizer: Adam

### 🔹 **Model 2 — DL_Net (Deep Learning)**

* Layers: `11 → 15 → 9 → 7`
* Dropout used for regularization

Training loop:

```python
training_loop(N_Epochs, model, loss_fn, optimizer)
```

Evaluation metrics:

* Accuracy
* Confusion Matrix
* Weighted Precision, Recall, F1-score

---

## 6. Regression Models

Three different regression techniques were tested.

### 🔹 Linear Regression (PyTorch)

Simple linear layer.

### 🔹 Deep Learning Regression Model

* Two hidden layers
* ReLU activations

### 🔹 XGBoost Regressor

Best performing model.

```python
regressor = xgb.XGBRegressor(n_estimators=100, max_depth=3)
regressor.fit(X_train, y_train)
```

R² Score calculation:

```python
r2_score(y_test, y_pred)
```

---

## 7. ONNX Export

Export models for deployment.

### 🔹 PyTorch → ONNX

```python
torch.onnx.export(model, dummy_input, "DLnet_WineData.onnx")
```

### 🔹 XGBoost → ONNX

```python
onnx_model = onnxmltools.convert_xgboost(regressor, initial_types)
onnxmltools.utils.save_model(onnx_model, "winequality-red.onnx")
```

### 🔹 Running ONNX Model

```python
sess = rt.InferenceSession("winequality-red.onnx")
pred = sess.run([label], {input_name: X_test.astype(np.float32)})
```

---

## 8. Results

### 📊 Classification Results

| Model        | Accuracy | F1-Score |
| ------------ | -------- | -------- |
| MLP          | 0.57     | 0.51     |
| Deep Network | **0.59** | **0.53** |

### 📈 Regression Results

| Model             | R² Score           |
| ----------------- | ------------------ |
| Linear Regression | Low                |
| Deep Learning     | 0.046              |
| **XGBoost**       | ⭐ Best performance |

---

## 9. Conclusions

This ML pipeline demonstrates:

* Full PyTorch workflow (classification + regression)
* XGBoost outperforming neural networks in regression
* ONNX export suitable for production deployment
* Model performance comparison based on metrics

Potential next steps:

* Hyperparameter tuning
* Batch normalization
* Regularization improvements
* SHAP analysis for XGBoost feature importance

---

## 10. Author

**Vemuri Charan**
Applied Machine Learning — Fall 2025

---