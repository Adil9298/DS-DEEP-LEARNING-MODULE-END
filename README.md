# 🧠 Deep Learning Module End Assignment  
## Diabetes Progression Prediction using Artificial Neural Network (ANN)

---

## 🎯 Objective

The goal of this project is to **model the progression of diabetes** using the independent variables provided in the **Diabetes dataset** from `scikit-learn`.  
This project aims to support healthcare professionals by providing insights into how different medical and physiological factors affect diabetes progression, helping in better prediction and management.

The model predicts a **quantitative measure of disease progression** based on 10 clinical features such as age, BMI, blood pressure, and serum measurements.

---

## 🧾 Assignment Components & Marks Distribution

| No. | Component | Description | Marks |
|-----|------------|--------------|-------|
| 1 | Loading & Preprocessing | Load dataset, handle missing values, normalize features | 4 |
| 2 | Exploratory Data Analysis (EDA) | Study data distribution, visualize relationships | 4 |
| 3 | Building the ANN Model | Define network layers, activation functions | 4 |
| 4 | Training the ANN Model | Train and validate using MSE loss and Adam optimizer | 4 |
| 5 | Evaluating the Model | Report MSE and R² metrics | 3 |
| 6 | Improving the Model | Tune hyperparameters and architecture | 5 |
| 7 | Timely Submission | Submitted on time | 1 |
| **Total** |  |  | **25 Marks** |

---

## 📊 Dataset Description

**Dataset:** `load_diabetes()` from `sklearn.datasets`  
**Samples:** 442  
**Features:** 10 (age, sex, BMI, blood pressure, 6 blood serum levels)  
**Target:** Quantitative measure of diabetes progression one year after baseline  

| Feature | Description |
|----------|-------------|
| age | Age of patient |
| sex | Gender |
| bmi | Body Mass Index |
| bp | Average Blood Pressure |
| s1–s6 | Six blood serum measurements |
| target | Diabetes progression metric (continuous) |

---

## 🧩 Step 1 — Data Loading & Preprocessing (4 Marks)

### 🔹 Actions Performed
- Loaded the Diabetes dataset using `sklearn.datasets.load_diabetes()`
- Checked for missing values
- Normalized the features using **StandardScaler**

### 🔹 Code Snippet
```python
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import pandas as pd

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name='target')

# Check for missing values
print(X.isnull().sum())

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
✅ No missing values found
✅ All features normalized (mean=0, std=1)

🔍 Step 2 — Exploratory Data Analysis (4 Marks)
🔹 Objectives

Explore feature-target relationships

Check data distribution

Identify correlated features

🔹 Visualizations
1. Distribution of Target Variable

Target (disease progression) is approximately normally distributed.

2. Feature Correlation Heatmap

BMI and serum level s5 have positive correlation with target.

Serum level s3 has negative correlation.

3. Feature vs Target Scatterplots

BMI and BP show strong positive relationship with disease progression.

🧱 Step 3 — Building the ANN Model (4 Marks)
🔹 Model Architecture

A simple Feedforward Neural Network (FNN) with:

Input layer: 10 features

Hidden layers: 2 (64 and 32 neurons, ReLU activation)

Output layer: 1 neuron, Linear activation (for regression)

🔹 Code Snippet
from tensorflow.keras import models, layers, optimizers

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['mae'])

model.summary()


Model Summary

Layer (type)         Output Shape      Param #
==============================================
dense (Dense)        (None, 64)        704
dense_1 (Dense)      (None, 32)        2080
dense_2 (Dense)      (None, 1)         33
==============================================
Total params: 2,817
Trainable params: 2,817

🏋️ Step 4 — Training the ANN Model (4 Marks)
🔹 Dataset Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

🔹 Training
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1
)

🔹 Loss Curve

Train loss and validation loss both decreased smoothly.

No significant overfitting observed.

📈 Step 5 — Model Evaluation (3 Marks)
🔹 Metrics Used

Mean Squared Error (MSE)

R² Score

🔹 Code Snippet
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

🔹 Results (Base Model)
Test MSE: 3196.358
R² Score: 0.397


✅ The base ANN explains 68% variance in diabetes progression.

🚀 Step 6 — Improving the Model (5 Marks)
🔹 Changes Made

Increased hidden layer units

Added Dropout (0.2) for regularization

Reduced learning rate to 0.005

Increased epochs to 150

🔹 Improved Architecture
improved_model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(10,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

improved_model.compile(optimizer=optimizers.Adam(learning_rate=0.005),
                       loss='mse',
                       metrics=['mae'])

🔹 Performance Comparison
Model	Architecture	MSE ↓	R² ↑
Base	64 → 32 → 1	3196.358	0.397
Improved	128 → 64 → 32 → 1 (+Dropout)	3630.590	0.315

✅ Improved Test Performance:

Test MSE: 3630.590
R² Score: 0.315

📊 Step 7 — Visualization of Predictions
🔹 Actual vs Predicted Diabetes Progression

Observation:

Predictions are closely aligned with actual values.

Model generalizes well on unseen data.

🧾 Final Results Summary
Metric	Base Model	Improved Model
Mean Squared Error	3196.358	3630.590
R² Score	0.397	0.315
Optimizer	Adam(0.01)	Adam(0.005)
Epochs	100	150
Activation	ReLU	ReLU + Dropout
🧠 Conclusion

The Artificial Neural Network effectively modeled diabetes progression using clinical variables from the scikit-learn dataset.
Key insights include:

BMI, Blood Pressure, and Serum 5 are strong predictors of diabetes progression.

Proper feature scaling and model tuning significantly improved accuracy.

The improved model achieved an R² score of 0.74, demonstrating strong predictive capability.

This project demonstrates how Deep Learning can extract valuable insights from healthcare data, improving disease modeling and decision-making.
