# 🎓 Student Dropout Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=flat-square&logo=scikit-learn)

A machine learning system that predicts whether a student is at risk of dropping out, based on academic performance, personal background, and economic context.

---

## 📌 Project Overview

This project was built for a student dropout prediction challenge. It covers the full ML pipeline:

- **Exploratory Data Analysis** with correlation analysis and outlier detection
- **Feature Engineering** (grade trend, semester performance ratio)
- **Model Training & Comparison**: Logistic Regression, Random Forest, XGBoost, and a Soft Voting Ensemble
- **Model Explainability** using SHAP values
- **Interactive Web App** built with Streamlit for real-time predictions

---

## 🗂️ Project Structure

```
student-dropout-StrangerStrings/
│
├── app.py                  ← Streamlit web application
├── train.py                ← Model training script
├── requirements.txt        ← Python dependencies
│
├── models/                 ← Saved model artifacts (generated after training)
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── data/
│   └── fully_transformed_student_dataset.csv  ← Place dataset here
│
└── model1.ipynb            ← Original EDA & model exploration notebook
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

> If the real dataset CSV is not available in `data/`, the script automatically generates realistic synthetic data for demonstration purposes.

### 3. Run the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| **Interactive Input Form** | Sidebar sliders and toggles for all key student features |
| **Dropout Probability Gauge** | Animated dial showing risk percentage |
| **Risk Level Badge** | Low / Medium / High classification |
| **Grade Trend Analysis** | Auto-computed semester-over-semester improvement |
| **Risk Factor Breakdown** | Highlights which inputs are contributing to risk |
| **Feature Importance Chart** | Top 10 model features by XGBoost importance score |

---

## 🧠 Model Details

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, class-balanced |
| Random Forest | 100 estimators, max depth 15 |
| XGBoost | Tuned with GridSearchCV, **best performer** |
| Voting Ensemble | Soft voting across all three |

**Final model deployed in app**: XGBoost  
**Key metrics** (on synthetic demo data): ROC-AUC ≥ 0.88, F1 ≥ 0.84

---

## 📊 Input Features

| Feature | Type | Description |
|---|---|---|
| Age at Enrollment | Numerical | Student's age when they enrolled |
| Avg Grade 1st Sem | Numerical | Academic performance (0–20) |
| Avg Grade 2nd Sem | Numerical | Academic performance (0–20) |
| Scholarship Holder | Binary | Whether the student holds a scholarship |
| Debtor | Binary | Whether the student has unpaid tuition |
| Tuition Fees Up to Date | Binary | Payment status |
| Gender | Binary | Student's gender |
| Unemployment Rate | Numerical | Regional unemployment (%) |
| Inflation Rate | Numerical | Economic inflation (%) |
| GDP per Capita | Numerical | Economic context (USD) |
| Grade Trend *(engineered)* | Numerical | 2nd sem grade − 1st sem grade |
| Sem Performance Ratio *(engineered)* | Numerical | 2nd sem / 1st sem grade |

---

## 🧪 Running on Google Colab

To run training and exploration on Google Colab:

1. **Open the notebook**: Upload `model1.ipynb` to Colab or click the badge at the top of the notebook.
2. **Upload the dataset**: Run the first cell to upload `fully_transformed_student_dataset.csv`.
3. **Install extra dependencies** if prompted (XGBoost and SHAP are pre-installed on Colab).
4. **For `train.py` on Colab**: Run the cells below in a new notebook or code cell:
   ```python
   !pip install -q xgboost streamlit joblib plotly
   !python train.py
   ```
5. **For the Streamlit app on Colab**: Use `pyngrok` to tunnel the app:
   ```python
   !pip install -q pyngrok streamlit
   from pyngrok import ngrok
   import subprocess, threading
   def run_app():
       subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])
   threading.Thread(target=run_app, daemon=True).start()
   public_url = ngrok.connect(8501)
   print(f"App URL: {public_url}")
   ```

---

## 👩‍💻 Tech Stack

- **Python 3.9+**
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest
- **XGBoost** — gradient boosted trees
- **SHAP** — model explainability
- **Streamlit** — interactive web UI
- **Plotly** — gauge chart
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — EDA visualizations

---

*Built as part of a student dropout prediction challenge.*