# 🧠 MindWell AI — Student Mental Health Score Predictor

> A full-stack machine learning application that estimates a student's mental health score from lifestyle, academic, and social-media usage patterns.

MindWell AI combines **machine learning, data preprocessing, FastAPI, and an interactive web interface** to demonstrate how predictive analytics can be used to analyze patterns associated with student well-being.

> ⚠️ **Disclaimer:** MindWell AI is an educational and predictive analytics project. It is **not a medical or psychological diagnostic tool** and should not be used to make clinical decisions.

---

## 📌 Overview

Student well-being can be influenced by several factors, including:

* 📱 Social-media usage
* 📚 Study habits
* 😴 Sleep duration
* 🏃 Physical activity
* 📞 Phone usage
* 😟 Stress levels
* 🎓 Education and demographic information

MindWell AI accepts these factors through a web form and uses a trained **Random Forest regression model** to generate an estimated **Mental Health Score between 1 and 10**.

The prediction is then displayed through an intuitive score visualization to make the result easier to understand.

---

## ✨ Key Features

* 🤖 Machine-learning-based mental health score prediction
* 🌐 Full-stack web application
* ⚡ FastAPI REST API
* 📊 Random Forest regression model
* 🧹 Automated data preprocessing pipeline
* 🔤 Categorical feature encoding
* 📈 Numerical feature scaling and transformation
* 🌍 Country grouping for high-cardinality categorical data
* ✅ Pydantic-based API input validation
* 📋 Interactive API documentation with Swagger
* 🎨 Responsive frontend interface
* ⚠️ Prediction interpretation and disclaimer
* 💾 Serialized ML pipeline using Joblib

---

## 🗂️ Dataset

The project uses the **Student Social Media and Mental Health Impact** dataset.

### Dataset Summary

| Property               | Details               |
| ---------------------- | --------------------- |
| Original rows          | 5,000                 |
| Columns                | 13                    |
| Duplicate rows removed | 2                     |
| Final rows             | 4,998                 |
| Target variable        | `Mental_Health_Score` |
| Target range           | 3.6 – 9.4             |

### Data Cleaning

The following preprocessing steps were performed:

1. Checked the dataset for missing values.
2. Removed 2 duplicate records.
3. Identified 22 negative values in `Physical_Activity_Hours`.
4. Treated the negative activity values as invalid and clipped them to `0`.
5. Reduced the number of country categories to make the categorical feature more manageable.

### Input Features

The model uses 12 predictive inputs:

#### 👤 Personal Information

* Age
* Gender
* Education Level
* Country

#### 📱 Social Media & Phone Usage

* Most Used Platform
* Purpose of Use
* Daily Usage Hours
* Daily Phone Unlocks

#### 🏃 Daily Routine & Well-being

* Study Hours
* Physical Activity Hours
* Sleep Hours
* Stress Level

### Target

```text
Mental_Health_Score
```

The target represents a numerical score derived from the dataset.

---

## 🌍 Country Feature Engineering

The dataset contains **111 countries**, creating a relatively high-cardinality categorical feature.

Instead of one-hot encoding all countries individually, the preprocessing workflow:

* Retains the **top 10 countries**
* Groups all remaining countries into:

```text
Other
```

The resulting feature is:

```text
Group_Country
```

The model uses `Group_Country` rather than the original `Country` column.

This reduces dimensionality while preserving information about the most frequently represented countries.

---

# 🤖 Machine Learning Pipeline

The project uses a complete Scikit-learn preprocessing and prediction pipeline.

### Data Split

```text
70% → Training
30% → Testing
```

Configuration:

```python
random_state = 42
```

Dataset split:

* Training: **3,498 rows**
* Testing: **1,500 rows**

---

## ⚙️ Preprocessing

A Scikit-learn `ColumnTransformer` applies different preprocessing operations according to feature type.

### Numerical Features

Numerical variables are scaled before being passed to the model.

`Study_Hours` receives an additional transformation because its distribution was identified as skewed.

### Ordinal Feature

`Stress_Level` contains ordered categories:

```text
Low
Medium
High
Very High
```

The preprocessing pipeline treats this variable as ordinal so that its ordering is preserved.

### Categorical Features

The following categorical variables are one-hot encoded:

* Gender
* Education Level
* Most Used Platform
* Purpose of Use
* Group_Country

After preprocessing, the model receives approximately:

```text
38 features
```

---

# 🌲 Model Development

Several regression models were evaluated.

| Model               |   Test R² | Train R² |       MAE |      RMSE |
| ------------------- | --------: | -------: | --------: | --------: |
| Linear Regression   |     0.740 |    0.724 |     0.536 |     0.676 |
| Random Forest       |     0.878 |    0.981 |     0.347 |     0.464 |
| Tuned Random Forest | **0.886** |    0.983 | **0.336** | **0.447** |

### Evaluation Metrics

**R² — Coefficient of Determination**

Measures how much variation in the target is explained by the model.

**MAE — Mean Absolute Error**

Measures the average absolute difference between predicted and actual scores.

**RMSE — Root Mean Squared Error**

Measures prediction error while giving greater weight to larger errors.

---

## 🔧 Hyperparameter Tuning

The Random Forest model was optimized using:

```text
RandomizedSearchCV
```

Configuration:

* 50 randomized parameter combinations
* 5-fold cross-validation

The best configuration identified during tuning included:

```text
n_estimators = 1000
max_depth = 30
max_features = 0.5
```

### Important Implementation Note

The current application uses the serialized model saved from the notebook:

```text
rf_pipeline
```

This is the **default Random Forest pipeline**, rather than the tuned Random Forest pipeline.

Therefore, the application should not be described as serving the tuned model unless the saved model is replaced with the tuned pipeline.

---

# 🏗️ Application Architecture

```text
┌──────────────────────────┐
│      User / Browser      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     HTML / CSS / JS      │
│      Frontend UI         │
└────────────┬─────────────┘
             │ POST /predict
             ▼
┌──────────────────────────┐
│        FastAPI           │
│     Backend / API        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Pydantic Input Validation│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Scikit-learn ML Pipeline │
│                          │
│ Preprocessing → Random   │
│ Forest → Prediction      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Mental Health Score      │
│       1 – 10             │
└──────────────────────────┘
```

---

# 🚀 Backend API

The backend is built using **FastAPI**.

## `POST /predict`

Accepts student information and returns the predicted mental health score.

Example response:

```json
{
  "mental_health_score": 7.42
}
```

The prediction is rounded to two decimal places.

---

## `GET /api/meta`

Provides metadata required by the frontend, including:

* Model information
* Model performance statistics
* Available country values

---

## `GET /docs`

FastAPI automatically provides interactive Swagger API documentation.

When running locally, it can be accessed at:

```text
http://localhost:8000/docs
```

---

# 🎨 Frontend

The frontend is built using:

* HTML
* CSS
* JavaScript

The interface provides:

1. Student information form
2. Lifestyle and social-media inputs
3. API request to the backend
4. Prediction result
5. Visual score scale
6. Error handling
7. Score interpretation guidance
8. Responsible-use disclaimer

---

# 🛠️ Tech Stack

### Programming

* Python
* JavaScript
* HTML
* CSS

### Machine Learning

* Pandas
* NumPy
* Scikit-learn
* Joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Development

* Jupyter Notebook
* Git
* GitHub

---

# 📁 Suggested Project Structure

```text
MindWell-AI/
│
├── backend/
│   ├── main.py
│   └── model/
│       └── rf_pipeline.pkl
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── notebooks/
│   └── model_training.ipynb
│
├── data/
│   └── student_mental_health.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 💻 Installation & Setup

## 1. Clone the repository

```bash
git clone https://github.com/your-username/MindWell-AI.git
cd MindWell-AI
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project currently requires the compatible Scikit-learn version used to serialize the model:

```text
scikit-learn==1.6.1
```

## 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

## 5. Open the application

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

---

# 🐛 Important Issues Resolved

Several integration issues were identified and fixed during development.

### 1. Incorrect Country Field

The original backend sent:

```text
Country
Grouped_country
```

while the trained model expected:

```text
Group_Country
```

The backend was updated to construct the correct feature name.

### 2. Model Filename Mismatch

The backend referenced a model filename that did not match the actual serialized model.

The model loading configuration was corrected.

### 3. Scikit-learn Compatibility

The serialized `.pkl` model depends on:

```text
scikit-learn 1.6.1
```

The dependency was therefore pinned to maintain compatibility.

### 4. Frontend Static Files

The CSS and JavaScript resources initially returned `404` errors.

The FastAPI static-file configuration was corrected so that the frontend resources are served properly.

---

# ⚠️ Limitations & Responsible Use

### 1. Model Overfitting

The Random Forest model achieves approximately:

```text
Training R²: 0.98
Testing R²: 0.88
```

The difference indicates that the model fits the training data considerably more closely than unseen data.

For reporting model performance, the **test-set metrics** are more relevant than the training metrics.

---

### 2. Limited Age Range

The training dataset contains students approximately between:

```text
18 – 24 years
```

However, the current application accepts a broader age range.

Predictions outside the training distribution should therefore be treated cautiously.

A future version should restrict the input range to the population represented by the training data or retrain the model using a broader dataset.

---

### 3. Not a Clinical Tool

MindWell AI does **not** diagnose:

* Depression
* Anxiety
* Stress disorders
* Other mental-health conditions

The output is simply a machine-learning prediction based on patterns in the training dataset.

It should not replace professional medical or psychological assessment.

---

### 4. Correlation Does Not Imply Causation

A relationship learned by the model does not prove that one lifestyle factor causes a change in mental health.

For example, a model association between social-media usage and the target score does not establish that changing social-media usage will necessarily change someone's well-being.

---

### 5. Dataset Limitations

The dataset's relatively strong predictive performance should be interpreted in the context of the dataset itself.

The relationships may not generalize to real-world populations, different age groups, countries, or clinical settings.

---

# 🔮 Future Improvements

Potential future improvements include:

* [ ] Replace the saved default model with the tuned Random Forest pipeline
* [ ] Add model explainability using SHAP
* [ ] Add prediction confidence/uncertainty information
* [ ] Improve validation with external datasets
* [ ] Restrict inputs to the model's training distribution
* [ ] Add authentication and user accounts
* [ ] Add prediction history
* [ ] Build a personalized dashboard
* [ ] Add automated model monitoring
* [ ] Add Docker deployment
* [ ] Deploy the application to a cloud platform
* [ ] Add unit and API tests
* [ ] Improve accessibility and responsive design

---

# 📊 Project Highlights

| Component           | Implementation               |
| ------------------- | ---------------------------- |
| Problem Type        | Regression                   |
| Target              | Mental Health Score          |
| Best Tested Model   | Tuned Random Forest          |
| Application Model   | Saved Random Forest Pipeline |
| Test R²             | 0.878 for saved/default RF   |
| MAE                 | 0.347 for saved/default RF   |
| RMSE                | 0.464 for saved/default RF   |
| Backend             | FastAPI                      |
| Frontend            | HTML, CSS, JavaScript        |
| ML Framework        | Scikit-learn                 |
| Model Serialization | Joblib                       |
| API Documentation   | Swagger / OpenAPI            |

---

# 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Exploratory Data Analysis
* Data cleaning
* Outlier handling
* Feature engineering
* Categorical encoding
* Feature transformation
* Feature scaling
* Regression algorithms
* Random Forest
* Hyperparameter tuning
* Cross-validation
* Model evaluation
* ML pipeline construction
* Model serialization
* REST API development
* FastAPI
* Frontend-backend integration
* Responsible ML communication

---

# 👨‍💻 Author

**Viraj Chavan**

MSc Mathematics & Data Science
Aspiring AI / Machine Learning Engineer

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 📊 Data Science
* 🔬 Deep Learning
* 🐍 Python
* 🚀 AI-powered applications

---

## ⭐ Project Purpose

MindWell AI was developed as a learning project to demonstrate the complete journey from:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Preprocessing
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
FastAPI Backend
     ↓
Web Application
     ↓
Prediction
```

The primary goal is to demonstrate how a machine-learning model can be transformed into a usable end-to-end application while communicating its limitations responsibly.
