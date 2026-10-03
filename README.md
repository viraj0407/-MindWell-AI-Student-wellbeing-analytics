# MindWell AI
# Student Mental Health Score Predictor
## What it is
A full-stack machine-learning web app. A user enters a student's profile, social-media habits and daily routine. A trained model returns an estimated mental health score from 1 to 10, and the page shows it on a color scale.

## Dataset
- **Source file:** "Student Social Media and Mental Health Impact", 5,000 rows and 13 columns.
- **Cleaning:** it had no missing values and 2 duplicate rows, which I removed (4,998 rows remain). Negative `Physical_Activity_Hours` values (22 rows flagged as outliers) were clipped to 0.
- **Inputs (12):**
  - Personal: age, gender, education level, country.
  - Social media: most-used platform, purpose of use, daily usage hours, daily phone unlocks.
  - Routine: study hours, physical activity hours, sleep hours, stress level.
- **Target:** `Mental_Health_Score`, ranging from 3.6 to 9.4.
- **Country handling:** the data has 111 countries, so you kept the top 10 as-is and grouped the rest into "Other". The result is a `Group_Country` column, which the model uses instead of `Country`.

## Machine learning pipeline
- **Split:** 70% train (3,498 rows) and 30% test (1,500 rows), with `random_state=42`.
- **Preprocessing:** a scikit-learn `ColumnTransformer` that handles each feature type separately.
  - `Study_Hours` was skewed, so it gets a transform.
  - `Stress_Level` is ordinal (Low, Medium, High, Very High).
  - Gender, education, platform, purpose and grouped country are one-hot encoded.
  - The other numeric columns are scaled.
  - The result is 38 features.
- **Models compared:**

| Model | Test R² | Train R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.740 | 0.724 | 0.536 | 0.676 |
| Random Forest (default) | 0.878 | 0.981 | 0.347 | 0.464 |
| Random Forest (tuned) | 0.886 | 0.983 | 0.336 | 0.447 |

- **Tuning:** `RandomizedSearchCV` with 50 iterations and 5-fold cross-validation. The best settings were 1000 trees, max depth 30, and `max_features=0.5`.
- **Saved model:** the notebook saves the default Random Forest (`rf_pipeline`), not the tuned one. I used the saved file as-is.

## Application architecture
- **Backend (FastAPI):**
  - `POST /predict` validates input with Pydantic, rebuilds the model's input row, and returns the score rounded to 2 decimals.
  - `GET /api/meta` supplies model stats and the country list to the page.
  - `GET /docs` gives automatic interactive API docs.
  - It also serves the frontend files.
- **Frontend (HTML, CSS, JS):** a form, a result card with a score scale, error handling, and a "how to interpret this" note with a disclaimer.
- **Stack:** Python, pandas, scikit-learn, joblib, FastAPI, Uvicorn.

## Problems I fixed along the way
1. The original `main.py` sent `Country` and `Grouped_country`, but the model expects `Group_Country`, so every request would have failed.
2. The model filename in the code didn't match your actual file.
3. The `.pkl` only loads on scikit-learn 1.6.1, so I pinned that version.
4. The CSS and JS links returned 404, which I fixed in `main.py`.

## Limitations to be honest about
- **Overfitting gap:** the Random Forest scores 0.98 on training data versus 0.88 on test data, so it fits the training data much more tightly than new data. Test R² is the number to quote.
- **Narrow age range:** the training data covers ages 18 to 24, but the app accepts 10 to 100. Predictions outside 18 to 24 are extrapolation. Consider limiting the age field to 18–24.
- **Not a diagnosis:** the score is a statistical pattern from survey data. The data's clean, high-R² relationships may mean it is partly synthetic or derived, so treat it as a demo of the workflow rather than a clinical tool.
- **Correlation, not causation:** the app doesn't claim that changing a habit changes wellbeing.
