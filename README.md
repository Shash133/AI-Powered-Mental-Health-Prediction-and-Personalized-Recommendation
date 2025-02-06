# Mental Health Treatment Prediction Model

## 🔧 Data Preprocessing

### Handling Missing Data
- **Default Values:** Missing values replaced with defaults based on column type:
  - `int`: `0`
  - `str`: `"unknown"`
  - `float`: `0.0`
- **Columns Dropped:** `comments`, `state`, and `Timestamp` (deemed irrelevant for prediction).

### Gender Normalization
- Variations (e.g., "M", "Male", "F", "Female") mapped to standardized categories:
  - `male`, `female`, `trans`.

### Age Normalization
- Values clamped to **18–120** range. Outliers replaced with the median age.

### Label Encoding
- Categorical columns (e.g., `gender`, `family_history`) converted to numerical values.

### Feature Engineering
- **Age Scaling:** `Age` column scaled using `MinMaxScaler`.
- **Age Ranges:** New `age_range` column created with bins: `18-30`, `31-45`, `46-60`, `61+`.

### Analysis
- **Correlation Heatmaps:** Explored feature relationships with `treatment` (target).
- **Histograms/Bar Plots:** Visualized distributions of key features.

---

## 🧠 Model Selection Rationale
The following models were evaluated for predicting treatment needs:

**Random Forest**  -High interpretability, robust for tabular data, handles non-linear trends.
**XGBoost**- Optimized gradient boosting for high accuracy with large datasets.    
**Logistic Regression**-Baseline for binary classification, simplicity, and speed.      
 **Neural Network**   -Captures complex feature interactions (Keras implementation).      

---

## 📊 Model Evaluation
### Metrics
- **Primary Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Key Focus:** **Recall** (minimize false negatives in treatment prediction).

### Interpretability
- **SHAP Values:** Identified `family_history`, `work_interfere`, and `age_range` as top predictors.
- **LIME:** Local explanations validated model behavior on individual cases.

### Final Selection
**Neural Network** outperformed others in **Recall** (critical for reducing false negatives) while maintaining strong interpretability.
## 🚀 How to Run

1) pip install -r requirements.txt
2) put the mental_health_model.h5 and predict_mental_health.py in same directory
3) python run predict_mental_health.py   in terminal
