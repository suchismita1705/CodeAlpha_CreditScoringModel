# End-to-End Credit Scoring System

An industry-grade Machine Learning pipeline built to predict consumer creditworthiness and automate loan approval risk assessments using historical financial tracking data.

## 📌 Project Architecture
The project is built using a modular framework separating data states, exploratory workflows, and production inference execution layouts:

- `data/raw/`: Original unedited immutable financial log records.
- `data/processed/`: Outlier-cleansed, missing-value-imputed, and hot-encoded feature matrices.
- `notebooks/`: Iterative development workspaces for visual data exploration and modeling.
- `saved_models/`: Serialized binary files for the finalized scaler parameters and optimal model weights.
- `src/predict.py`: Standalone real-time risk decision execution script for deployment tracking.

## 📊 Core Data Engineering & Cleaning Pipelines
- **Outlier Mitigation:** Identified and removed physical anomalies within the historical data ledger (e.g., records indicating an applicant age of 144 or employment duration of 123 years).
- **Imputation Framework:** Applied median imputation strategy to columns (`person_emp_length`, `loan_int_rate`) to safely eliminate blank observations without shifting core population shapes.
- **Categorical Transformation:** Transformed textual parameters (Home Ownership, Loan Intent, Loan Grade) into distinct mathematical indicators using One-Hot Encoding with first-category removal to bypass multicollinearity traps.
- **Feature Engineering:** Formulated a custom transactional attribute, `loan_to_income_ratio`, mapping out direct consumer debt burden profiles.

## 📈 Model Performance Matrix
We evaluated a baseline linear classifier against an optimized tree ensemble structure to handle high-dimensional non-linear signals.

| Evaluation Metric | Baseline Logistic Regression | Tuned Random Forest Classifier |
| :--- | :--- | :--- |
| **Overall Model Accuracy** | 86.00% | 92.00% |
| **Defaulter Precision (Class 1)** | 76.00% | 85.00% |
| **Defaulter Recall (Class 1)** | 54.00% | 74.00% |
| **Area Under ROC Curve (ROC-AUC)** | 0.8642 | 0.9298 |

*Key Takeaway:* Upgrading to a Tuned Random Forest allowed the predictive framework to locate complex risk intersections, driving consumer default tracking capability (**Recall**) up by a massive margin of 20.00%.

## 🛠️ Installation & Reproduction Setup
Ensure you have Python installed locally, then execute the following environment configuration sequence:

```bash
# 1. Clone or access the project folder directory root
cd credit-scoring-model

# 2. Construct and launch an isolated python virtual sandbox
python -m venv env
env\Scripts\activate

# 3. Synchronize project package dependencies
pip install -r requirements.txt