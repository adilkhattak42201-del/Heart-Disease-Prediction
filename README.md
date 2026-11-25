<h1 align="center">❤️ 10-Year Coronary Heart Disease Prediction (Framingham Dataset)</h1>
<p align="center"><b>Supervised Machine Learning classification pipeline predicting 10-year CHD risk (<code>TenYearCHD</code>) using Python, Scikit-Learn, Support Vector Classifier (SVC), and Decision Tree Classifier.</b></p>

## 📌 Project Overview
This project builds a supervised Machine Learning classification workflow on the **Framingham Heart Study** dataset (`framingham.csv` — **4,238 rows**, **15 features**, target **`TenYearCHD`**) to predict 10-year coronary heart disease risk[cite: 12]. It covers missing-value inspection and median imputation, duplicate verification, train/test splitting, feature scaling, and model performance comparison using classification reports[cite: 12].

## 📊 Dataset & Features
* **Total Records:** `4,238` rows × `16` columns (`15` features + target `TenYearCHD`)[cite: 12]
* **Target Distribution (`TenYearCHD`):** `0` (No CHD): `3,594` · `1` (10-Year CHD): `644` (class imbalance present)[cite: 12]
* **Demographic & Lifestyle Features:** `male`, `age`, `education`, `currentSmoker`, `cigsPerDay`[cite: 12]
* **Clinical & Health Features:** `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`[cite: 12]
* **Missing Values Handled:** Imputed using `SimpleImputer(strategy='median')` (notably `glucose`: 388, `education`: 105, `BPMeds`: 53, `totChol`: 50, `cigsPerDay`: 29, `BMI`: 19, `heartRate`: 1)[cite: 12]

## 🔄 Project Workflow
1. Load `framingham.csv` using `pandas` and inspect top rows / dataset structure[cite: 12]
2. Check missing values per column and verify target class counts (`TenYearCHD`)[cite: 12]
3. Check and clean duplicate rows (`0` duplicates found across `4,238` rows)[cite: 12]
4. Split features (`X`) and target (`y = TenYearCHD`), then perform an **85% / 15% train-test split** (`3,602` train samples, `636` test samples, `random_state=42`)[cite: 12]
5. Handle missing values with median imputation (`SimpleImputer`) and scale features with `MinMaxScaler`[cite: 12]
6. Train and evaluate **Linear SVC** (`SVC(kernel='linear')`) and **Decision Tree Classifier** (`DecisionTreeClassifier(criterion='entropy')`) using `classification_report`[cite: 12]

## 📈 Model Results Summary (`636` Test Samples)
| Model | Configuration | Test Accuracy | Notes |
|---|---|---|---|
| **Support Vector Classifier (SVC)** | `kernel="linear"` | **0.85** (`85%`) | Predicts majority class `0` heavily (`0` precision: `0.85`, recall: `1.00`; class `1` precision/recall: `0.00`)[cite: 12] |
| **Decision Tree Classifier** | `criterion="entropy"` | **0.75** (`75%`) | `1.00` train accuracy vs `0.75` test accuracy (`0` precision: `0.87`, recall: `0.84`; class `1` precision: `0.23`, recall: `0.28`) — shows overfitting on minority class `1`[cite: 12] |

## 🛠️ Tech Stack
`Python` · `Pandas` · `Scikit-learn` (`SimpleImputer`, `MinMaxScaler`, `train_test_split`, `SVC`, `DecisionTreeClassifier`, `classification_report`) · `Jupyter Notebook`[cite: 12]

## 🚀 How to Run
```bash
pip install pandas scikit-learn jupyter
jupyter notebook
