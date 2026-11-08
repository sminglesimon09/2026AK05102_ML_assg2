# Steel Plate Fault Classification

Live app: <STREAMLIT_LINK>

## a. Problem statement
Classify steel plates into one of 7 fault types (Pastry, Z_Scratch, K_Scatch,
Stains, Dirtiness, Bumps, Other_Faults) from 27 geometric and luminosity
measurements, to support automated quality inspection.

## b. Dataset description  [1 mark]
- Source: UCI Machine Learning Repository, Steel Plates Faults
- Instances: 1941 | Features: 27 | Classes: 7
- Feature groups: position (X/Y min/max), area and perimeter, luminosity,
  steel type and thickness, shape indices
- Class distribution: ___ (note imbalance)
- Preprocessing: merged the 7 binary fault columns into one class label,
  StandardScaler fitted on the training split, stratified 75/25 split, seed ___

## c. GitHub repository link  [1 mark]
<GITHUB_LINK>

## d. Models used  [5 marks]

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

Precision, Recall and F1 are weighted averages. AUC is one-vs-rest, weighted.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| Overall winner for your dataset | |

## Run locally
```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
