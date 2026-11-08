# Steel Plate Fault Classification

## a. Problem statement
Classify steel plates into one of 7 fault types (Pastry, Z_Scratch, K_Scatch,
Stains, Dirtiness, Bumps, Other_Faults) from 27 geometric and luminosity
measurements, to support automated quality inspection.

## b. Dataset description
- Source: UCI Machine Learning Repository, Steel Plates Faults
- Instances: 1941 | Features: 27 | Classes: 7
- Feature groups: position (X/Y min/max), area and perimeter, luminosity,
  steel type and thickness, shape indices
- Class distribution: ___ (note imbalance)
- Preprocessing: merged the 7 binary fault columns into one class label,
  StandardScaler fitted on the training split, stratified 75/25 split, seed ___

## c. Github Link
- https://github.com/sminglesimon09/2026AK05102_ML_assg2/

## d. Models used  

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7202 | 0.9082 | 0.7151 | 0.7202 | 0.7152 | 0.6415 |
| Decision Tree | 0.7222 | 0.8219 | 0.7164 | 0.7222 | 0.7168 | 0.6403 |
| KNN | 0.7366 | 0.9029 | 0.7341 | 0.7366 | 0.7297 | 0.6669 |
| Naive Bayes | 0.6481 | 0.8931 | 0.7143 | 0.6481 | 0.6326 | 0.5960 |
| Support Vector Machine | 0.7613 | 0.9280 | 0.7614 | 0.7613 | 0.7527 | 0.6916 |

Precision, Recall and F1 are weighted averages. AUC is one-vs-rest, weighted.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A solid linear baseline: accuracy 0.7202 and a strong AUC of 0.9082, so it ranks the fault classes well even though its hard predictions are only moderately accurate. It is the second-best model on AUC, which suggests the scaled features carry a lot of roughly linear signal. Its MCC of 0.6415 is clearly below its accuracy because the classes are imbalanced. |
| Decision Tree | Accuracy (0.7222) is almost the same as Logistic Regression, but its AUC of 0.8219 is the lowest of all five models. A single tree outputs coarse leaf probabilities, so it ranks classes less reliably. Its MCC (0.6403) is also the second-lowest. It is easy to interpret but weaker overall. |
| KNN | Second-best model: accuracy 0.7366, AUC 0.9029 and MCC 0.6669. Because the features were standardised, distances are meaningful, so similar plates are grouped well. It stays slightly behind SVM on every metric. |
| Naive Bayes | Weakest model on accuracy (0.6481), F1 (0.6326) and MCC (0.5960). Its precision (0.7143) is much higher than its recall (0.6481), so it misses many plates. This is likely because Gaussian Naive Bayes assumes independent features, while many of the 27 features here are strongly related (area, perimeter, luminosity and the log and index versions). Its AUC (0.8931) is still reasonable. |
| Support Vector Machine | Best model on every metric: accuracy 0.7613, AUC 0.9280, precision 0.7614, recall 0.7613, F1 0.7527 and MCC 0.6916. The RBF kernel handles the non-linear boundaries between fault types well. In its confusion matrix K_Scatch is classified almost perfectly (F1 0.97), but Pastry has a low recall of 0.26 (20 of 39 plates predicted as Other_Faults), and Bumps and Other_Faults are often confused with each other. |
| Overall winner for your dataset | **Support Vector Machine.** It has the highest accuracy, AUC, precision, recall, F1 and MCC, with KNN second and Naive Bayes last. For every model MCC is lower than accuracy because the classes are imbalanced (Dirtiness and Stains have few samples, Other_Faults has many). |

## Run locally
```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
