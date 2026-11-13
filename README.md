# Income Classification

In this project, I cleaned a real-world income dataset and used it to predict a person's income class (0 or 1) from features like age, education, occupation and hours worked per week.

## What I did

1. **Cleaned the data:** removed rows with missing values and duplicate rows (26,215 → 21,537 rows).
2. **Encoded the categories:** ordinal encoding for education, binary encoding for sex and one-hot encoding for the rest (10 → 41 columns).
3. **Scaled the features:** used MinMaxScaler, fitted on the training data only to avoid data leakage.
4. **Trained the models:** Logistic Regression and SVM, using 10-fold cross-validation and GridSearchCV for tuning.
5. **Tried clustering:** used K-Means (k = 2) to see if natural groups in the data matched the income classes.

## Results

| Model | Test accuracy |
|---|---|
| Logistic Regression (tuned) | 80.1% |
| SVM (tuned) | 80.2% |
| K-Means (k = 2) | 60.3% |

The two supervised models performed almost the same. K-Means did much worse because it never sees the income labels.

## How to run

```bash
pip install pandas numpy scikit-learn jupyter
```

Put `income.csv` in the same folder as the notebook, open `ML_ASSIGNMENT1-3.ipynb` and run all cells.

## Tools

Python, pandas, NumPy, scikit-learn, Jupyter Notebook
