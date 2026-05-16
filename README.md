# Titanic Survival Prediction 🚢

Predicting survival of Titanic passengers using Machine Learning.
Compared 4 algorithms, applied feature scaling, and analysed model internals.

## Results

| Model | Before Scaling | After Scaling |
|-------|---------------|---------------|
| Random Forest | 82.00% | 82.00% (not affected) |
| Logistic Regression | 79.89% | 79.89% (minor effect) |
| KNN | 70.39% | 80.45% (+10%) |
| SVM | 65.36% | 82.12% (+17%) |

## What I Did

- Loaded real Titanic dataset (891 passengers, 12 columns)
- Cleaned missing data (Age filled with mean, Cabin dropped, Embarked filled with mode)
- Encoded categorical variables (Sex, Embarked → numbers)
- Applied 80/20 train-test split
- Trained and compared 4 ML algorithms
- Identified why SVM and KNN underperformed (feature scaling issue)
- Applied StandardScaler and improved SVM by 17%, KNN by 10%
- Verified Random Forest internals — inspected individual tree accuracies and question patterns
- Analysed feature importance across all 100 trees

## Key Insight

SVM and KNN both use distance calculations between passengers.
Fare ranges from 7 to 512. Sex ranges from 0 to 1.
Without scaling, SVM ignored Sex almost completely and based decisions on Fare alone.
StandardScaler fixed this — bringing all features to equal scale.
Result: SVM jumped from 65% to 82%.

## Visualisations

- Survival rate by gender and class
- Feature correlation heatmap
- Feature importance chart
- Model comparison before and after scaling

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Concepts Demonstrated

- Data cleaning and preprocessing
- Feature encoding
- Train-test split
- Model selection and comparison
- Feature scaling (StandardScaler)
- Model interpretability (feature importance, tree inspection)
- Data leakage prevention
