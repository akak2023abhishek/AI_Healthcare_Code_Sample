
"""

-------------------------------------------------------------------
This script runs end-to-end: creates synthetic healthcare data, trains a Random Forest model,
and predicts kidney failure risk. It also visualizes feature importance.

Usage:
    python AI_Kidney_Failure_Executable.py
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def main():
    # Step 1: Generate synthetic healthcare dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, random_state=42)

    columns = ['age', 'blood_pressure', 'creatinine', 'glucose', 'bmi', 
               'albumin', 'sodium', 'potassium', 'hemoglobin', 'smoking']

    df = pd.DataFrame(X, columns=columns)
    df['kidney_failure_risk'] = y

    # Step 2: Train/Test Split
    X = df.drop('kidney_failure_risk', axis=1)
    y = df['kidney_failure_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 4: Prediction and Evaluation
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Step 5: Feature Importance Visualization
    importances = model.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("Kidney Failure Prediction - Feature Importance")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
