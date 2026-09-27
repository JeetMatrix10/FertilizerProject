"""
Fertilizer Recommendation System using Machine Learning
AI Lab Course Project - Extended Version

Adds on top of the basic pipeline:
- Exploratory Data Analysis (EDA) charts
- Comparison across 5 classifiers (Decision Tree, Random Forest, KNN,
  Naive Bayes, SVM)
- 5-fold cross-validation for a more reliable accuracy estimate than a
  single train/test split can give on a small dataset
- Feature importance chart (which inputs matter most)

HOW TO USE:
- Place the dataset CSV in the same folder as this script.
- Run: python fertilizer_recommendation.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DATA_FILE = "Fertilizer Prediction.csv"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset shape (rows, columns):", df.shape)

# ---------------------------------------------------------------------------
# 1b. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
df["Fertilizer Name"].value_counts().plot(kind="bar", color="#1D9E75")
plt.title("Fertilizer distribution in the dataset")
plt.xlabel("Fertilizer")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fertilizer_distribution.png")
print("Saved fertilizer_distribution.png")

numeric_cols_for_eda = [c for c in df.columns
                         if c not in ("Soil Type", "Crop Type", "Fertilizer Name")]
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols_for_eda].corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation between numeric features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("Saved correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 2. Preprocessing: encode text columns into numbers
# ---------------------------------------------------------------------------
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

df["Soil Type"] = le_soil.fit_transform(df["Soil Type"])
df["Crop Type"] = le_crop.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = le_fertilizer.fit_transform(df["Fertilizer Name"])

X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Save the raw (pre-scaling) numeric ranges -- this is what new demo/GUI
# input gets checked against to warn on unrealistic values
numeric_cols = [c for c in X.columns if c not in ("Soil Type", "Crop Type")]
feature_ranges = {c: (X[c].min(), X[c].max()) for c in numeric_cols}

# Feature scaling: KNN and SVM are distance-based, so a feature like
# Moisture (0-70) would dominate a feature like Nitrogen (0-40) unless all
# features are put on the same scale. Decision Tree / Random Forest are
# threshold-based and unaffected by scaling either way.
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining rows: {len(X_train)}  |  Test rows: {len(X_test)}")

# ---------------------------------------------------------------------------
# 4. Train and compare 5 classifiers
# ---------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", random_state=42),
}

results = {}
cv_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    # 5-fold cross-validation across the whole dataset -- more reliable
    # than a single 20-row test split on a small dataset like this one
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    cv_results[name] = cv_scores.mean()

    print(f"\n--- {name} ---")
    print(f"Test-split accuracy: {acc:.2%}")
    print(f"5-fold cross-validation accuracy: {cv_scores.mean():.2%} "
          f"(+/- {cv_scores.std():.2%})")

print("\n=== Summary ===")
print(f"{'Model':15s} {'Test acc':>10s} {'CV acc':>10s}")
for name in models:
    print(f"{name:15s} {results[name]:>9.2%} {cv_results[name]:>9.2%}")

best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest performing model (by test-split accuracy): "
      f"{best_name} ({results[best_name]:.2%})")

preds = best_model.predict(X_test)
print(f"\nDetailed report for {best_name}:")
print(classification_report(
    y_test, preds,
    labels=range(len(le_fertilizer.classes_)),
    target_names=le_fertilizer.classes_,
    zero_division=0
))

# ---------------------------------------------------------------------------
# 5. Confusion matrix for the best model
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, preds, labels=range(len(le_fertilizer.classes_)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_fertilizer.classes_, yticklabels=le_fertilizer.classes_)
plt.xlabel("Predicted fertilizer")
plt.ylabel("Actual fertilizer")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nSaved confusion_matrix.png")

# ---------------------------------------------------------------------------
# 6. Feature importance (from Random Forest -- the standard way to explain
#    which inputs drove the model's decisions, regardless of which model
#    ultimately won on accuracy)
# ---------------------------------------------------------------------------
rf_for_importance = models["Random Forest"]
importances = pd.Series(rf_for_importance.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#7F77DD")
plt.title("Feature importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved feature_importance.png")

# ---------------------------------------------------------------------------
# 7. Save everything needed for the demo / GUI
# ---------------------------------------------------------------------------
joblib.dump(best_model, "fertilizer_model.pkl")
joblib.dump(le_soil, "le_soil.pkl")
joblib.dump(le_crop, "le_crop.pkl")
joblib.dump(le_fertilizer, "le_fertilizer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
joblib.dump(feature_ranges, "feature_ranges.pkl")
print("\nSaved model, encoders, scaler, feature_columns.pkl, feature_ranges.pkl")


# ---------------------------------------------------------------------------
# 8. Recommendation function
# ---------------------------------------------------------------------------
def recommend_fertilizer(temperature, humidity, moisture, soil_type,
                          crop_type, nitrogen, potassium, phosphorous):
    soil_encoded = le_soil.transform([soil_type])[0]
    crop_encoded = le_crop.transform([crop_type])[0]
    input_row = pd.DataFrame(
        [[temperature, humidity, moisture, soil_encoded,
          crop_encoded, nitrogen, potassium, phosphorous]],
        columns=X.columns
    )
    input_scaled = pd.DataFrame(scaler.transform(input_row), columns=X.columns)
    prediction = best_model.predict(input_scaled)
    return le_fertilizer.inverse_transform(prediction)[0]


if __name__ == "__main__":
    example = recommend_fertilizer(
        temperature=26, humidity=52, moisture=38,
        soil_type=le_soil.classes_[0], crop_type=le_crop.classes_[0],
        nitrogen=37, potassium=0, phosphorous=0
    )
    print(f"\nExample recommendation: {example}")
