# Fertilizer Recommendation System using Machine Learning

AI Lab course project — a classification model that recommends the right
fertilizer based on soil and crop conditions.

## Problem Statement
Farmers often choose fertilizer based on tradition or guesswork rather than
actual soil condition, leading to under- or over-fertilization. This project
uses supervised machine learning to recommend the correct fertilizer from
measurable inputs: temperature, humidity, moisture, soil type, crop type,
and nitrogen/phosphorus/potassium levels.

## Dataset
["Fertilizer Prediction"](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction)
dataset (Kaggle, by gdabhishek). Download the CSV separately and place it in
this folder as `Fertilizer Prediction.csv` before running (not included in
this repo).

## Approach
Five classifiers are trained and compared: Decision Tree, Random Forest, KNN,
Naive Bayes, and SVM. Each is evaluated with both a held-out test split and
5-fold cross-validation. Categorical fields are label-encoded and numeric
fields are standardized (`StandardScaler`) before training.

## Project Structure
| File | Purpose |
|---|---|
| `fertilizer_recommendation.py` | Full pipeline: load data, EDA, train/evaluate 5 models, save the best one |
| `demo.py` | Terminal-based live demo using the saved model |
| `gui.py` | Simple graphical demo (Tkinter) with dropdowns and input fields |

## How to Run
1. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn joblib`
2. Place `Fertilizer Prediction.csv` in this folder
3. Run `python fertilizer_recommendation.py` to train and evaluate all models
4. Run `python demo.py` or `python gui.py` to try live predictions

## Results
On this dataset, the Decision Tree classifier achieved the strongest
performance. See the console output and generated charts
(`confusion_matrix.png`, `feature_importance.png`, etc.) after running the
pipeline for exact numbers on your machine.

## Limitations
- Small dataset (99 records) — high accuracy reflects strong fit to this
  data, not a guarantee of real-world generalization
- No validation against real field/farm data
- Decision trees do not detect out-of-range input; the demo tools include a
  manual range-check as a safeguard
