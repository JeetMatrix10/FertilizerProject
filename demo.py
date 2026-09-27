"""
Interactive demo for the Fertilizer Recommendation model.

Run this AFTER fertilizer_recommendation.py has already been run once
(so fertilizer_model.pkl and the encoder .pkl files exist in this folder).
This script loads the saved, already-trained model instead of retraining --
it's instant, and it's what you run live for your Stage 7 demonstration.
"""

import joblib
import pandas as pd

model = joblib.load("fertilizer_model.pkl")
le_soil = joblib.load("le_soil.pkl")
le_crop = joblib.load("le_crop.pkl")
le_fertilizer = joblib.load("le_fertilizer.pkl")
scaler = joblib.load("scaler.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
FEATURE_RANGES = joblib.load("feature_ranges.pkl")


def check_ranges(values_by_column):
    warnings = []
    for col, val in values_by_column.items():
        if col in FEATURE_RANGES:
            lo, hi = FEATURE_RANGES[col]
            if val < lo or val > hi:
                warnings.append(f"{col}={val} is outside the training data's "
                                 f"range ({lo}-{hi}) -- this prediction may be unreliable")
    return warnings


def match_class(value, valid_classes):
    """Find the correctly-capitalized version of a user-typed value,
    e.g. 'black' -> 'Black'. Raises ValueError if there's no match."""
    for c in valid_classes:
        if c.lower() == value.strip().lower():
            return c
    raise ValueError(f"'{value}' not in {list(valid_classes)}")


def recommend_fertilizer(temperature, humidity, moisture, soil_type,
                          crop_type, nitrogen, potassium, phosphorous):
    soil_type = match_class(soil_type, le_soil.classes_)
    crop_type = match_class(crop_type, le_crop.classes_)
    soil_encoded = le_soil.transform([soil_type])[0]
    crop_encoded = le_crop.transform([crop_type])[0]
    input_row = pd.DataFrame(
        [[temperature, humidity, moisture, soil_encoded,
          crop_encoded, nitrogen, potassium, phosphorous]],
        columns=FEATURE_COLUMNS
    )
    input_scaled = pd.DataFrame(scaler.transform(input_row), columns=FEATURE_COLUMNS)
    prediction = model.predict(input_scaled)
    return le_fertilizer.inverse_transform(prediction)[0]


def main():
    print("=== Fertilizer Recommendation Demo ===\n")
    print("Valid soil types:", list(le_soil.classes_))
    print("Valid crop types:", list(le_crop.classes_))
    print("\nType 'quit' at any prompt to exit.\n")

    while True:
        soil = input("Soil Type: ").strip()
        if soil.lower() == "quit":
            break
        crop = input("Crop Type: ").strip()
        try:
            temperature = float(input("Temperature: "))
            humidity = float(input("Humidity: "))
            moisture = float(input("Moisture: "))
            nitrogen = float(input("Nitrogen: "))
            potassium = float(input("Potassium: "))
            phosphorous = float(input("Phosphorous: "))

            values_by_column = dict(zip(
                [c for c in FEATURE_COLUMNS if c not in ("Soil Type", "Crop Type")],
                [temperature, humidity, moisture, nitrogen, potassium, phosphorous]
            ))
            for w in check_ranges(values_by_column):
                print(f"Warning: {w}")

            result = recommend_fertilizer(
                temperature, humidity, moisture, soil, crop,
                nitrogen, potassium, phosphorous
            )
            print(f"\n>>> Recommended fertilizer: {result}\n")

        except ValueError as e:
            print(f"\nThat soil/crop type or number wasn't recognized ({e}). "
                  f"Check the valid lists above and try again.\n")


if __name__ == "__main__":
    main()
