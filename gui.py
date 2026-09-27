"""
Simple graphical interface for the Fertilizer Recommendation model.

Run this AFTER fertilizer_recommendation.py has been run at least once
(so fertilizer_model.pkl, the encoders, and scaler.pkl already exist in
this folder).

Requires: tkinter -- this comes bundled with the standard Python installer
on Windows, so no extra "pip install" is needed.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

model = joblib.load("fertilizer_model.pkl")
le_soil = joblib.load("le_soil.pkl")
le_crop = joblib.load("le_crop.pkl")
le_fertilizer = joblib.load("le_fertilizer.pkl")
scaler = joblib.load("scaler.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
FEATURE_RANGES = joblib.load("feature_ranges.pkl")

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("Soil Type", "Crop Type")]


def predict(values):
    """values: dict mapping each column name to its raw (un-encoded,
    un-scaled) value. Returns (prediction_text, list_of_warning_strings)."""
    soil_encoded = le_soil.transform([values["Soil Type"]])[0]
    crop_encoded = le_crop.transform([values["Crop Type"]])[0]

    row = {}
    for col in FEATURE_COLUMNS:
        if col == "Soil Type":
            row[col] = soil_encoded
        elif col == "Crop Type":
            row[col] = crop_encoded
        else:
            row[col] = values[col]

    input_row = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    input_scaled = pd.DataFrame(scaler.transform(input_row), columns=FEATURE_COLUMNS)
    prediction = model.predict(input_scaled)
    result = le_fertilizer.inverse_transform(prediction)[0]

    warnings = []
    for col in NUMERIC_COLUMNS:
        lo, hi = FEATURE_RANGES[col]
        if values[col] < lo or values[col] > hi:
            warnings.append(f"{col}={values[col]} is outside the training "
                             f"range ({lo}-{hi})")
    return result, warnings


def on_submit():
    try:
        values = {
            "Soil Type": soil_var.get(),
            "Crop Type": crop_var.get(),
        }
        for col, entry in numeric_entries.items():
            values[col] = float(entry.get())

        result, warnings = predict(values)
        result_label.config(text=f"Recommended fertilizer: {result}")
        warning_label.config(text=("Note: " + "; ".join(warnings)) if warnings else "")

    except ValueError:
        messagebox.showerror(
            "Invalid input", "Please enter valid numbers for all numeric fields."
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


def on_clear():
    soil_var.set(le_soil.classes_[0])
    crop_var.set(le_crop.classes_[0])
    for entry in numeric_entries.values():
        entry.delete(0, tk.END)
    result_label.config(text="")
    warning_label.config(text="")


root = tk.Tk()
root.title("Fertilizer Recommendation")
root.geometry("380x430")

frame = ttk.Frame(root, padding=16)
frame.pack(fill="both", expand=True)

ttk.Label(frame, text="Soil Type").grid(row=0, column=0, sticky="w", pady=4)
soil_var = tk.StringVar(value=le_soil.classes_[0])
ttk.Combobox(frame, textvariable=soil_var, values=list(le_soil.classes_),
             state="readonly").grid(row=0, column=1, pady=4)

ttk.Label(frame, text="Crop Type").grid(row=1, column=0, sticky="w", pady=4)
crop_var = tk.StringVar(value=le_crop.classes_[0])
ttk.Combobox(frame, textvariable=crop_var, values=list(le_crop.classes_),
             state="readonly").grid(row=1, column=1, pady=4)

numeric_entries = {}
for i, col in enumerate(NUMERIC_COLUMNS, start=2):
    lo, hi = FEATURE_RANGES[col]
    ttk.Label(frame, text=f"{col} ({lo}-{hi})").grid(row=i, column=0, sticky="w", pady=4)
    entry = ttk.Entry(frame)
    entry.grid(row=i, column=1, pady=4)
    numeric_entries[col] = entry

submit_row = len(NUMERIC_COLUMNS) + 2
ttk.Button(frame, text="Recommend", command=on_submit).grid(
    row=submit_row, column=0, pady=12
)
ttk.Button(frame, text="Clear", command=on_clear).grid(
    row=submit_row, column=1, pady=12
)

result_label = ttk.Label(frame, text="", font=("Segoe UI", 11, "bold"), wraplength=320)
result_label.grid(row=submit_row + 1, column=0, columnspan=2, pady=4)

warning_label = ttk.Label(frame, text="", foreground="#a32d2d", wraplength=320)
warning_label.grid(row=submit_row + 2, column=0, columnspan=2, pady=4)

root.mainloop()
