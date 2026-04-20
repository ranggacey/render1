from flask import Flask, request, render_template
import pickle
import pandas as pd
import os
from pathlib import Path

# Create flask app
flask_app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_regression_model.pkl"
TRANSFORMER_PATH = BASE_DIR / "transformer.pkl"
DATA_PATH = BASE_DIR / "healthcare_dataset.csv"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(TRANSFORMER_PATH, "rb") as transformer_file:
    transformer = pickle.load(transformer_file)

dataset_warning = None
try:
    df = pd.read_csv(DATA_PATH)
except Exception as exc:
    dataset_warning = f"Gagal membaca dataset: {exc}"
    df = pd.DataFrame(columns=DISPLAY_COLUMNS if "DISPLAY_COLUMNS" in locals() else [])

FEATURE_COLUMNS = ["Age", "Gender", "Blood Type", "Medical Condition"]
DISPLAY_COLUMNS = [
    "Name",
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Doctor",
    "Hospital",
    "Insurance Provider",
    "Billing Amount",
    "Admission Type",
    "Medication",
    "Test Results",
]


def get_template_context(prediction_text):
    def safe_unique(column_name):
        if column_name in df.columns:
            return sorted(df[column_name].dropna().unique())
        return []

    sample_data = []
    if not df.empty and all(col in df.columns for col in DISPLAY_COLUMNS):
        sample_data = df[DISPLAY_COLUMNS].head(12).to_dict(orient="records")

    effective_warning = dataset_warning
    if not effective_warning and df.empty:
        effective_warning = (
            "Dataset kosong atau tidak tersedia di server. Pastikan file "
            "`healthcare_dataset.csv` ikut ter-deploy."
        )
    return {
        "prediction_text": prediction_text,
        "sample_data": sample_data,
        "genders": safe_unique("Gender"),
        "blood_types": safe_unique("Blood Type"),
        "medical_conditions": safe_unique("Medical Condition"),
        "column_names": df.columns.tolist(),
        "dataset_warning": effective_warning,
    }


@flask_app.route("/")
def Home():
    return render_template("index.html", **get_template_context(prediction_text=None))

@flask_app.route("/predict", methods = ["POST"])
def predict():
    try:
        age_raw = request.form.get("Age", "").strip()
        gender = request.form.get("Gender", "").strip()
        blood_type = request.form.get("Blood Type", "").strip()
        medical_condition = request.form.get("Medical Condition", "").strip()

        if not age_raw or not gender or not blood_type or not medical_condition:
            raise ValueError("Semua field input wajib diisi.")

        age = float(age_raw)
        features_df = pd.DataFrame(
            [[age, gender, blood_type, medical_condition]],
            columns=FEATURE_COLUMNS,
        )
        transformed_features = transformer.transform(features_df)
        prediction = model.predict(transformed_features)[0]
        prediction_text = f"Prediksi Billing Amount: ${prediction:,.2f}"
    except ValueError as exc:
        prediction_text = f"Input tidak valid: {exc}"
    except Exception:
        prediction_text = (
            "Prediksi gagal diproses. Coba jalankan ulang training model "
            "(`python regression.py`) lalu restart aplikasi."
        )

    return render_template("index.html", **get_template_context(prediction_text=prediction_text))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    flask_app.run(host="0.0.0.0", port=port, debug=False)
