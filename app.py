from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import gradio as gr

# ==========================
# Chargement des fichiers
# ==========================

BASE_DIR = Path(__file__).resolve().parent

encoders = joblib.load(BASE_DIR / "encoders.joblib")
uniques = joblib.load(BASE_DIR / "uniques.joblib")
scaler = joblib.load(BASE_DIR / "scaler.joblib")
model = joblib.load(BASE_DIR / "xgb.joblib")

# Classes
class_names = uniques[2]


# ==========================
# Prédiction simple
# ==========================

def predict_phone(price, address, brand, screen_size, ram, storage):

    address = encoders[0].transform([address])[0]
    brand = encoders[1].transform([brand])[0]

    x = np.array(
        [[price,
          address,
          brand,
          screen_size,
          ram,
          storage]]
    )

    x = scaler.transform(x)

    prediction = model.predict(x)[0]

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(x).max() * 100
    else:
        confidence = None

    if confidence is not None:
        return f"{class_names[prediction]} (Confidence : {confidence:.2f}%)"

    return class_names[prediction]


# ==========================
# Prédiction CSV
# ==========================

def predict_csv(file):

    df = pd.read_csv(file.name)

    expected_columns = [
        "Prix",
        "Adresse",
        "Marque",
        "Dimension écran",
        "Ram",
        "Stockage"
    ]

    if len(df.columns) != 6:
        return "Erreur : le fichier CSV doit contenir exactement 6 colonnes."

    predictions = []

    for row in df.values:

        pred = predict_phone(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        )

        predictions.append(pred)

    df["Etat prédit"] = predictions

    output_path = BASE_DIR / "predictions.csv"

    df.to_csv(output_path, index=False)

    return str(output_path)


# ==========================
# Interface 1
# ==========================

single_interface = gr.Interface(
    fn=predict_phone,

    inputs=[
        gr.Number(label="Prix"),

        gr.Dropdown(
            choices=list(uniques[0]),
            label="Adresse"
        ),

        gr.Dropdown(
            choices=list(uniques[1]),
            label="Marque"
        ),

        gr.Number(label="Dimension écran"),

        gr.Number(label="RAM"),

        gr.Number(label="Stockage")
    ],

    outputs=gr.Textbox(label="Etat du portable"),

    title="Phone Condition Classification",

    description="""
Predict the condition of a mobile phone using
Price,
Location,
Brand,
Screen Size,
RAM,
and Storage.
"""
)


# ==========================
# Interface 2
# ==========================

csv_interface = gr.Interface(

    fn=predict_csv,

    inputs=gr.File(
        label="Upload CSV"
    ),

    outputs=gr.File(
        label="Download Predictions"
    ),

    title="Batch Prediction",

    description="""
Upload a CSV file containing multiple phones
and receive predictions automatically.
"""
)


# ==========================
# Application
# ==========================

demo = gr.TabbedInterface(

    [single_interface, csv_interface],

    ["Single Prediction", "Batch Prediction"]

)

# ==========================
# Lancement
# ==========================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)