# Importer les packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib as jb
import gradio as gr



# Fonction de prédiction
# importer les encodeurs
encoders = jb.load('encoders.joblib')
# importer les valeurs uniques
uniques = jb.load('uniques.joblib')
# importer le normaliseur
scaler = jb.load('scaler.joblib')
# importer le modèle
xgb = jb.load('xgb.joblib')

# noms des classes
clasnames = uniques[2]
# fonction de prédiction simple
def Pred_func(prix, adresse, marque,dim_ecr, ram, stockage):

  # Encoder les valeurs des Fuel_Type, Seller_Type et Transmission
  adresse = encoders[0].transform([adresse])[0]
  marque = encoders[1].transform([marque])[0]
  # vecteur des valeurs numériques
  x_new = np.array([prix, adresse, marque,dim_ecr, ram, stockage])
  x_new = x_new.reshape(1,-1) # convert en un 2D array
  # Normaliser les données
  x_new = scaler.transform(x_new)
  # Prédire
  y_pred = xgb.predict(x_new)
  return clasnames[y_pred[0]]



# Fonction de prédiction multiple
def Pred_func_csv(file):
  # Lire le fichier csv
  df = pd.read_csv(file)
  predictions = []
  # Boucle sur les lignes du dataframe
  for row in df.iloc[:, :].values:
    # prédiction simple
    y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
    predictions.append(y_pred)

  df['etat'] = predictions
  df.to_csv('predictions.csv', index = False)
  return 'predictions.csv'

# définir les blocks
demo = gr.Blocks(theme = gr.themes.Monochrome())

# Créer les inputs
inputs = [gr.Number(label='Prix'),
          gr.Dropdown(choices=uniques[0], label='Adresse'),
          gr.Dropdown(choices=uniques[1], label='Marque'),
          gr.Number(label='Dimensin écran'),
          gr.Number(label='Nombre de ram'),
          gr.Number(label = 'Stockage')]
# Créer les outputs
outputs = gr.Textbox(label='Etat du portable')
# Créer l'interface 1
interface1 = gr.Interface(fn = Pred_func,
                         inputs = inputs,
                         outputs = outputs,
                         title="Prédire l'état d'un portable avec une entrée",
                         description = """Ce modèle de machine permet de prédire l'état d'un portable en partant
                        du prix, l'adresse, la marque, la dimension de l'écran, le nombre de ram et le stockage.
                         """)
# Créer l'interface 2
interface2 = gr.Interface(fn = Pred_func_csv,
                         inputs = gr.File(label='Importer un fichier csv'),
                          outputs = gr.File(label='Télécharger un fichier csv'),
                         title="Prédire l'état d'un portable avec plusieurs entrées",
                         description = """Ce modèle de machine permet de prédire l'état d'un portable en partant
                        du prix, l'adresse, la marque, la dimension de l'écran, le nombre de ram et le stockage.
                         """)

# faire un tabbing des interfaces
with demo:
  gr.TabbedInterface([interface1, interface2], ['Simple Prediction', 'Prédiction multiple'])

# lancer l'interface
demo.launch(server_name="0.0.0.0", server_port=7860)

