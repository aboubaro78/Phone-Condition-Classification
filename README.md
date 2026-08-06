# 📱 Phone Condition Classification using XGBoost

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![Gradio](https://img.shields.io/badge/Gradio-Web%20App-red)

</div>

---

## 🚀 Live Demo

👉 https://phone-condition-classification-1.onrender.com/

---

## 📖 Overview

This project presents an end-to-end Machine Learning pipeline for predicting the condition of a mobile phone listed on online marketplaces.

The application classifies a phone into one of several conditions using its technical characteristics and marketplace information.

The final model was selected after comparing several ensemble learning algorithms, with **XGBoost** achieving the best predictive performance.

---

## 🎯 Problem Statement

Buying or selling second-hand smartphones often involves uncertainty regarding the actual condition of the device.

This project aims to automate the classification process using Machine Learning.

The prediction relies on:

- 💰 Price
- 📍 Selling Location
- 📱 Brand
- 📺 Screen Size
- 💾 RAM
- 💽 Storage Capacity

---

## 🧠 Machine Learning Pipeline

The workflow consists of:

1. Data preprocessing
2. Feature encoding
3. Feature scaling
4. Model training
5. Model comparison
6. Best model selection
7. Model deployment using Gradio

---

## 🤖 Models Compared

The following ensemble models were evaluated:

- Random Forest
- Gradient Boosting
- XGBoost

🏆 **Selected Model:** XGBoost Classifier

---

## 💻 Web Application

The Gradio application provides two prediction modes.

### 📱 Single Prediction

Predict the condition of one smartphone using manual inputs.

### 📂 Batch Prediction

Upload a CSV file containing multiple phones and automatically download predictions.

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Joblib
- Gradio

---

## 📁 Repository Structure

```text
Phone-Condition-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── encoders.joblib
├── scaler.joblib
├── uniques.joblib
├── xgb.joblib
│
├── notebook/
│   └── Clas_Random_Forest_GBoost_XGboost.ipynb
│
├── report/
│   └── Report.pdf
│
└── sample.csv
```

---

## 📊 Dataset

Source:

**CoinAfrique Senegal**

The dataset contains smartphone advertisements described by:

- Price
- Brand
- Selling location
- Screen size
- RAM
- Storage

---

## 🌐 Deployment

The application is deployed on Render.

🔗 https://phone-condition-classification-1.onrender.com/

---

## 👨‍💻 Author

**Abou Birane BARO**

Data Scientist | Machine Learning Engineer

- GitHub: https://github.com/aboubaro78
- Portfolio: https://abou-baro-portfolio.vercel.app/

---

## ⭐ Support

If you found this project useful,

⭐ Star this repository.
