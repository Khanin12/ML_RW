from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# === Load Naive Bayes Model & Scaler ===
# === Load Naive Bayes Model & Scaler ===
model_dir = os.path.join(os.path.dirname(__file__), 'naive_bayes')
with open(os.path.join(model_dir, 'naive_bayes_data_wine.pkl'), 'rb') as f:
    nb_model = pickle.load(f)

with open(os.path.join(model_dir, 'wine_quality_scaler_naive_bayes.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# === Load ID3 Model dari folder 'id3' ===
id3_dir = os.path.join(os.path.dirname(__file__), 'id3')
with open(os.path.join(id3_dir, 'id3_data_wine.pkl'), 'rb') as f:
    id3_model = pickle.load(f)

# 🔸 Ganti nilai akurasi sesuai output notebook-mu
NB_ACCURACY = 0.582   # ← ganti jika perlu
ID3_ACCURACY = 0.679  # ← ganti sesuai hasil notebook ID3

feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# === Routes ===
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/wine')
def wine_page():
    return render_template('wine.html', feature_names=feature_names, accuracy=NB_ACCURACY)

@app.route('/id3')
def id3_page():
    return render_template('id3.html', feature_names=feature_names, accuracy=ID3_ACCURACY)

# === Prediksi Naive Bayes ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for feature in feature_names:
            value = float(request.form[feature])
            input_data.append(value)
        
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        prediction = nb_model.predict(input_scaled)[0]
        probabilities = nb_model.predict_proba(input_scaled)[0]
        
        prob_dict = {}
        for cls, prob in zip(nb_model.classes_, probabilities):
            prob_dict[int(cls)] = round(prob, 4)
        
        return render_template(
            'wine.html',
            feature_names=feature_names,
            accuracy=NB_ACCURACY,
            prediction=int(prediction),
            probabilities=prob_dict
        )
    
    except Exception as e:
        return render_template(
            'wine.html',
            feature_names=feature_names,
            accuracy=NB_ACCURACY,
            error=f"Error: {str(e)}"
        )

# === Prediksi ID3 ===
@app.route('/id3/predict', methods=['POST'])
def predict_id3():
    try:
        input_data = []
        for feature in feature_names:
            value = float(request.form[feature])
            input_data.append(value)
        
        input_array = np.array(input_data).reshape(1, -1)
        # ❗ ID3 TIDAK PERLU SCALING ❗
        
        prediction = id3_model.predict(input_array)[0]
        probabilities = id3_model.predict_proba(input_array)[0]
        
        prob_dict = {}
        for cls, prob in zip(id3_model.classes_, probabilities):
            prob_dict[int(cls)] = round(prob, 4)
        
        return render_template(
            'id3.html',
            feature_names=feature_names,
            accuracy=ID3_ACCURACY,
            prediction=int(prediction),
            probabilities=prob_dict
        )
    
    except Exception as e:
        return render_template(
            'id3.html',
            feature_names=feature_names,
            accuracy=ID3_ACCURACY,
            error=f"Error: {str(e)}"
        )

if __name__ == '__main__':
    app.run(debug=True)