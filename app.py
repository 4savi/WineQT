from flask import Flask, render_template,request, jsonify
import numpy as np
import pickle


app = Flask(__name__)
@app.route("/")

def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = request.form
    fixed_acidity = data["fixed_acidity"]
    volatile_acidity =float(data["volatile_acidity"])
    citric_acid = float(data["citric_acid"])
    residual_sugar = float(data["residual_sugar"])
    chlorides = float(data["chlorides"])
    free_sulfur_dioxide = float(data["free_sulfur_dioxide"])
    total_sulfur_dioxide =float(data["total_sulfur_dioxide"])
    density = float(data["density"])
    pH = float(data["pH"])
    sulphates = float(data["sulphates"])

    data1 = np.array([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide, total_sulfur_dioxide, density,pH,sulphates],ndmin=2)
    with open("model_pkl.pkl", "rb") as file:
        model = pickle.load(file)
    result = model.predict(data1)
    return render_template("index.html",res = result[0])
if __name__ == "__main__":
    app.run(host = "localhost",port = 8080, debug = False)