from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load the model (XGBoost Booster)
model = xgb.Booster()
model.load_model("model.json")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Collect input from form
        preg = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        bp = float(request.form["bloodpressure"])
        st = float(request.form["skinthickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        dpf = float(request.form["dpf"])
        age = float(request.form["age"])

        # Prepare data in 2D array
        input_data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])

        # Create DMatrix for Booster prediction
        dmatrix = xgb.DMatrix(input_data)

        # Make prediction (you may want to threshold it)
        prediction = model.predict(dmatrix)[0]  # Output: probability or value

        # You can add a threshold (e.g., 0.5) to interpret the result
        result = "Diabetic" if prediction > 0.5 else "Non-Diabetic"

        return render_template("result.html", prediction=result)
    
if __name__=="__main__":
    app.run(debug=True)