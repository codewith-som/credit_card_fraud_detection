from flask import Flask,request
import pickle
import numpy as np

app = Flask(__name__)

# load  model
with open("model/model.pkl","rb") as f:
    model =pickle.load(f)
    
# load scaler
with open ("model/scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# home
@app.route('/')
def home():
    return "This is our fraud detection API, is running"

# prediction
@app.route('/predict',methods=["POST"])
def predict():
    data = request.json
    
    # convert input into array
    features = np.array(list(data.values())).reshape(1,-1)
    
    #scale
    features = scaler.transform(features)
    
    # prediction 
    predict = model.predict(features)
    return {"predicted value is":int(predict[0])}
if __name__=="__main__":
    app.run()