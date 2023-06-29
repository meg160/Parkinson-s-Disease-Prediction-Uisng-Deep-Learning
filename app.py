from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('ann.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
        int(request.form['enrlpd']),
        int(request.form['enrlprod']),
        int(request.form['enrllrrk2']),
        int(request.form['enrlgba']),
        int(request.form['enrlsnca']),
        int(request.form['enrlprkn']),
        int(request.form['enrlpink1']),
        int(request.form['conpd']),
        int(request.form['conprod']),
        int(request.form['conlrrk2']),
        int(request.form['congba']),
        int(request.form['consnca']),
        int(request.form['conprkn']),
        int(request.form['conpink1'])
    ]

    prediction = model.predict([input_data])[0][0]
    prediction = 'Admitted' if prediction >= 0.5 else 'Rejected'

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
    
    #app.run(debug=True)
