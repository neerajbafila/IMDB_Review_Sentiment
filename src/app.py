from flask import Flask, render_template, jsonify, request
from flask_cors import cross_origin, CORS
from src.stage_04_model_prediction import Model_prediction
import argparse

# import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET'])
@cross_origin()
def homePage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        movie_review = request.form['Movie Review']
        global loaded_model
        global ob_model_prediction
        result = ob_model_prediction.predict_sentiment_for_flask(movie_review, loaded_model)
        
        return render_template('results.html', prediction='{}'.format(result))

    except Exception as e:
        print(e)
        return 'Somthing wrong'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--c', default='config/config.yaml')
    parsed_arg = parser.parse_args()
    ob_model_prediction = Model_prediction(parsed_arg.config)
    loaded_model = ob_model_prediction.load_model()
    app.run(debug=True)
