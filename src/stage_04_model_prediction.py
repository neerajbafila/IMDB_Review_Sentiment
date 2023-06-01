from src.Logger.logger import logger
from src.utils.common import read_config
import os
from pathlib import Path
import tensorflow as tf
from src.stage_02_model_creation import Create_model
import argparse
# 👇️ disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# print("TensorFlow version:", tf.__version__)

STAGE = "stage_04_model_prediction"

class Model_prediction:
    def __init__(self, config="config/config.yaml"):
        self.config = config
        self.config_content = read_config(self.config)
        self.logs = logger(self.config)
                 
    def load_model(self):
        try:
            model_base_loc = self.config_content['MODEL_BASE_LOG_DIR']
            trained_model_dir = self.config_content['TRAINED_MODEL_DIR']
            trained_model_full_path = Path(os.path.join(model_base_loc, trained_model_dir, 'my_model'))
            # loaded_model = tf.keras.saving.load_model(trained_model_full_path, compile=True, safe_mode=True)
            self.logs.write_log(f"Loading model from {trained_model_full_path}")
            print(f"Loading model from {trained_model_full_path}.....................................")
            loaded_model = tf.keras.models.load_model(trained_model_full_path, compile=False) 
            self.logs.write_log(f"Model {trained_model_full_path} loaded successfully")
            return loaded_model
        except Exception as e:
            self.logs.write_exception(e)
    
    

    def predict_sentiment(self, text: str):
        try:
            text = text
            self.logs.write_log(f"Got Text {text} for prediction")
            print("="*30)
            print(f'Sample text for prediction \n{text}')
            loaded_model = self.load_model()
            score = loaded_model.predict([text])
            score = score[0][0]
            if score > 0:
                print("="*30)
                print(f"result: positive sentiment with score: {score}")
            else:
                print("="*30)
                print(f"result: negative sentiment with score: {score}")
            self.logs.write_log(f"Prediction completed")
        except Exception as e:
            self.logs.write_exception(e)
    
    def predict_sentiment_for_flask(self, text: str, loaded_model):
        try:
            text = text
            self.logs.write_log(f"Got Text {text} for prediction")
            score = loaded_model.predict([text])
            score = score[0][0]
            if score > 0:
                return f"result: positive sentiment with score: {score}"
            else:
                return f"result: negative sentiment with score: {score}"
            self.logs.write_log(f"Prediction completed")
        except Exception as e:
            self.logs.write_exception(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--c', default='config/config.yaml')
    parsed_arg = parser.parse_args()
    logs = logger(parsed_arg.config)
    logs.write_log(f"=================={STAGE} started=======================")
    try:
        ob_model_prediction = Model_prediction(parsed_arg.config)
        bad_sample_text = ("The movie was horrible. The animation and the graphics were out of the terrible. I would never recommend this movie.")
        txt = "The movie was really good. and the animation and acting were top notch. I would definetly recommend this movie "

        ob_model_prediction.predict_sentiment(txt)
    except Exception as e:
        logs.write_exception(e)

