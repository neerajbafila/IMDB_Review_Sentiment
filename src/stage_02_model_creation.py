import argparse
from src.Logger.logger import logger
from src.utils.common import read_config, create_directories
from src.stage_01_get_data import Get_data
import tensorflow as tf

STAGE = 'stage_02_model_creation'

class Create_model:

    def __init__(self, config_file= 'config/config.yaml', params_file="config/params.yaml"):
        self.config_data = read_config(config_file)
        self.param_content = read_config(params_file)
        self.logs = logger(config_file)
        self.get_data_ob = Get_data(config_file)
    
    def get_encoder(self):
        try:

            vocab_size = self.param_content['VOCAB_SIZE']
            encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
            train_ds, test_ds = self.get_data_ob.prepare_train_test_data()
            encoder.adapt(train_ds.map(lambda text, label: text))
            return encoder
        except Exception as e:
            self.logs.write_exception(e)
    
    def embedding_layer(self):
        """it handles the variable sequences lengths, makes use of <sos> - Start of Sentence, <pad> - padding <eod> - End of the Data
        """
        try:
        
            # encoder_len = len(self.get_encoder().get_vocabulary())
            # print(encoder_len)
            embedding = tf.keras.layers.Embedding(
                input_dim=1000,
                # output_dim=self.param_content['OUTPUT_DIM'],
                output_dim=64,
                mask_zero=True)
            return embedding
        
        except Exception as e:
            self.logs.write_exception(e)

    def bulid_model(self, lstm_units=128):
        try:

            encoder = self.get_encoder()
            embedding = self.embedding_layer()
            print(embedding)
            model = tf.keras.Sequential([encoder,embedding,
                                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True)),
                                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
                                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),                              
                                        tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dropout(0.7),
                                        tf.keras.layers.Dense(32, activation='relu'),     
                                        tf.keras.layers.Dense(1)])
            return model
        except Exception as e:
            self.logs.write_exception(e)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--c', default='config/config.yaml')
    parser.add_argument('--params', '--p', default='config/params.yaml')
    parsed_arg = parser.parse_args()
    logs = logger(parsed_arg.config)
    logs.write_log(f"=================={STAGE} started=======================")
    try:
        ob_model = Create_model(parsed_arg.config, parsed_arg.params)
        model = ob_model.bulid_model()
        logs.write_log(f'model created successfully')
    except Exception as e:
        logs.write_exception(e)
