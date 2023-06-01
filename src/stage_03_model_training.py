from src.Logger.logger import logger
from src.utils.common import read_config, create_directories, get_unique_name
from src.utils.model_utils import get_callback, get_optimizer, get_loss_function, get_plot
from src.stage_02_model_creation import Create_model
from src.stage_01_get_data import Get_data
import pandas as pd
import os
import pickle
import argparse

STAGE = "stage_03_model_training"

class Model_training:

    def __init__(self, config="config/config.yaml", params="config/params.yaml"):
        self.config = config
        self.params = params
        self.config_content = read_config(self.config)
        self.params_content = read_config(self.params)
        self.logs = logger(self.config)
        self.get_data_ob = Get_data(self.config, self.params)
        self.get_model_ob = Create_model(self.config, self.params)
    
    def model_training(self):
        try:
            self.logs.write_log(f"getting optimizer")
            optimizer = get_optimizer(self.logs, self.params)

            self.logs.write_log(f"getting loss function")
            loss = get_loss_function(self.logs, self.params)

            self.logs.write_log(f"getting callbacks")
            callbacks_lst = get_callback(self.logs, self.config)


            metrics = self.params_content['metrics']
            epochs = self.params_content['EPOCHS']
            history_path = self.config_content['History_dir']
            hist_unique_name = get_unique_name()
            history_full_path = os.path.join(history_path, hist_unique_name)
            create_directories([history_full_path])


            train_ds, test_ds = self.get_data_ob.prepare_train_test_data()
            model = self.get_model_ob.bulid_model()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            self.logs.write_log(f"============ Model training Started =================")
            history = model.fit(train_ds, epochs=epochs, validation_data=test_ds, validation_steps=30, callbacks=callbacks_lst, use_multiprocessing=True)
            self.logs.write_log(f"============ Model training Completed =================")

            model_dir = self.config_content['TRAINED_MODEL_DIR']
            base_dir = self.config_content['MODEL_BASE_LOG_DIR']
            model_dir_full_path = os.path.join(base_dir, model_dir)
            create_directories([model_dir_full_path])
            model.save(f"{model_dir_full_path}/my_model") # if you want to save entire model for exporting to other places
            self.logs.write_log(f"Model saved at {model_dir_full_path}/my_model")

            get_plot(self.logs, history, self.config, metrics)
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(f"{history_full_path}/history_df.csv")
            self.logs.write_log(f"============ saving training history =================")
            with open(f'{history_full_path}/training_history', 'wb') as pickle_file:
                training_history = pickle.dump(history.history, pickle_file) 
            self.logs.write_log(f"=======training history saved at {history_full_path}=================")
        except Exception as e:
            self.logs.write_exception(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--c', default='config/config.yaml')
    parser.add_argument('--params', '--p', default='config/params.yaml')
    parsed_arg = parser.parse_args()
    logs = logger(parsed_arg.config)
    logs.write_log(f"=================={STAGE} started=======================")
    try:
        model_training_ob = Model_training()
        model_training_ob.model_training()

    except Exception as e:
        logs.write_exception(e)

        
