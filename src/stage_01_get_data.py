import os
import argparse
import tensorflow as tf
from src.Logger.logger import logger
from src.utils.common import read_config, create_directories
from src.utils.download_data import download_data

STAGE = "stage_01_get_data"
class Get_data:

    def __init__(self, config="config/config.yaml", params="config/params.yaml"):
            
            self.logs = logger(config)
            try:
                self.content = read_config(config)
                self.params = read_config(params)
            except Exception as e:
                 self.logs.write_exception(e)
    def download_dataset(self):
         try:
              download_path = os.path.join(self.content['Data']['root_data_folder'], self.content['Data']['imdb_data_folder'])
              create_directories([download_path])
              dataset_name = self.content['Data']['dataset_name']
              dataset, info, result = download_data(download_path, dataset_name)
              self.logs.write_log(result)
              return dataset
         except Exception as e:
              self.logs.write_exception(e)
     
    def prepare_train_test_data(self):
         dataset = self.download_dataset()
         train_ds, test_ds = dataset["train"], dataset["test"]
         train_ds = train_ds.shuffle(self.params['BUFFER_SIZE']).batch(self.params['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE)
         test_ds = test_ds.batch(self.params['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE)
         return train_ds, test_ds


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('--config', '--c', default='config/config.yaml')
     parser.add_argument('--params', '--p', default='config/params.yaml')
     parsed_args =parser.parse_args()
     logs = logger(parsed_args.config)
     logs.write_log(f"=================={STAGE} started=======================")
     try:
          # print("Here is the")
          get_data_ob = Get_data(parsed_args.config, parsed_args.params)
          get_data_ob.prepare_train_test_data()

     except Exception as e:
          logs.write_exception(e) 

        
