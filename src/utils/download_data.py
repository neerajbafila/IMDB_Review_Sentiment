import tensorflow_datasets as tfds
from src.Logger.logger import logger
import os


def download_data(download_path, dataset_name):
   
    # size = [os.path.getsize(ele) for ele in os.scandir(download_path)]
    # print(size)
    # if dataset_name in os.listdir(download_path):
    #     size = 0
    #     for files in os.scandir(download_path):
    #         size += os.stat(files).st_size
        
    #     if size > 1024:
    #         print(f"Data already present")
    #         return "Data already present"
    # else:
    print(f"downloading********************************")
    dataset, info = tfds.load(dataset_name, data_dir=download_path, with_info=True, as_supervised=True)
    return (dataset, info, f"Data downloaded at {download_path}\\{dataset_name}")

    
