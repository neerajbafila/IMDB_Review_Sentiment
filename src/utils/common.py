import yaml
from pathlib import Path
import os
import sys
import datetime
def read_config(config_file="config/config.yaml"):
    config_file = Path(config_file)
    with open(config_file, "r") as config_file_yaml:
        content = yaml.safe_load(config_file_yaml)
    return content


def create_directories(path_to_folder:list):
    try:

        full_path = ""
        for folder in path_to_folder:
            full_path = os.path.join(full_path, folder)
        os.makedirs(full_path, exist_ok=True)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_no = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"Exception occurred \nexc_type {exc_type}, exc_obj {exc_obj}, line_no {line_no}, file_name {file_name}")
    
def get_unique_name():
    now = datetime.datetime.now()
    name = now.strftime("%y-%m-%d")
    return name
