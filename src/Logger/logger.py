import os
import logging
import sys

from src.utils.common import create_directories, read_config, get_unique_name

class logger:
    def __init__(self, config_file="config/config.yaml") -> None:
        self.config_data = read_config(config_file)
        self.log_file_name = self.config_data['Logger']['log_file_name']
        self.log_location = self.config_data['Logger']['log_location']
        self.unique_log_file_name = get_unique_name()
        create_directories([self.log_location])
        self.unique_log_file_name =f"{self.log_file_name}_{self.unique_log_file_name}"
        self.log_file_path = os.path.join(self.log_location, self.unique_log_file_name)
    
    def write_log(self, msg, log_level='INFO'):
        # get logger

        self.log_level = log_level
        self.my_logger = logging.getLogger()

        # set handler
        self.file_handler = logging.FileHandler(self.log_file_path)
        # set log_level
        self.my_logger.setLevel(self.log_level.upper())

        # define formatter
        self.formats = logging.Formatter('[%(asctime)s- %(levelname)s- %(module)s- %(message)s]')
        # add format to file handler
        self.file_handler.setFormatter(self.formats)
        # add file handler to logger
        self.my_logger.addHandler(self.file_handler)

        if self.log_level.upper()=='INFO':
            self.my_logger.info(msg)
        elif self.log_level.upper()=='WARNING':
            self.my_logger.warning(msg)
        elif self.log_level.upper()=='DEBUG':
            self.my_logger.debug(msg)
        # elif self.log_level.upper()=='ERROR':
        #     self.my_logger.error(msg)
        else:
            self.my_logger.info(msg)
        
        # remove exiting handler when job finished as if you call method seconde time it will write it no of time it was previously called or duplicate msg
        self.my_logger.removeHandler(self.file_handler)
    
    def write_exception(self, e, log_level='ERROR'):
                
                self.log_level = log_level
                self.my_logger = logging.getLogger(__name__)

                # set handler
                self.file_handler = logging.FileHandler(self.log_file_path)
                # set log_level
                self.my_logger.setLevel(self.log_level.upper())

                # define formatter
                self.formats = logging.Formatter('[%(asctime)s- %(levelname)s- %(module)s- %(message)s]')
                # add format to file handler
                self.file_handler.setFormatter(self.formats)
                # add file handler to logger
                self.my_logger.addHandler(self.file_handler)

                exc_type, exc_obj, exc_tb = sys.exc_info()
                line_no = exc_tb.tb_lineno
                filename = exc_tb.tb_frame.f_code.co_filename
                msg = (f"Exception occurred {e} \ndetails are below:\nexc_type {exc_type}, exc_obj {exc_obj}, line_no {line_no}, file_name {filename}")
                self.my_logger.error(msg)
                self.my_logger.removeHandler(self.file_handler)

