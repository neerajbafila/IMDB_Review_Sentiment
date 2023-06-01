from src.utils.common import read_config, create_directories, get_unique_name
import tensorflow as tf
from tensorflow.keras import optimizers
import os
import logging
import matplotlib.pyplot as plt



def get_callback(logs: logging.Logger, config='config/config.yaml'):
    try:

        config_content = read_config(config)
        print(config_content)
        base_dir = config_content['MODEL_BASE_LOG_DIR'] #MODEL_BASE_LOG_DIR
        tb_log_dir = config_content['TB_ROOT_LOG_DIR']
        tb_unique_name = get_unique_name()
        tb_log_dir_full_path = os.path.join(base_dir, tb_log_dir, tb_unique_name)
        create_directories([tb_log_dir_full_path])
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir_full_path)

        ckpt_path = config_content['CHECKPOINT_DIR']
        ckpt_full_path = os.path.join(base_dir, ckpt_path, tb_unique_name)
        create_directories([ckpt_full_path])
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_full_path, save_best_only=True)

        callbacks_lst = [tb_cb, ckpt_cb]

        return callbacks_lst
    except Exception as e:
        print(e)
        logs.write_exception(e)

def get_optimizer(logs, params="config/params.yaml"):
    try:
        params_content = read_config(params) 
    # optimizer = optimizers.deserialize(params_content['optimizer'])
        optimizer = tf.keras.optimizers.deserialize(params_content['optimizer'])
    # print(f"{optimizer.learning_rate}")
    # print(optimizer.__getattribute__('learning_rate'))
        return optimizer
    except Exception as e:
        logs.write_exception(e)


def get_loss_function(logs, params="config/params.yaml"):
    try:
        params_content = read_config(params)
        loss_name = params_content['loss']
        loss = tf.keras.losses.deserialize(params_content['loss'])
        return loss
    except Exception as e:
        logs.write_exception(e)

def get_plot(logs, history, config='config/config.yaml', metric=['accuracy']):
    try:
        metric = metric[0]
        config_content = read_config(config)
        history_dir = config_content['History_dir']
        plot_dir = config_content['Training_data_plot']
        plot_dir_full_path = os.path.join(history_dir, plot_dir)
        create_directories([plot_dir_full_path])
        plt.style.use("fivethirtyeight")
        history_obj = history.history
        plt.plot(history_obj[metric])
        plt.plot(history_obj[f'val_{metric}'])
        plt.xlabel("Epochs -->")
        plt.ylabel(f"{metric} -->")
        plt.legend([metric, f'val_{metric}'])
        plt.savefig(f"{plot_dir_full_path}/train_history.jpg")
        logs.write_log(f"plot has been saved at {plot_dir_full_path}/train_history.jpg")
    except Exception as e:
        logs.write_exception(e)






