import mlflow

def main():

    with mlflow.start_run() as mlrun:
        mlflow.run('.', 'get_data', env_manager='local')
        mlflow.run('.', 'base_model_creation', env_manager='local')
        mlflow.run('.', 'model_training', env_manager='local')

if __name__ == '__main__':
    main()