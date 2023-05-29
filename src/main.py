import mlflow

def main():

    with mlflow.start_run() as mlrun:
        mlflow.run('.', 'test', env_manager='local')

if __name__ == '__main__':
    main()