import datetime

from airflow.decorators import dag, task



markdown_text = """
### ETL Process for mnist_784

This DAG extracts information from the original mnist_784 library.
It preprocesses the data by creating dummy variables and scaling numerical features, representing pixels 28x28x1
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 70/30 and they are stratified.
"""


default_args = {
    'owner': "Carlos Rodriguez",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="process_etl_mnist_784_data",
    description="ETL process for mnist_784 data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "mnist_784"],
    default_args=default_args,
    catchup=False,
)
def process_etl_mnist_784_data():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["ucimlrepo==0.0.3",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    
    def get_data():
        """
        Load the raw data from UCI repository
        """
        import awswrangler as wr
        from ucimlrepo import fetch_ucirepo
        from airflow.models import Variable
        from sklearn.datasets import fetch_openml

        import pandas as pd
        import numpy as np

        # fetch dataset
        #mnist_784_dataset = fetch_ucirepo(id=45)

        try:

            data_path = "s3://data/raw/mnist_784.csv"
            #dataframe = mnist_784_dataset.data.original

            #target_col = Variable.get("target_col_mnist_784")

            mnist = fetch_openml('mnist_784', as_frame = False)

            dataframe = pd.DataFrame(mnist.data, columns=mnist.feature_names)
            target_col = pd.DataFrame(mnist.target, columns=['label'])

            dataframe['label'] = target_col['label']

            # Replace level of mnist_784 decease to just distinguish presence 
            # (values 1,2,3,4) from absence (value 0).
            #dataframe.loc[dataframe[target_col] > 0, target_col] = 1

            wr.s3.to_csv(df=dataframe,
                        path=data_path,
                        index=False)
            
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                print(e)
                raise e



    @task.virtualenv(
        task_id="train_model",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def train_model():

        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import awswrangler as wr
        import pandas as pd
        import numpy as np
        import sklearn

        from sklearn.tree import DecisionTreeClassifier
        from sklearn import metrics
        from sklearn.metrics import accuracy_score

        from airflow.models import Variable

        from sklearn.model_selection import train_test_split


        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)
            
        try:

            data_original_path = "s3://data/raw/mnist_784.csv"
            #data_end_path = "s3://data/raw/mnist_784_dummies.csv"
            dataframe = wr.s3.read_csv(data_original_path)

            data_dict = dataframe.to_dict()

            data_string = json.dumps(data_dict, indent=2)

            client = boto3.client('s3')

            client.put_object(
                Bucket='data',
                Key='data_info/mnist_784.json',
                Body=data_string
                #Body=dataframe.columns
            )
            
            X_digits=dataframe.drop(['label'], axis=1).to_numpy()
            y_digits=dataframe['label'].to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(X_digits,
                                                            y_digits,
                                                            test_size=0.3,
                                                            random_state=32)
            
            save_to_csv(X_train, "s3://data/final/train/mnist_784_X_train.csv")
            save_to_csv(X_test, "s3://data/final/test/mnist_784_X_test.csv")
            save_to_csv(y_train, "s3://data/final/train/mnist_784_y_train.csv")
            save_to_csv(y_test, "s3://data/final/test/mnist_784_y_test.csv")

            mlflow.set_tracking_uri('http://mlflow:5000')
            experiment = mlflow.set_experiment("mnist_784")

            mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                            experiment_id=experiment.experiment_id,
                            tags={"experiment": "etl", "dataset": "mnist_784"},
                            log_system_metrics=True)

            mlflow_dataset = mlflow.data.from_pandas(dataframe,
                                                    source="https://archive.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits",
                                                    targets=dataframe['label'],
                                                    name="mnist_784_data_complete")

            mlflow.log_input(mlflow_dataset, context="Dataset")

            list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

            tree_clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)
            tree_clf.fit(X_train, y_train)
            y_pred = tree_clf.predict(X_test)
            print(' ')
            print(f"accuracy_score: {accuracy_score(y_test, y_pred)}")
            print(f"recall_score: {metrics.recall_score(y_test, y_pred, average = 'macro')}")
            print(f"f1_score: {metrics.f1_score(y_test, y_pred, average = 'macro')}")

            mlflow.set_tracking_uri('http://mlflow:5000')
            experiment = mlflow.set_experiment("mnist_784")

            list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

            with mlflow.start_run(run_id=list_run[0].info.run_id):

                mlflow.log_param("Train observations", X_train.shape[0])
                mlflow.log_param("Test observations", X_test.shape[0])
                #mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_)
                #mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
                #mlflow.log_param("Standard Scaler scale values", sc_X.scale_)

        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                print(e)
                raise e

    get_data() >> train_model()

dag = process_etl_mnist_784_data()