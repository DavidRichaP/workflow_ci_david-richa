import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.ensemble import RandomForestClassifier


def hyperparam_tuning_autolog(training_source_path, target_col):

    print("reading data")
    df = pd.read_csv(training_source_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("splitting data to train and test set")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    print("data split complete")
    print("X_train size = ", X_train.shape)
    print("y_train size = ", y_train.shape)
    print("X_test size = ", X_test.shape)
    print("y_test size = ", y_test.shape)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'random_state': [42]
    }

    # Set the experiment name
    mlflow.sklearn.autolog()
    mlflow.set_experiment("Wine_Quality_Autolog")

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(run_name=f"RF_est_{params['n_estimators']}_depth_{params['max_depth']}", nested=True):
            model = RandomForestClassifier(**params)

            model.fit(X_train, y_train) 

            model.score(X_test, y_test)


if __name__ == "__main__":
    results = hyperparam_tuning_autolog("winequality-red_preprocessed.csv",
                                        "quality")
