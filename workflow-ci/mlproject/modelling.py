import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.ensemble import RandomForestClassifier


def hyperparam_tuning_autolog(training_source_path, target_col, n_est, max_depth, random_state):

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
        'n_estimators': n_est,
        'max_depth': max_depth,
        'random_state': random_state
    }

    local_mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mlruns"))
    mlflow.set_tracking_uri(f"file://{local_mlruns_path}")
    mlflow.set_experiment(f"{target_col}_Autolog")

    # Set the experiment name
    mlflow.sklearn.autolog()

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(run_name=f"RF_est_{params['n_estimators']}_depth_{params['max_depth']}"):
            model = RandomForestClassifier(**params)

            model.fit(X_train, y_train)

            model.score(X_test, y_test)

            print(f"Finished run with params: {params}")


if __name__ == "__main__":
    n_est = [50, 100, 150]
    max_depth = [None, 10, 20]
    random_state = [42]
    results = hyperparam_tuning_autolog("gender_classif_v7_preprocessed.csv",
                                        "gender",
                                        n_est,
                                        max_depth,
                                        random_state
                                        )
