import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report


def hyperparam_tuning(training_source_path, target_col):

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
    mlflow.set_experiment("Wine_Quality_Manual_Tuning")

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(run_name=f"RF_est_{params['n_estimators']}_depth_{params['max_depth']}", nested=True):
            # 1. Log Parameters (Equivalent to autolog)
            mlflow.log_params(params)

            # 2. Train Model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # 3. Predict and Calculate Metrics
            preds = model.predict(X_test)
            precision = precision_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            f1_macro_avg = classification_report(y_test, preds, output_dict=True)['macro avg']['f1-score']
            f1_weighted_avg = classification_report(y_test, preds, output_dict=True)['weighted avg']['f1-score']

            # 4. Log Metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("macro average F1 score", f1_macro_avg)
            mlflow.log_metric("weighted average F1 score", f1_weighted_avg)

            # 5. Log Model Artifact
            mlflow.sklearn.log_model(model, "random_forest_model")

            print(f"Finished run with params: {params} | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    results = hyperparam_tuning("winequality-red_preprocessed.csv", "quality")
