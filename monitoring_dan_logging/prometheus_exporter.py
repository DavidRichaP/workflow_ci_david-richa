import mlflow.pyfunc
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time


class MonitoredWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, autolog_model_uri):
        self.autolog_model_uri = autolog_model_uri

    def load_context(self, context):
        # loading artifact
        self.model = mlflow.pyfunc.load_model(self.autolog_model_uri)

        # prometheus scraper on localhost 8000
        try:
            start_http_server(8000)
            print("--> Prometheus exporter listening locally on http://localhost:8000")
        except OSError:
            print("--> Prometheus exporter already active on port 8000")

        # custom monitoring metrics

        # counts individual rows
        self.prediction_counter = Counter(
            'model_prediction_count_total',
            'Total number of individual text reviews processed by the model'
        )

        # counts batch inferences made
        self.batch_counter = Counter('model_batches_total',
                                     'Total batch request inference served'
                                     )

        # adds histogram for each prediction's latency
        custom_buckets = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

        self.latency_histogram = Histogram('model_prediction_latency_seconds',
                                           'Inference execution latency duration',
                                           buckets=custom_buckets
                                           )

        # counts total model error
        self.error_counter = Counter('model_errors_total', 'Total inference crashes')

        # gauges incoming feature
        self.alcohol_gauge = Gauge('incoming_feature_alcohol_value',
                                   'snapshot of the alcohol feature value from the latest request row'
                                   )

        # gauges predicted quality value
        self.quality_gauge = Gauge('model_predicted_quality_value',
                                   'snapshot of the average predicted wine quality from the latest batch'
                                   )

    def predict(self, context, model_input):
        start_time = time.time()

        batch_size = model_input.shape[0] if hasattr(model_input, 'shape') else len(model_input)
        self.prediction_counter.inc(batch_size)

        try:
            # 1. NEW METRIC: Track input feature (Data Drift Proxy)
            # Grabs the last row of the 'alcohol' column from the incoming DataFrame
            if hasattr(model_input, 'columns') and 'alcohol' in model_input.columns:
                self.alcohol_gauge.set(float(model_input['quality'].iloc[-1]))

            # Your original prediction logic
            predictions = self.model.predict(model_input)

            # Calculate the overall positive ratio of the whole batch
            if len(predictions) > 0:
                import numpy as np
                batch_average = np.mean(predictions)  # e.g., 0.85 means 85% positive
                self.quality_gauge.set(float(batch_average))

            return predictions

        except Exception as e:
            # 3. NEW METRIC: If the model breaks, tick the error counter up
            self.error_counter.inc()
            raise e

        finally:
            # Ensures latency is recorded even if things went wrong
            duration = time.time() - start_time
            self.latency_histogram.observe(duration)


if __name__ == "__main__":
    print("memulai pembacaan model")
    # arahkan variabel ini ke path dari ID run yang ingin digunakan
    AUTOLOG_MODEL_URI = "./mlruns/models/968791154692521892/e55626dec82a478d81efdd26b9f03913/artifacts/model"

    # Instantiate our wrapper
    wrapped_model = MonitoredWrapper(autolog_model_uri=AUTOLOG_MODEL_URI)

    # Save the wrapped model into a local folder
    output_dir = "monitored_model_local"
    mlflow.pyfunc.save_model(path=output_dir, python_model=wrapped_model)
    print(f"Success! Monitored wrapper saved to folder: ./{output_dir}")
