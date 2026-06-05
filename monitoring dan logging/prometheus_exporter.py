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
        self.prediction_counter = Counter('model_predictions_total',
                                          'Total inferences served'
                                          )

        self.latency_histogram = Histogram('model_prediction_latency_seconds',
                                           'Inference execution latency duration'
                                           )

        self.error_counter = Counter('model_errors_total', 'Total inference crashes')

        self.alcohol_gauge = Gauge('incoming_feature_alcohol_value',
                                   'snapshot of the alcohol feature value from the latest request row'
                                   )

        self.quality_gauge = Gauge('model_predicted_quality_value',
                                   'snapshot of the average predicted wine quality from the latest batch'
                                   )

    def predict(self, context, model_input):
        start_time = time.time()
        self.prediction_counter.inc()

        try:
            # 1. NEW METRIC: Track input feature (Data Drift Proxy)
            # Grabs the last row of the 'alcohol' column from the incoming DataFrame
            if 'alcohol' in model_input.columns:
                self.alcohol_gauge.set(float(model_input['alcohol'].iloc[-1]))

            # Your original prediction logic
            predictions = self.model.predict(model_input)

            # 2. NEW METRIC: Track output prediction (Prediction Drift Proxy)
            if len(predictions) > 0:
                self.quality_gauge.set(float(predictions[0]))

            return predictions

        except Exception as e:
            # 3. NEW METRIC: If the model breaks, tick the error counter up
            self.error_counter.inc()
            raise e

        finally:
            # Ensures latency is recorded even if things went wrong
            duration = time.time() - start_time
            self.latency_histogram.observe(duration)

        return predictions


if __name__ == "__main__":
    print("memulai pembacaan model")
    # arahkan variabel ini ke path dari ID run yang ingin digunakan
    AUTOLOG_MODEL_URI = "./mlruns/968791154692521892/e55626dec82a478d81efdd26b9f03913/artifacts/model"

    # Instantiate our wrapper
    wrapped_model = MonitoredWrapper(autolog_model_uri=AUTOLOG_MODEL_URI)

    # Save the wrapped model into a local folder
    output_dir = "monitored_model_local"
    mlflow.pyfunc.save_model(path=output_dir, python_model=wrapped_model)
    print(f"Success! Monitored wrapper saved to folder: ./{output_dir}")
