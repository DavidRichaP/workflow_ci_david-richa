menjalankan inference dan prometheus

# jalankan untuk menghasilkan export local model
python prometheus_exporter.py

# arahkan terminal ke folder parent yang menyimpan model local yang sudah di export oleh prometheus_client
cd monitoring dan logging

# jalankan untuk menyalakan server local
mlflow models serve -m "monitored_model_local" -p 5001 --env-manager=local --host 0.0.0.0

# arahkan terminal ke folder parent yang menyimpan prometheus.yml
cd monitoring dan logging

# Start Prometheus (port 9090 untuk keluar/masuk data)
docker run -d --name=prometheus -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

# Start Grafana
docker run -d --name=grafana -p 3000:3000 grafana/grafana