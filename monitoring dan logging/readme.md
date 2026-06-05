menjalankan inference dan prometheus

# jalankan untuk menghasilkan export local model
python prometheus_exporter.py

# arahkan terminal ke folder parent yang menyimpan model local yang sudah di export oleh prometheus
cd monitoring dan logging

# jalankan untuk menyalakan server local
mlflow models serve -m "monitored_model_local" -p 5001 --env-manager=local

# 3. Start Prometheus (port 9090 untuk keluar data)
docker run -d --name=prometheus -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

# 4. Start Grafana
docker run -d --name=grafana -p 3000:3000 grafana/grafana