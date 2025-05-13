nc -vz 127.0.0.1 9876
ssh -NT -R 127.0.0.1:9876:localhost:9876 sam3@134.117.167.139

ssh -N \                                   # no remote shell – just the tunnels
  -L 8080:localhost:80 \                   # →  http://localhost:8080      (NGINX / website front‑end, remote port 80)
  -L 8443:localhost:443 \                  # →  https://localhost:8443     (NGINX HTTPS)
  -L 3000:localhost:3000 \                 # →  http://localhost:3000      (Grafana)
  -L 9090:localhost:9090 \                 # →  http://localhost:9090      (Prometheus)
  -L 16686:localhost:16686 \               # →  http://localhost:16686     (Jaeger UI)
  -L 15672:localhost:15672 \               # →  http://localhost:15672     (RabbitMQ Management)
  -L 8081:localhost:8080 \                 # →  http://localhost:8081      (cAdvisor)
  -L 5173:localhost:5173 \                 # →  http://localhost:5173      (Vite dev server / website hot‑reload)
  -L 5001:localhost:5001 \                 # →  http://localhost:5001      (FastAPI “server” container)
  -L 8002:localhost:8002 \                 # →  http://localhost:8002      (data‑storage service)
  -L 9400:localhost:9400 \                 # →  http://localhost:9400/metrics  (DCGM exporter – GPU metrics)
  -R 127.0.0.1:9876:localhost:9876 \       # ←  Rerun viewer running **on the laptop**, containers connect to 127.0.0.1:9876 on the server
  sam3@134.117.167.139

ssh -N \
  -L 8080:localhost:80 \
  -L 8443:localhost:443 \
  -L 3000:localhost:3000 \
  -L 9090:localhost:9090 \
  -L 16686:localhost:16686 \
  -L 15672:localhost:15672 \
  -L 8081:localhost:8080 \
  -L 5173:localhost:5173 \
  -L 5001:localhost:5001 \
  -L 8002:localhost:8002 \
  -L 9400:localhost:9400 \
  -R 127.0.0.1:9876:localhost:9876 \
  sam3@134.117.167.139

RERUN_BIND=0.0.0.0:9876 rerun viewer

ssh -N \                                   # no remote shell – just the tunnels
  -L 8080:localhost:80 \                   # →  http://localhost:8080      (NGINX / website front‑end, remote port 80)
  -L 8443:localhost:443 \                  # →  https://localhost:8443     (NGINX HTTPS)
  -L 3000:localhost:3000 \                 # →  http://localhost:3000      (Grafana)
  -L 9090:localhost:9090 \                 # →  http://localhost:9090      (Prometheus)
  -L 16686:localhost:16686 \               # →  http://localhost:16686     (Jaeger UI)
  -L 15672:localhost:15672 \               # →  http://localhost:15672     (RabbitMQ Management)
  -L 8081:localhost:8080 \                 # →  http://localhost:8081      (cAdvisor)
  -L 5173:localhost:5173 \                 # →  http://localhost:5173      (Vite dev server / website hot‑reload)
  -L 5001:localhost:5001 \                 # →  http://localhost:5001      (FastAPI “server” container)
  -L 8002:localhost:8002 \                 # →  http://localhost:8002      (data‑storage service)
  -L 9400:localhost:9400 \                 # →  http://localhost:9400/metrics  (DCGM exporter – GPU metrics)
  -R 127.0.0.1:9876:localhost:9876 \       # ←  Rerun viewer running **on the laptop**, containers connect to 127.0.0.1:9876 on the server
  sam3@134.117.167.139

What each flag is doing

| Flag                                 | Direction                     | Remote&nbsp;port | Local&nbsp;port | Purpose                                                                                                    |
|--------------------------------------|-------------------------------|------------------|-----------------|------------------------------------------------------------------------------------------------------------|
| `-L 8080:localhost:80`               | server → laptop               | 80               | 8080            | NGINX HTTP (website front‑end)                                                                             |
| `-L 8443:localhost:443`              | server → laptop               | 443              | 8443            | NGINX HTTPS (if you have TLS)                                                                              |
| `-L 3000:localhost:3000`             | server → laptop               | 3000             | 3000            | Grafana dashboard                                                                                          |
| `-L 9090:localhost:9090`             | server → laptop               | 9090             | 9090            | Prometheus metrics UI                                                                                      |
| `-L 16686:localhost:16686`           | server → laptop               | 16686            | 16686           | Jaeger distributed‑tracing UI                                                                              |
| `-L 15672:localhost:15672`           | server → laptop               | 15672            | 15672           | RabbitMQ Management                                                                                         |
| `-L 8081:localhost:8080`             | server → laptop               | 8080             | 8081            | cAdvisor container metrics (picked 8081 locally to avoid clash with NGINX tunnel)                           |
| `-L 5173:localhost:5173`             | server → laptop               | 5173             | 5173            | Vite hot‑reload dev server (website)                                                                        |
| `-L 5001:localhost:5001`             | server → laptop               | 5001             | 5001            | FastAPI back‑end (“server” container)                                                                       |
| `-L 8002:localhost:8002`             | server → laptop               | 8002             | 8002            | `data_storage` service                                                                                      |
| `-L 9400:localhost:9400`             | server → laptop               | 9400             | 9400            | NVIDIA DCGM exporter GPU metrics                                                                            |
| `-R 127.0.0.1:9876:localhost:9876`   | laptop → server (reverse)     | 9876             | 9876            | Lets containers reach **your** local Rerun viewer via `127.0.0.1:9876` on the server                        |
Why 8080/8443 and 8081?
Binding to ports < 1024 on the laptop would require sudo. Using 8080/8443/8081 keeps everything un‑privileged.

How to use
	1.	Run the command from your laptop/desktop, not on the server.
	2.	Keep that terminal open; as long as the SSH session is up, all the URLs above work at http://localhost:PORT.
	3.	Point your browser to, for example, http://localhost:3000 and you should get Grafana instantly without the sluggish hair‑pin round‑trip through the campus router.
