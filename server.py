import flwr as fl
import time
from report import evaluate_metrics_aggregation_fn

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
)

start = time.time()
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy
)
end = time.time()

print("Total FL training time:", end - start)