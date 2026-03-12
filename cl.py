import flwr as fl
import tensorflow as tf
from model import create_model
from data_load import load_partition
import sys
from sklearn.metrics import precision_score, recall_score, f1_score

client_id = int(sys.argv[1])
NUM_CLIENTS = 5

X_train, X_test, y_train, y_test = load_partition(client_id, NUM_CLIENTS)
# print("Client", client_id)
# print("Fraud ratio:", sum(y_train)/len(y_train))
model = create_model(X_train.shape[1])

class FraudClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=0)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return loss, len(X_test), {
            "accuracy": acc,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

fl.client.start_numpy_client(server_address="localhost:8080", client=FraudClient())