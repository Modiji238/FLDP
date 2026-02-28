import tensorflow as tf
import flwr as fl

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

model=tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    classes=10,
    weights=None
)

optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=32,
    learning_rate=0.001
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    ),
    metrics=["accuracy"]
)

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

x_train = x_train/255.0
x_test  = x_test/255.0

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train,y_train, epochs=1, batch_size=32)
        #loss, accuracy = model.evaluate(x_test,y_test)
        return model.get_weights(), len(x_train), {} #Return empty dict for metrics

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test,y_test)
        return loss, len(x_test), {"accuracy": accuracy}
    

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)


