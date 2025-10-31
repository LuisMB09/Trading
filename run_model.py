import tensorflow as tf
import mlflow

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model_version = 'latest'
model_name = 'cifar_net'

model = mlflow.tensorflow.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )

print("Model loaded successfully.")
print(model.summary())

y_pred = model.predict(x_test)
predicted_classes = tf.argmax(y_pred, axis=1)
