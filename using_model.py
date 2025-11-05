import mlflow

model_name = 'cifar_net'
model_version = 'latest'

model = mlflow.tensorflow.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

print(model.summary())
