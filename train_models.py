import tensorflow as tf
import mlflow

mlflow.tensorflow.autolog()
mlflow.set_experiment("CNN Tuning")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

params = [
    {'conv_filters': 32, 'conv_layers': 2, 'activation': 'relu'},
    {'conv_filters': 16, 'conv_layers': 3, 'activation': 'sigmoid'},
    {'conv_filters': 24, 'conv_layers': 2, 'activation': 'relu'},
]

def create_model(params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

    conv_filters = params.get('conv_filters', 32)
    conv_layers = params.get('conv_layers', 2)
    activation = params.get('activation', 'relu')

    for _ in range(conv_layers):
        model.add(tf.keras.layers.Conv2D(conv_filters, kernel_size=(3, 3), activation=activation))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        conv_filters *= 2

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

for param in params:
    with mlflow.start_run() as run:
        conv_filters = param.get('conv_filters')
        conv_layers = param.get('conv_layers')
        activation = param.get('activation')
        run_name = f"cnn-{conv_filters}-{conv_layers}-{activation}"
        
        model = create_model(param)
        hist = model.fit(x_train, y_train, epochs = 5,
                         validation_data=(x_test, y_test),
                         verbose=2
                         )
        
        final_metrics = {
            'val_accuracy': hist.history['val_accuracy'][-1],
            'val_loss': hist.history['val_loss'][-1]
        }

        print(f"final metrics: {final_metrics['val_accuracy']}")
