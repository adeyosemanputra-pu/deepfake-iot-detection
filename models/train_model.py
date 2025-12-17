import tensorflow as tf

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_model():
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Dummy dataset example
    import numpy as np
    x_train = np.random.rand(100,128,128,3)
    y_train = np.random.randint(0,2,100)
    model.fit(x_train, y_train, epochs=5, batch_size=16)

    model.save("models/cnn_model.h5")

    # Export to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("models/model.tflite", "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    train_model()
