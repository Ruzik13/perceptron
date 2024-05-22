import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras
print("We're using TF", tf.__version__)
import keras.models as M
import keras.layers as L
import keras.backend as K
print("We are using Keras", keras.__version__)
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)
K.clear_session()
model = M.Sequential()
model.add(L.Dense(units=128, input_dim=784, activation='elu'))
model.add(L.Dense(units=128, input_dim=784, activation='elu'))
model.add(L.Dense(units=10, activation='softmax'))
model.compile(
    loss='categorical_crossentropy', # минимизируем кросс-энтропию
    optimizer='adam',
    metrics=['accuracy'] # выводим процент правильных ответов
)
x_train_float = x_train.astype(np.float32) 
x_val_float = x_val.astype(np.float32)
model.fit(
    x_train_float.reshape(-1, 28*28),
    y_train_oh,
    batch_size=64, # 64 объекта для подсчета градиента на каждом шаге
    epochs=10, # 10 проходов по датасету
    validation_data=(x_val_float.reshape(-1, 28*28), y_val_oh)
)
