
//consistently gets slighlty over .98 accuracy or .982
import tensorflow as tf
import keras.backend as K
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential();
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'));
//I used a higher .40 dropout meaning that overfitting, or too many right too soon, is avoided
model.add(tf.keras.layers.Dropout(.40))
//I used a softplus activation function here instead of a softmax
model.add(tf.keras.layers.Dense(10, activation='softplus'))

//designed my own metric here
def sqrt(y_true, y_pred):
    return K.sqrt(y_pred)
//used an rmsprop optimizer instead of adam, included my own metric and the loss function in the list of metrics
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'mean_squared_error', 'sparse_categorical_crossentropy', sqrt])
 
 //kept same amount of training at 5 epochs
 model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
