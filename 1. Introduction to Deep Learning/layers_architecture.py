import tensorflow as tf

layer = tf.keras.layers.Dense(units=2)

# Below is what the above does
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        # Initialize weights and bias
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Forward progate inputs
        z = tf.matmul(inputs, self.W) + self.b

        # Feed through a non-linear activation
        output = tf.math.sigmoid(z)
        return output


# Multi output perceptron -> Tensorflow
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(1)
])

# -> Pytorch
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(m,n),
    nn.ReLU(),
    nn.Linear(n,1)
)


# Deep Neural Network -> Tensorflow
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n1),
    tf.keras.layers.Dense(n2),
    # Add more layers as needed
    tf.keras.layers.Dense(2)
])

# -> Pytorch
import torch
model = nn.Sequential(
    nn.Linear(m,n1),
    nn.ReLU(),
    nn.Linear(n1,n2),
    nn.ReLU(),
    # Add more layers as needed
    nn.Linear(n2,2)
)