from time import time

import keras
import numpy
import tensorflow as tf
from keras.callbacks import TensorBoard

HIDDEN_SIZE = 128
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


class Policy:

    def __init__(self, observations_size, number_of_actions):
        self.model = keras.Sequential([
            keras.layers.Dense(HIDDEN_SIZE, activation='relu', input_dim=observations_size),
            keras.layers.Dense(number_of_actions, activation='softmax')
        ])

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def select_action(self, observations):
        action_probabilities = self.model.predict(numpy.asarray([observations]), verbose=0)
        return numpy.random.choice(len(action_probabilities[0]), p=action_probabilities[0])

    def train(self, train_observations, train_actions):
        self.model.fit(train_observations, train_actions, epochs=30, verbose=0, callbacks=[tensorboard])
