import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from typing import Union, List, Tuple
from sklearn.utils import shuffle
from collections import OrderedDict

tf.get_logger().setLevel('ERROR')
SEED = 0

# Initializers
INITIALIZERS = OrderedDict(
    RandomNormal=tf.random_normal_initializer(mean=0, stddev=0.1, seed=SEED),
    TruncatedNormal=tf.truncated_normal_initializer(
        mean=0, stddev=0.1, seed=SEED),
    HeNormal=tf.contrib.layers.variance_scaling_initializer(seed=SEED),
    XavierNormal=tf.glorot_normal_initializer(seed=SEED))

# Optimizers
OPTIMIZERS = OrderedDict(
    GradientDescent=tf.train.GradientDescentOptimizer,
    Adam=tf.train.AdamOptimizer,
    Adagrad=tf.train.AdagradOptimizer)

# Activations
ACTIVATIONS = OrderedDict(Relu=tf.nn.relu)


class Conv2d:
    def __init__(self, name, shape, strides, padding, activation):
        self.name = name
        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.filter_weights = None
        self.bias = None

    def compile(self, initializer: str):
        """ Initialize the layer in the tensorflow graph. """
        init = INITIALIZERS[initializer]
        self.filter_weights = tf.Variable(init(self.shape, dtype=tf.float32),
                                          name=f"{self.name}_W")
        self.bias = tf.Variable(
            tf.zeros(self.shape[-1]), name=f"{self.name}_b")

    def forward(self, x):
        """ Forward propagation through the layer.

        Args:
            x: features input
        Returns:
            the layer output
        """
        kwargs = dict(filter=self.filter_weights,
                      strides=self.strides,
                      padding=self.padding)
        weights = tf.nn.conv2d(input=x, **kwargs)
        if self.activation is None:
            conv = tf.add(weights, self.bias, name=self.name)
        else:
            activation = ACTIVATIONS[self.activation.capitalize()]
            conv = activation(tf.add(weights, self.bias), name=self.name)
        return conv


class Pool:
    def __init__(self, name, shape, strides, padding, pooling_type="MAX"):
        self.name = name
        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.type = pooling_type

    def forward(self, x):
        """ Forward propagation through the layer.

        Args:
            x: features input
        Returns:
            the layer output
        """
        kwargs = dict(ksize=self.shape,
                      strides=self.strides,
                      padding=self.padding,
                      name=self.name)
        if self.type == "MAX":
            return tf.nn.max_pool(x, **kwargs)
        elif self.type == "AVG":
            return tf.nn.avg_pool(x, **kwargs)


class Dense:
    def __init__(self, name, shape, activation=None, dropout=False):
        self.name = name
        self.shape = shape
        self.activation = activation
        self.has_dropout = dropout
        self.dropout_active = False
        self.weights = None
        self.bias = None

    def compile(self, initializer, activate_dropout=False):
        """ Initialize the Dense layer in the tensorflow graph.

        Args:
            initializer: initializer alias for the tf.Variable initialization
            activate_dropout: if True, the dropout will be activated
        """
        if activate_dropout is True and self.has_dropout:
            self.dropout_active = True
        self.weights = tf.Variable(
            INITIALIZERS[initializer](self.shape), name=f"{self.name}_W")
        self.bias = tf.Variable(
            tf.zeros(self.shape[-1]), name=f"{self.name}_b")

    def forward(self, x, keep_prob, logits=False):
        """ Forward propagation through the layer.

        Args:
            x: features input
            keep_prob: dropout keep probability
            logits: if True, the output tensor will be named "logits"
        Returns:
            the activated output features or logits
        """
        name = self.name if logits is False else "logits"
        weights = self.weights
        bias = self.bias

        if self.dropout_active is True:
            if self.activation:
                activation = ACTIVATIONS[self.activation.capitalize()]
                fc = activation(tf.add(tf.matmul(x, weights), bias))
            else:
                fc = tf.add(tf.matmul(x, weights), bias)
            return tf.nn.dropout(fc, keep_prob, seed=SEED, name=name)
        else:
            if self.activation:
                activation = ACTIVATIONS[self.activation.capitalize()]
                return activation(
                    tf.add(tf.matmul(x, weights), bias), name=name)
            else:
                return tf.add(tf.matmul(x, weights), bias, name=name)


class Flatten:
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        """ Simply flatten the weights of the previous layer. """
        return tf.reshape(x, shape=(-1, self.size))


class Concat:
    def __init__(self, layers: List):
        self.layers = layers

    @staticmethod
    def forward(concat_tensors):
        flattened_tensors = []
        for tensor in concat_tensors:
            flattenend_size = int(np.prod(tensor.shape[1:]))
            flattened_tensor = tf.reshape(tensor, shape=(-1, flattenend_size))
            flattened_tensors.append(flattened_tensor)
        return tf.concat(flattened_tensors, 1)


LayerUnion = Union[Conv2d, Pool, Dense, Flatten, Concat]


class Model:

    def __init__(self, name: str = None):
        self.name = name
        self.layers = []
        self.recent_train_pars = OrderedDict()
        self.dropout_active = False

    def compile(self, layers: List, initializer: str, activate_dropout=True):
        """ Initialize the tensorflow graph.

        Args:
            layers: list containing the layer classes
            initializer: initializer alias for tf.Variable initialization
            activate_dropout: if True, dropout layers will be activated
        """

        for layer in layers:  # type: LayerUnion
            if isinstance(layer, Conv2d):
                layer.compile(initializer)
            elif isinstance(layer, Dense):
                layer.compile(initializer, activate_dropout)
            self.layers.append(layer)

            if isinstance(layer, Dense) and layer.dropout_active is True:
                self.dropout_active = True

        # Update training pars for plotting titles
        self.recent_train_pars.update(OrderedDict(initializer=initializer))

    def train(self, train_data: Tuple, valid_data: Tuple, optimizer: str,
              learning_rate: float, epochs: int, batch_size: int,
              keep_prob=1.0, verbose=2, save=False):
        """ Train the network.

        Args:
            train_data:
                tuple of normalized training images and one-hot encoded labels
            valid_data:
                tuple of normalized validation images and one-hot encoded labels
            optimizer:
                alias of the optimizer to use
            learning_rate:
                learning rate to use during optimization
            epochs:
                the number of epochs to train
            batch_size:
                the minibatch size
            keep_prob:
                dropout keep probability (for possible Dense layer dropouts)
            verbose:
                silent training (0), print every 10th epoch (1), print all (2)
            save:
                if True, save checkpoints
        """

        # Update the recemt train pars for the plot titles
        self.recent_train_pars.update(OrderedDict(optimizer=optimizer,
                                                  learning_rate=learning_rate,
                                                  keep_prob=keep_prob,
                                                  batch_size=batch_size))

        x_train, y_train = train_data
        x_valid, y_valid = valid_data

        # Create placeholders
        n_samples, n_height, n_width, n_channels = x_train.shape
        x, y, keep_prob_placeholder = self.create_placeholders(
            n_height, n_width, n_channels)

        # Forward propagation
        logits = self.forward_propagation(x, keep_prob_placeholder)

        # One-hot encode labels
        y_one_hot = tf.one_hot(y, logits.shape[-1], name="y_one_hot")

        # Compute loss
        loss = self.compute_loss(logits, y_one_hot)

        # Backward propagation
        optimizer = OPTIMIZERS[optimizer]
        optimize = optimizer(learning_rate=learning_rate).minimize(loss)

        # Evaluation
        correct_predictions = tf.equal(
            tf.argmax(logits, 1), tf.argmax(y_one_hot, 1), name="correct_pred")
        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32), name="accuracy")

        saver = tf.train.Saver(max_to_keep=None)

        # Global initialization of all variables (weights, biases, ...)
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:

            # Initialization
            session.run(init)
            train_losses = []
            train_accuracies = []
            valid_accuracies = []

            # Loop over epochs
            for epoch in range(1, epochs + 1):

                total_train_loss = 0
                total_train_acc = 0
                total_valid_acc = 0

                # Shuffle training data and draw minibatches at each epoch
                x_train, y_train = shuffle(x_train, y_train, random_state=SEED)
                train_batches = self.draw_minibatches(
                    x_train, y_train, batch_size)

                # Additionally draw minibatches for validation accuracy
                x_valid, y_valid = shuffle(x_valid, y_valid, random_state=SEED)
                valid_batches = self.draw_minibatches(
                    x_valid, y_valid, batch_size)

                # Training and determination of training loss
                for x_train_batch, y_train_batch in train_batches:
                    _, train_loss = session.run(
                        fetches=[optimize, loss],
                        feed_dict={x: x_train_batch,
                                   y: y_train_batch,
                                   keep_prob_placeholder: keep_prob})
                    total_train_loss += (train_loss * len(x_train_batch))
                total_train_loss /= len(x_train)

                # Determination of training and validation accuracy
                for x_train_batch, y_train_batch in train_batches:
                    train_acc = session.run(
                        fetches=accuracy,
                        feed_dict={x: x_train_batch,
                                   y: y_train_batch,
                                   keep_prob_placeholder: 1.0})
                    total_train_acc += (train_acc * len(x_train_batch))
                total_train_acc /= len(x_train)

                for x_valid_batch, y_valid_batch in valid_batches:
                    valid_acc = session.run(
                        fetches=accuracy,
                        feed_dict={x: x_valid_batch,
                                   y: y_valid_batch,
                                   keep_prob_placeholder: 1.0})
                    total_valid_acc += (valid_acc * len(x_valid_batch))
                total_valid_acc /= len(x_valid)

                if verbose == 2 or verbose == 1 and epoch % 10 == 0:
                    print(f"Epoch {epoch:2}/{epochs}:   "
                          f"Train Loss: {total_train_loss:.4f}   "
                          f"Train Acc: {total_train_acc:.4f}   "
                          f"Valid Acc: {total_valid_acc:.4f}")

                train_losses.append(total_train_loss)
                train_accuracies.append(total_train_acc)
                valid_accuracies.append(total_valid_acc)

                if save:
                    saver.save(session, f"models/{self.name}.ckpt",
                               global_step=epoch)

        return train_losses, train_accuracies, valid_accuracies

    @staticmethod
    def evaluate(features, labels):

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")

        session = tf.get_default_session()
        accuracy = session.run(accuracy, feed_dict={x: features,
                                                    y: labels,
                                                    keep_prob: 1.0})
        return accuracy

    @staticmethod
    def create_placeholders(n_h: int, n_w: int, n_c: int):
        """ Create the placeholders for input features and labels.

        The feature and label placeholders will be of shape::

            x.shape -> (None, n_h, n_w, n_c)
            y.shape -> (None)

        Args:
            n_h: pixel height of the input image
            n_w: pixel width of the input image
            n_c: number of channels of the input image

        Returns:
            (x, y, keep_prob) placeholder tuple
        """
        x = tf.placeholder(tf.float32, shape=(None, n_h, n_w, n_c), name="x")
        y = tf.placeholder(tf.int64, shape=(None,), name="y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        return x, y, keep_prob

    def forward_propagation(self, x, keep_prob):
        """
        Propagate the feature batch through the network and return the logits.

        Args:
            x: feature batch (placeholder)
            keep_prob: dropout keep probability (placeholder)
        """

        # Determine names of tensors that eventually have to be concatenated
        concat_tensors = []
        concat_tensor_names = []
        for layer in self.layers:
            if isinstance(layer, Concat):
                layer_names = layer.layers
                concat_tensor_names = [f"{name}:0" for name in layer_names]

        # Forward propagation
        for i, layer in enumerate(self.layers):  # type: LayerUnion
            if isinstance(layer, Dense):
                logits = True if i == len(self.layers) - 1 else False
                x = layer.forward(x, keep_prob, logits=logits)
            elif isinstance(layer, Concat):
                x = layer.forward(concat_tensors)
            else:
                x = layer.forward(x)
            # Store tensors that have to be concatenated
            if x.name in concat_tensor_names:
                concat_tensors.append(x)
        logits = x
        return logits

    @staticmethod
    def compute_loss(logits, labels):
        """ Compute the loss for the given logits and labels.

        Args:
            logits: Output of the last layer before activation
            labels: one-hot encoded labels
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
        return loss

    @staticmethod
    def draw_minibatches(x, y, batch_size):

        minibatches = []
        samples = x.shape[0]

        # Complete mini batches
        complete_batches = samples // batch_size
        for i in range(0, complete_batches):
            minibatch_X = x[i * batch_size: i * batch_size + batch_size]
            minibatch_Y = y[i * batch_size: i * batch_size + batch_size]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)

        # Eventually uncomplete last minibatch
        if samples % batch_size != 0:
            minibatch_X = x[complete_batches * batch_size: samples]
            minibatch_Y = y[complete_batches * batch_size: samples]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)

        return minibatches

    @staticmethod
    def restore(checkpoint):
        session = tf.get_default_session()
        saver = tf.train.import_meta_graph(f"{checkpoint}.meta")
        saver.restore(session, f"{checkpoint}")

    # TODO:
    def predict(self, features, k=5):

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        logits = graph.get_tensor_by_name("logits: 0")

        prob_op = tf.nn.softmax(logits)
        top_k_prob_op = tf.nn.top_k(prob_op, k=k)

        session = tf.get_default_session()
        top_k_probs = session.run(fetches=top_k_prob_op,
                                  feed_dict={x: features, keep_prob: 1.0})

        return top_k_probs

    @property
    def trainable_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters
