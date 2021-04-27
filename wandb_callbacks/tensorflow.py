import wandb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K


class ActivationCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_data, layer_name):
        '''
        validation_data: tuple of form (sample_images, sample_labels).
        layer_name: string of the layer of whose features we are interested in.
        '''
        super(ActivationCallback, self).__init__()
        self.validation_data = validation_data
        self.layer_name = layer_name

    def on_epoch_end(self, logs, epoch):
        # Build intermediate layer with the target layer
        self.intermediate_model = keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.layer_name).output)
        # Unpack validation data
        images, labels = self.validation_data
        for image, label in zip(images, labels):
            # Compute output activation of the provided layer name
            img = np.expand_dims(image, axis=0)
            features = self.intermediate_model.predict(img)

            features = features.reshape(features.shape[1:])
            features = np.rollaxis(features, 2, 0)
            # Prepare the plot to be logged to wandb
            n_features = features.shape[0]
            # get all divisors of the n. of features
            divisors = self.__get_divisors(n_features)
            # -2, because the largest is equal to the number
            biggest_div = divisors[-2]
            fig, axs = plt.subplots(
                nrows=int(n_features/biggest_div),
                ncols=biggest_div,
                figsize=(15, 8))
            c = 0
            for i in range(int(n_features/biggest_div)):
                for j in range(biggest_div):
                    axs[i][j].imshow(features[c], cmap='gray')
                    axs[i][j].set_xticks([])
                    axs[i][j].set_yticks([])
                    c += 1
            wandb.log({"features_labels_{}".format(label): plt})
            plt.close()

    @staticmethod
    def __get_divisors(n):
        divisors = []
        for i in range(1, int(n / 2) + 1):
            if n % i == 0:
                divisors.append(i)
        divisors.append(n)

        return divisors


class DeadReluDetector(tf.keras.callbacks.Callback):
    """
    Reports the number of dead ReLUs after each training epoch
    ReLU is considered to be dead if it did not fire once for entire training set

    # Arguments
        x_train: Training dataset to check whether or not neurons fire
        verbose: verbosity mode
            True means that even a single dead neuron triggers a warning message
            False means that only significant number of dead neurons (10% or more)
            triggers a warning message
    """

    def __init__(self, x_train, verbose=False):
        super(DeadReluDetector, self).__init__()

        self.x_train = x_train
        self.verbose = verbose
        self.dead_neurons_share_threshold = 0.1

    @staticmethod
    def is_relu_layer(layer):
        # Should work for all layers with relu
        # activation. Tested for Dense and Conv2D
        return layer.get_config().get('activation', None) == 'relu'

    def get_relu_activations(self):
        model_input = self.model.input

        funcs = {}
        for index, layer in enumerate(self.model.layers):
            if not layer.get_weights():
                continue
            funcs[index] = keras.models.Model(
                inputs=model_input,
                outputs=layer.output)

        layer_outputs = {}
        for index, func in funcs.items():
            layer_outputs[index] = tf.dtypes.cast(
                func(self.x_train)[0], tf.float64)

        for layer_index, layer_activations in layer_outputs.items():
            if self.is_relu_layer(self.model.layers[layer_index]):
                layer_name = self.model.layers[layer_index].name
                # layer_weight is a list [W] (+ [b])
                layer_weight = self.model.layers[layer_index].get_weights()

                # with kernel and bias, the weights are saved as a list [W, b].
                # If only weights, it is [W]
                if not isinstance(layer_weight, list):
                    raise ValueError("'Layer_weight' should be a list, "
                                     "but was {}".format(type(layer_weight)))

                # there are no weights for current layer; skip it
                # this is only legitimate if layer is "Activation"
                if len(layer_weight) == 0:
                    continue

                layer_weight_shape = np.shape(layer_weight[0])
                yield (layer_index,
                       layer_activations,
                       layer_name,
                       layer_weight_shape)

    def on_epoch_end(self, epoch, logs={}):
        for relu_activation in self.get_relu_activations():
            layer_index, activation_values, layer_name, layer_weight_shape = relu_activation

            shape_act = activation_values.shape

            weight_len = len(layer_weight_shape)
            act_len = len(shape_act)

            # should work for both Conv and Flat
            if K.image_data_format() == 'channels_last':
                # features in last axis
                axis_filter = -1
            else:
                # features before the convolution axis, for weight_
                # len the input and output have to be subtracted
                axis_filter = -1 - (weight_len - 2)
            total_featuremaps = shape_act[axis_filter]

            axis = []
            for i in range(act_len):
                if (i != axis_filter) and (
                        i != (len(shape_act) + axis_filter)):
                    axis.append(i)
            axis = tuple(axis)

            dead_neurons = np.sum(np.sum(activation_values, axis=axis) == 0.0,
                                  dtype='double')
            dead_neurons_share = float(dead_neurons) / float(total_featuremaps)
            if self.verbose and dead_neurons > 0 \
                    or dead_neurons_share >= self.dead_neurons_share_threshold:
                # print the warning
                str_warning = (
                    'Layer {} (#{}) has {} '
                    'dead neurons ({:.2%})!').format(
                    layer_name,
                    layer_index,
                    dead_neurons,
                    dead_neurons_share)
                print(str_warning)

            # log to wandb
            wandb.log(
                {'n. of dead relus': dead_neurons,
                 'percentage dead relus': round(
                     dead_neurons_share * 100, 2)})
