import tensorflow as tf
from tensorflow import keras
import wandb
import matplotlib.pyplot as plt
import numpy as np


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
            # Preparea the plot to be logged to wandb
            fig, axs = plt.subplots(nrows=4, ncols=8, figsize=(15, 8))
            c = 0
            for i in range(4):
                for j in range(8):
                    axs[i][j].imshow(features[c], cmap='gray')
                    axs[i][j].set_xticks([])
                    axs[i][j].set_yticks([])
                    c += 1
            wandb.log({"features_labels_{}".format(label): plt})
            plt.close()
