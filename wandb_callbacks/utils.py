import cv2
import numpy as np

import tensorflow as tf


def get_samples_for_activation(class_names, X_val, Y_val):
    """get_samples_for_activation.
    Returns sample images from the given data.
    Sample images contain one image per class in the dataset.

    :param class_names: class names of the dataset.
    :param X_val: features of the dataset.
    :param Y_val: labels of the dataset.
    """
    sample_images, sample_labels, sample_labels_enc = [], [], []

    indices = [0] * len(class_names)
    for i, lbl in enumerate(Y_val):
        # append the list if the label is not in yet
        lbl_i = np.argmax(lbl)
        if indices[lbl_i] == 0:
            indices[lbl_i] = 1
            sample_images.append(X_val[i])
            sample_labels.append(class_names[lbl_i])
            sample_labels_enc.append(Y_val[i])

        # check if all labels are in list
        lbl_sum = np.sum(indices)
        if lbl_sum == len(class_names):
            break

    return np.asarray(sample_images), \
           np.asarray(sample_labels), \
           np.asarray(sample_labels_enc)


class GradCAM:
    """
    Reference:
        https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    """

    def __init__(self, model, layerName):
        self.model = model
        self.layerName = layerName

        self.gradModel = tf.keras.models.Model(inputs=[self.model.inputs],
                                               outputs=[self.model.get_layer(self.layerName).output, self.model.output])

    def compute_heatmap(self, image, classIdx, eps=1e-8):
        with tf.GradientTape() as tape:
            tape.watch(self.gradModel.get_layer(self.layerName).variables)
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = self.gradModel(inputs)

            if len(predictions) == 1:
                # Binary Classification
                loss = predictions[0]
            else:
                loss = predictions[:, classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_HOT):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)
