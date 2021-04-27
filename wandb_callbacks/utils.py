import numpy as np


def get_samples_for_activation(class_names, X_val, Y_val):
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
