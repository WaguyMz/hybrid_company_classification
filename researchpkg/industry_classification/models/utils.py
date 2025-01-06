import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


class NN_Utils:
    """
    This class contains utility functions for the neural network models.
    """

    @staticmethod
    def compute_cnn_activation_shape(
        dim, kernel_size, dilation=(1, 1), stride=(1, 1), padding="same"
    ):
        """
        Computes the output shape of a convolutional layer.
        :param dim: tuple of input shape
        :param kernel_size: tuple of kernel size
        :param dilation: tuple of dilation
        :param stride: tuple of stride
        :param padding: padding type
        :return: tuple of output shape
        """
        if padding == "same":

            def shape_each_dim(i):
                odim_i = int(dim[i] / stride[i])
                return odim_i

        else:

            def shape_each_dim(i):
                odim_i = (
                    dim[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1
                )
                return max(1, int((odim_i // stride[i])) + 1)

        return shape_each_dim(0), shape_each_dim(1)

    @staticmethod
    def compute_cnn_pooling_shape(dim, kernel_size, padding=(0, 0)):
        """
        Computes the output shape of a pooling layer.
        :param dim: tuple of input shape
        :param kernel_size: tuple of kernel size
        :param padding: tuple of padding
        :return: tuple of output shape
        """
        stride = kernel_size

        def shape_each_dim(i):
            odim_i = dim[i] + 2 * padding[i] - kernel_size[i]
            return max(1, int(odim_i // stride[i]) + 1)

        return shape_each_dim(0), shape_each_dim(1)

    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, class_names, normalize=False):
        """
        Compute confusion matrix and return matplotlib figure.

        :param y_true: true labels
        :param y_pred: predicted labels
        :param class_names: list of class names
        :param normalize: if True, normalize the confusion matrix
        :return: confusion matrix
        """

        cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true).tolist()))

        if normalize:
            # Normalize the confusion matrix
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm = cm.astype("float") / cm_sum

        plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        if normalize:
            fmt = ".2f"
        else:
            fmt = "d"

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def get_num_params(model):
        """
        Get the number of parameters in the model.
        :param model: pytorch model

        :return: number of parameters
        """
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def get_num_trainable_params(model):
        """
        Get the number of trainable parameters in the model.
        :param model: pytorch model

        :return: number of parameters that require gradient
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_model_size(model):
        """
        Get the size of the model in MB. round to 2 decimal places.

        :param model: pytorch model

        :return: size of the model in MB
        """
        size = NN_Utils.get_num_params(model) * 4 / 1024 / 1024
        return round(size, 2)
