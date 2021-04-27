[![PyPI version](https://badge.fury.io/py/wandb-callbacks.svg)](https://badge.fury.io/py/wandb-callbacks)

# Weights &amp; Biases Callbacks
`wandb-callbacks` provides some additional Callbacks for Weights &amp; Biases.

Callbacks currently implemented:
* `ActivationCallback`
  * visualizes the activations of a layer
  * activations are computed for a sample of each class
* `DeadReluCallback`
  * logs the number of dead relus in each layer
  * prints warning if the percentage is higher than a threshold
* `GradCAMCallback`
  *  [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)
  *  produces a coarse localization map highlighting the important regions in the image for predicting the class of the image

## Sample Implementation
Can be found in `notebooks/sample_implementation.ipynb`.
