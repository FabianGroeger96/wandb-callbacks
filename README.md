<p align="center"><img width=40% src="http://fabiangroeger.com/wp-content/uploads/2021/05/wandb-callbacks-logo.png"></p>

<h2 align="center">
 
[![PyPI version](https://badge.fury.io/py/wandb-callbacks.svg)](https://badge.fury.io/py/wandb-callbacks)
[![GitHub Issues](https://img.shields.io/github/issues/FabianGroeger96/wandb-callbacks)](https://img.shields.io/github/issues/FabianGroeger96/wandb-callbacks)
[![License](https://img.shields.io/github/license/FabianGroeger96/wandb-callbacks)](https://img.shields.io/github/license/FabianGroeger96/wandb-callbacks)
![Contribotion](https://img.shields.io/badge/Contribution-Welcome-brightgreen)
<br>
<a href="https://www.buymeacoffee.com/fabiangroeger" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

</h2>

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

## Installation

### Last Stable Release
```python
pip install wandb-callbacks
```

### Latest Development Changes
```bash
git clone https://github.com/FabianGroeger96/wandb-callbacks
```

## Sample Implementation
Can be found in `notebooks/sample_implementation.ipynb`.

## Contributing
Open to ideas and for helpers to develop the package further.
