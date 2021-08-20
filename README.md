# geoML
Spatial modeling using machine learning concepts.

This is a work in progress. Current functionality includes:

* Gaussian process modeling in 1D, 2D, and 3D, with 
anisotropy ellipsoid;
* Variational Gaussian process for classification and multivariate
modeling;
* Support for directional data (structural geology
measurements, scalar field gradients, etc.);
* Support for classification with boundary data (points
lying in the boundary between two rock types);
* Non-stationary modeling with deep learning;
* Exports results to [PyVista](https://github.com/pyvista/pyvista) format;
* Back-end powered by TensorFlow.

## Installation
Clone the repo and update the path to include the package's folder.

Dependencies:
* `scikit-image`
* `pandas`
* `numpy`
* `tensorflow` (preferably `tensorflow-gpu`)
* `tensorflow-probability`
* `pyvista` and `plotly` for 3D visualization

## Examples
The following notebooks demonstrate the capabilities of the package (if one
 of them seems broken, it is probably going through an update).

* [Walker Lake](https://colab.research.google.com/drive/1zH-dAytMwR_OocDgJWE3Sy8pbcq0PdAJ)
* [2D classification with structural constraints](https://colab.research.google.com/drive/1eiIa8kavRIp5SK5R89ozkIj5lmeRrx9x)
* [Sunspot cycle prediction](https://colab.research.google.com/drive/1tbc7I8K0NmpCM4mOZZ1kghlXWnLamE5l)
* [3D classification](https://colab.research.google.com/drive/1oC8b-eCgrfLxMcVsxVv6EvQyeKelUUjE)
* [Potential field modeling using only directional data](https://colab.research.google.com/drive/141zuv7VH431fVt0dwHQiKCSJmYd6E9u8)
* [Jura](https://colab.research.google.com/drive/1v7Us_ljM5zwkLy6IIKfOjREazZSLepjU?usp=sharing)
* [Compositional data](https://colab.research.google.com/drive/14bvDkre3UNxXywUWq2QEs6Q4w-gd30Mb?usp=sharing)
* [Gold modeling with auxiliary variables](https://colab.research.google.com/drive/16OFpI1a-V-Wfsgkw_jhlh2NXGFwuHZ0C?usp=sharing)
* [VGP training step by step](https://colab.research.google.com/drive/1rF7bWdrTK54qLiXWcv46J9aMloDTe6r_?usp=sharing)


