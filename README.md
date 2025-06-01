# geoML
Spatial modeling using machine learning concepts.

This package contains machine learning models specialized in spatial interpolation and smoothing with calibrated confidence intervals. Current functionality includes:

* Gaussian process modeling in 1D, 2D, and 3D, with 
anisotropy ellipsoid;
* Variational Gaussian process for classification and multivariate
modeling;
* Support for compositional data;
* Support for directional data (structural geology
measurements, scalar field gradients, etc.);
* Support for implicit modelling with boundary data (points
lying in the boundary between two rock types);
* Deep learning for non-stationary modeling;
* Exports results to [PyVista](https://github.com/pyvista/pyvista) format;
* Back-end powered by [TensorFlow](https://www.tensorflow.org/).

## Installation

```
pip install git+https://github.com/italo-goncalves/geoML
```

Dependencies:
* `scikit-image`
* `pandas`
* `numpy`
* `tensorflow`
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


## Learning materials
* [geoML short course](https://drive.google.com/drive/folders/1LGtZx_v7kZ0MmjrszVkHu3qzeGa4fT-b?usp=sharing)

## References

* [2020 - Sunspot Cycle Prediction Using Warped Gaussian Process Regression}](https://www.sciencedirect.com/science/article/pii/S0273117719308026)
* [2021 - A machine learning model for structural trend fields](https://doi.org/10.1016/j.cageo.2021.104715)
* [2022 - Learning spatial patterns with variational Gaussian processes: Regression](https://doi.org/10.1016/j.cageo.2022.105056)
* [2023 - Variational Gaussian processes for implicit geological modeling](https://linkinghub.elsevier.com/retrieve/pii/S0098300423000274)
* [2024 - Moho depth model of South America from a machine learning approach](https://www.sciencedirect.com/science/article/abs/pii/S0895981124003377)
* [2025 - Uncertainty Propagation in Deep Gaussian Process Networks (open access)](https://link.springer.com/10.1007/s11004-025-10187-4)
