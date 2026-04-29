<p align="center">
  <img src="/geoml/images/geoML logo horizontal.png" width="100%">
</p>


# Spatial modeling using machine learning concepts

Geoscientific data is becoming increasingly larger and more complex. Multiple variables, structural constraints, etc. are becoming the norm rather than the exception.

Conventional modelling approaches are very labour-intensive, subjective, and hard to reproduce. CAD-drawn orebodies, variograms, pre- and post-processing steps can result in workflows so complicated that even professionals can become lost within their own work. geoML is being designed to simplify geomodelling and empower the user with advanced machine larning tools that respect geology. 

<p align="center">
  <img src="/geoml/images/walker_gif.gif" width="100%">
</p>

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
* `pandas`
* `numpy`
* `scipy`
* `scikit-image`
* `scikit-learn`
* `tensorflow`
* `tensorflow-probability`[Beauvoir_G_Nwaila_Prep_V001.ipynb](..%2F..%2FProjetos%2FModelagem%20geol%F3gica%2F2025%20South%20Africa%20-%20Glen%2FIvana_2026_data_revised%2FBeauvoir_G_Nwaila_Prep_V001.ipynb)
* `pyvista` and `plotly` for 3D visualization

## Examples
The following notebooks demonstrate the capabilities of the package (if one
 of them seems broken, it is probably going through an update).

* [Sunspot cycle prediction](https://colab.research.google.com/drive/1tbc7I8K0NmpCM4mOZZ1kghlXWnLamE5l)
* [3D classification](https://colab.research.google.com/drive/1oC8b-eCgrfLxMcVsxVv6EvQyeKelUUjE)
* [Potential field modeling using only directional data](https://colab.research.google.com/drive/141zuv7VH431fVt0dwHQiKCSJmYd6E9u8)
* [Gold modeling with auxiliary variables](https://colab.research.google.com/drive/16OFpI1a-V-Wfsgkw_jhlh2NXGFwuHZ0C?usp=sharing)
* [Dealing with faults (experimental)](https://colab.research.google.com/drive/1O_EsFy6bGbUIpzYySafUYrscUcbmTUHc?usp=sharing)


## Learning materials
* [geoML short course presentation](https://1drv.ms/p/c/4a4617b38edb43d3/ERKF7Z9DUQNPon_oZGSxclwB_rneluFp9bAhep6lpbtP6g?e=stGHkW)
* [Notebook 01 - covariance functions](https://colab.research.google.com/drive/1_LS-tem6ATRi62inKvZVOPJRJPjlzGxO?usp=sharing)
* [Notebook 02 - Gaussian process](https://colab.research.google.com/drive/1dq19BU-vgsQAlAyx2gZjiGTPM-d2Oh26?usp=sharing)
* [Notebook 03 - 2D modelling with warping](https://colab.research.google.com/drive/136aCKd9df39S4PmShZ0KIbfmQjaRWkP_?usp=sharing)
* [Notebook 04 - multivariate modelling](https://colab.research.google.com/drive/1ZqeQhbrbVHKEVJgwLQT2jY8AmZR6ENhd?usp=drive_link)
* [Notebook 05 - implicit modelling](https://colab.research.google.com/drive/1mKcRF_Kme6ac-nNzj6dON_updZk01LAS?usp=drive_link)
* [Notebook 06 - gradients](https://colab.research.google.com/drive/1SUNlzThWzg5sbj5sK878WOkgVILdRLjV?usp=drive_link)
* [Notebook 07 - deep Gaussian process](https://colab.research.google.com/drive/1kWvsTAfVMs_K4eHwImJDlPnDUt9k3kZd?usp=sharing)
* [Notebook 08 - multivariate non-stationary modelling](https://colab.research.google.com/drive/1HotYLXScQT-J-aqBMrvI_vR2UOyIWn2o?usp=drive_link)
* [Notebook 09 - compositional data](https://colab.research.google.com/drive/1oxFsdAvouiJSmsiDhs4vh-DsWGvaAUMm?usp=drive_link)

## References

* [2020 - Sunspot Cycle Prediction Using Warped Gaussian Process Regression](https://www.sciencedirect.com/science/article/pii/S0273117719308026)
* [2021 - A machine learning model for structural trend fields](https://doi.org/10.1016/j.cageo.2021.104715)
* [2022 - Learning spatial patterns with variational Gaussian processes: Regression](https://doi.org/10.1016/j.cageo.2022.105056)
* [2023 - Variational Gaussian processes for implicit geological modeling](https://linkinghub.elsevier.com/retrieve/pii/S0098300423000274)
* [2024 - Moho depth model of South America from a machine learning approach](https://www.sciencedirect.com/science/article/abs/pii/S0895981124003377)
* [2025 - Uncertainty Propagation in Deep Gaussian Process Networks (open access)](https://link.springer.com/10.1007/s11004-025-10187-4)
* [2026 - Scalable variational Gaussian process framework for implicit geological modelling and compositional grade interpolation (open access)](https://doi.org/10.1016/j.aiig.2026.100218)

## External links
* [Gaussian process book](https://gaussianprocess.org/gpml/)
* [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
* [Interactive Gaussian Process Visualization](https://www.infinitecuriosity.org/vizgp/)

## Licensing

geoML is available under a dual license model:

### Open-Source License (GPL v3)
geoML is free and open-source software licensed under the GNU General Public License v3.0. Under this license, you may use, study, modify, and distribute geoML at no cost, provided that any software incorporating geoML is also distributed under the GPL v3. This applies to academic research, personal projects, and any open-source work.

### Commercial License
If you wish to integrate geoML into a proprietary or closed-source product — such as commercial geostatistical software, consulting workflows, or any application whose source code will not be made publicly available under the GPL — a separate commercial license is required. The commercial license grants you the right to use geoML without the copyleft obligations of the GPL.
Typical use cases for a commercial license include:

* Embedding geoML in commercial mining, oil & gas, or environmental software
* Using geoML in paid consulting deliverables distributed as closed-source tools
* Integrating geoML into a SaaS platform

To inquire about commercial licensing terms and pricing, please fill [this contact form](https://forms.gle/zh73ZUjFEFhxvxLN7).