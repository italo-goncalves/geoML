# geoML
Spatial modeling using machine learning concepts.

This is a work in progress. Current functionality includes:

* Gaussian process modeling in 1D, 2D, and 3D, with 
anisotropy ellipsoid;
* Variational Gaussian process for classification and multivariate
modeling;
* Support for compositional data;
* Support for directional data (structural geology
measurements, scalar field gradients, etc.);
* Support for classification with boundary data (points
lying in the boundary between two rock types);
* Deep learning for non-stationary modeling;
* Exports results to [PyVista](https://github.com/pyvista/pyvista) format;
* Back-end powered by [TensorFlow](https://www.tensorflow.org/).

## Installation
Clone the repo and update the path to include the package's folder.

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
* [Understanding inducing points](https://colab.research.google.com/drive/1P84msQDE3j64MXMcZ5c8q0u6b4myPaT7?usp=sharing)
* [VGP training step by step](https://colab.research.google.com/drive/1rF7bWdrTK54qLiXWcv46J9aMloDTe6r_?usp=sharing)

## References
```
@article{Goncalves2020,
   abstract = {Solar cycle prediction is a key activity in space weather research. Several techniques have been employed in recent decades in order to try to forecast the next sunspot-cycle maxima and time. In this work, the Gaussian Process, a machine-learning technique, is used to make a prediction for the solar cycle 25 based on the annual sunspot number 2.0 data from 1700 to 2018. A variation known as Warped Gaussian Process is employed in order to deal with the non-negativity constraint and asymmetrical data distribution. Tests using holdout data yielded a root mean square error of 10.0 within 5 years and 25.0-35.0 within 10 years. Simulations using the predictive distribution were performed to account for the uncertainty in the prediction. Cycle 25 is expected to last from 2019 – 2029, with a peak sunspot number about 117 (110 by the median) occurring most likely in 2024. Thus our method predicts that solar Cycle 25 will be weaker than previous ones, implying a continuing trend of declining solar activity as observed in the past two cycles.},
   author = {Ítalo G. Gonçalves and Ezequiel Echer and Everton Frigo},
   doi = {10.1016/j.asr.2019.11.011},
   journal = {Advances in Space Research},
   keywords = {gaussian process,machine learning,solar cycle,sunspot number},
   pages = {677-683},
   title = {Sunspot Cycle Prediction Using Warped Gaussian Process Regression},
   volume = {65},
   url = {https://www.sciencedirect.com/science/article/pii/S0273117719308026},
   year = {2020},
}
```

```
@article{Goncalves2021,
   author = {Ítalo Gomes Gonçalves and Felipe Guadagnin and Sissa Kumaira and Saulo Lopes da Silva},
   doi = {10.1016/j.cageo.2021.104715},
   issn = {0098-3004},
   issue = {January},
   journal = {Computers and Geosciences},
   keywords = {Gaussian Process,Implicit modeling,Kriging,Machine learning,Structural trend,Vector field},
   pages = {104715},
   publisher = {Elsevier Ltd},
   title = {A machine learning model for structural trend fields},
   volume = {149},
   url = {https://doi.org/10.1016/j.cageo.2021.104715},
   year = {2021},
}
```

```
@article{Goncalves2022,
   author = {Ítalo Gomes Gonçalves and Felipe Guadagnin and Diogo Peixoto Cordova},
   doi = {10.1016/j.cageo.2022.105056},
   issn = {00983004},
   journal = {Computers & Geosciences},
   keywords = {Gaussian process,Kriging,Machine learning,Variational inference},
   pages = {105056},
   publisher = {Elsevier Ltd},
   title = {Learning spatial patterns with variational Gaussian processes: Regression},
   volume = {161},
   url = {https://doi.org/10.1016/j.cageo.2022.105056},
   year = {2022},
}
```

```
@article{,
   author = {Ítalo Gomes Gonçalves and Felipe Guadagnin and Diogo Peixoto Cordova},
   doi = {10.1016/j.cageo.2023.105323},
   issn = {00983004},
   journal = {Computers & Geosciences},
   month = {5},
   pages = {105323},
   title = {Variational Gaussian processes for implicit geological modeling},
   volume = {174},
   url = {https://linkinghub.elsevier.com/retrieve/pii/S0098300423000274},
   year = {2023},
}
```