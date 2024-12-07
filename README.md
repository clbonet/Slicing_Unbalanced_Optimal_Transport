# Slicing Unbalanced Optimal Transport

This repository contains the code to reproduce the experiments of the paper [Slicing Unbalanced Optimal Transport](https://openreview.net/forum?id=AjJTg5M0r8). We propose in this paper two ways of slicing the unbalanced optimal transport (UOT) problem. The first ones averages the UOT problem between distributions projected over lines (SUOT). The second ones performs a global reweighting of the input measures (USOT).

## Abstract

Optimal transport (OT) is a powerful framework to compare probability measures, a fundamental task in many statistical and machine learning problems. Substantial advances have been made in designing OT variants which are either computationally and statistically more efficient or robust. Among them, sliced OT distances have been extensively used to mitigate optimal transport's cubic algorithmic complexity and curse of dimensionality. In parallel, unbalanced OT was designed to allow comparisons of more general positive measures, while being more robust to outliers. In this paper, we bridge the gap between those two concepts and develop a general framework for efficiently comparing positive measures. We notably formulate two different versions of sliced unbalanced OT, and study the associated topology and statistical properties. We then develop a GPU-friendly Frank-Wolfe like algorithm to compute the corresponding loss functions, and show that the resulting methodology is modular as it encompasses and extends prior related work.  We finally conduct an empirical analysis of our loss functions and methodology on both synthetic and real datasets, to illustrate their computational efficiency, relevance and applicability to real-world scenarios including geophysical data.



## Citation

```
@article{bonet2024slicing,
    title={Slicing Unbalanced Optimal Transport},
    author={Clément Bonet and Kimia Nadjahi and Thibault Séjourné and Kilian Fatras and Nicolas Courty},
    year={2024},
    journal={Transactions on Machine Learning Research}
}
```


## Install the package

```
$ python setup.py install
```

Additional packages required for some of the experiments can be installed using
```
$ pip install -r requirements.txt
```

## Description of the library

This library contains mainly two functions: `unbalanced_sliced_ot` and `sliced_unbalanced_ot`, which allow to compute USOT and SUOT respectively.

Both follow the same API. Here is an example on how to compute them:
```
import torch
from sliceduot.sliced_uot import unbalanced_sliced_ot, sliced_unbalanced_ot

a = torch.ones(100)/100
b = torch.ones(100)/100

Xs = torch.randn((100, 2))
Xt = torch.randn((100, 2))

usot, _, _, a_USOT, b_USOT, _ = unbalanced_sliced_ot(a, b, Xs, Xt, p=2, num_projections=500, rho1=1, rho2=1, niter=10)
suot, _, _, a_SUOT, b_SUOT, _ = sliced_unbalanced_ot(a, b, Xs, Xt, p=2, num_projections=500, rho1=1, rho2=1, niter=10)
```



## Experiments

In the folder `Experiments`, you can find the code to reproduce all the experiments of the paper.

- In `Experiments/xp_documents`, you can find the code to reproduce the document classification results of Section 5.1.
- In `Experiments/xp_color_transfer`, you can find the code to reproduce Figure 3.
- In `Experiments/xp_barycenter`, you can find the code to reproduce Figure 4 and 8.
- In `Experiments/ablation_parameters`, you can find the convergence of the Frank-Wolfe algorithm to compute SUOT and USOT (Figure 5), and their sample complexity (Figure 10).
- In `Experiments`, you can also find the code to compute Figure 1 and USOT on hyperbolic spaces (Appendix C.3).

