# sage-experiments

This repository contains code for the experiments in [this paper](https://arxiv.org/abs/2004.00668). The code is frozen to use a specific implementation of SAGE, so if you want to use the current version please visit [this repository](https://github.com/iancovert/sage).

If you find any problems here, please send me an email.

# Replicating experiments

1) Running this code requires a number of Python packages, in addition to the package for SAGE. You can install them all into your virtual environment with the following command:

```bash
pip install .
```

Code for SAGE can be imported using the name `sage`.

2) In the `experiments/` directory, run the `train models.ipynb` notebook to train and save models for all five datasets. 

3) To generate global explanations using SAGE and the baseline methods, run the following notebooks in the `experiments/` directory:

- `sage explanations.ipynb`
- `feature ablation.ipynb`
- `permutation tests.ipynb`
- `mean importance.ipynb`
- `univariate predictors.ipynb`

4) Run sampling algorithms for SHAP and SAGE while saving intermediate results by running these notebooks (also in the `experiments/` directory):

- `sage convergence.ipynb`
- `shap convergence.ipynb`

5) Train models with random subsets of features by running the `random subsets.ipynb` in the `experiments/` directory.

6) Generate figures like the ones in the text by running the following notebooks (also in the `experiments/` directory):

- `examples.ipynb`
- `more examples.ipynb`
- `model monitoring.ipynb`
- `cumulative correlation.ipynb`
- `feature selection.ipynb`
- `convergence.ipynb`

## References

Ian Covert, Scott Lundberg, Su-In Lee. "Understanding Global Feature Contributions With Additive Importance Measures." NeurIPS 2020.
