# nonstationary-nn-streams

Reproduction and extension of **Soft Resets** — an online learning method that fights *loss of plasticity* in neural networks trained on non-stationary data streams, by modelling parameter drift with an Ornstein-Uhlenbeck process whose drift parameter $\gamma_t$ is learned online.

Research project — *Data Stream Processing*, M2 Data Science, Institut Polytechnique de Paris (2026-2027).

## Reference paper

> A. Galashov, M. K. Titsias, A. György, C. Lyle, R. Pascanu, Y. W. Teh, M. Sahani.
> **Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Reset.**
> *NeurIPS 2024.* [arXiv:2411.04034](https://arxiv.org/abs/2411.04034)

```bibtex
@inproceedings{galashov2024softresets,
  title     = {Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Reset},
  author    = {Galashov, Alexandre and Titsias, Michalis K. and Gy{\"o}rgy, Andr{\'a}s and Lyle, Clare and Pascanu, Razvan and Teh, Yee Whye and Sahani, Maneesh},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

The official code was not released by the authors (IP constraints). Everything in this repository is an independent re-implementation in PyTorch.

## Scope

| Benchmark | Non-stationarity | Paper ref. |
|---|---|---|
| Random-label MNIST (data-efficient) | labels re-drawn at each task | Fig. 2, App. D.1 |
| Random-label CIFAR-10 (memorization) | labels re-drawn at each task | Fig. 2, App. D.1 |
| Permuted MNIST | pixel permutation per task | Fig. 2, App. D.1 |
| Permuted-patch MNIST | patch permutation per task | Fig. 5a, App. H.3 |
| Partially random-label MNIST | 20/40/60 % random labels | Fig. 5b-c, App. D.2 |

Methods: Online SGD, Hard Reset, L2 Init, Shrink & Perturb, Soft Reset (+ proximal and Bayesian variants).

## Repository structure

```
configs/        experiment configs (one YAML per benchmark)
src/softresets/ package: data streams, models, methods, metrics
scripts/        entry points (data preparation, training)
notebooks/      exploration and figures
tests/          sanity checks on the task streams
report/         scientific report (LaTeX)
```

## Installation

Environment managed with [uv](https://docs.astral.sh/uv/).

```bash
cd nonstationary-nn-streams
uv sync                       # creates .venv and installs deps + dev group
source .venv/bin/activate     # activate the environment
```

## Status

- [ ] Plasticity task streams (MNIST / CIFAR-10)
- [ ] Baselines
- [ ] Soft Reset
- [ ] Reproduction of Fig. 2
- [ ] Limitations and extensions
- [ ] Report

## Authors

- Younes Oudina — M2 Data Science, Institut Polytechnique de Paris
- Adam Chgour — M2 Data Science, Institut Polytechnique de Paris
