## hDQN in JAX

This repository implements the [hierarchical DQN (hDQN)](https://arxiv.org/pdf/1604.06057.pdf) algorithm in JAX. Implementation is built following the [PyTorch implementation](https://github.com/higgsfield/RL-Adventure/blob/master/9.hierarchical%20dqn.ipynb).

## Instructions

Install JAX using the following-

```
pip install "jax[cuda111]<=0.21.1" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

In case of memory errors use the following flags-

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 python ...
XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 python ...
```

Run hDQN on the stochastic MDP using the following-

```
python train.py
```

Result will be saved as `plot.png`.
