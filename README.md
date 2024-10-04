# Flow Matching with Jax
_notebook by Georges Le Bellier_  - [Twitter](https://twitter.com/_lebellig), [Website](https://gle-bellier.github.io)

![flow(1)](https://github.com/gle-bellier/jax_fm/blob/main/swiss_1000.gif)

This notebook proposes a minimal implementation of _Flow Matching_ using Jax, Flax and JaxTyping. I mostly wanted to define the conditional velocity as the time derivative of the interpolant (i.e. conditional path):
```python
def interpolant(x0: Float[Array, "N"] , x1: Float[Array, "N"], t: float) -> Float[Array, "N"]:
    return x0 + (x1 - x0) * t

velocity = jax.jacrev(interpolant, argnums=2)
```

## References:

📄 [1] **Flow Matching for Generative Modeling** by Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le - [Article](https://arxiv.org/abs/2210.02747)

🐍 [2] **Jax** - [Code](https://github.com/jax-ml/jax)

🐍 [3] **Flax** - [Code](https://github.com/google/flax)

🐍 [4] **JaxTyping** by Patrick Kidger - [Doc](https://github.com/patrick-kidger/jaxtyping)

🐍 [5] **Introduction to Flow Matching** by Georges Le Bellier - [Notebook](https://github.com/gle-bellier/flow-matching/tree/main)
