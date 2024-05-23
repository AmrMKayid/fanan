# fanan 🎨 💗


## Setup
```shell
# pipx & poetry
brew install pipx
pipx ensurepath
pipx install poetry=1.7

# conda env
conda create -n fanan python=3.10.12 -y --channel conda-forge
conda activate fanan
poetry install

# pre-commit
pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push

## dev
poetry install -vvv --with dev

## tpus
poetry install -vvv --with tpu
```


## Running

```shell
poetry run fanan --config-path configs/default.yaml
```

## Platform

```python
import jax
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```
