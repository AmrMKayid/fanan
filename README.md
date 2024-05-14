# fanan 🎨 💗


## Setup
```shell
# pipx & poetry
brew install pipx
pipx ensurepath
pipx install poetry=1.7

# conda env
conda create -n fanan python=3.11.9 -y --channel conda-forge
conda activate fanan
poetry install

# pre-commit
pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
```


## Running

```shell
poetry run fanan --config-path configs/default.yaml
```
