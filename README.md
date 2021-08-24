# Flower and Opacus example

This is an introductory example to Flower and Opacus with scaling on GPU using PyTorch, it is an adaptation of the "[Flower Example using PyTorch](https://flower.dev/docs/quickstart_pytorch.html)" example and provides some insights on how to perform DP-SGD in federated settings and scale Flower to hundreds of clients using GPU. For more information regarding the GPU issue for now with Flower you can check out my [other repository](https://github.com/matturche/flower_scaling_example). DP-SGD is pretty easy to implement on vanilla models but when for federated models we have to keep track of the privacy budget among every clients. This repository proposes one way to compute the privacy budget accross rounds. 

## Project Setup

Start by cloning the example project. 

This will create a new directory called `flower_opacus_example` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- flower_helpers.py
-- run.sh
-- README.md
```

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. It is recommended to use [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning with PyTorch and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply run them in a terminal as follows:

```shell
./run.sh
```

This will create a Flower server and two clients, training for 3 rounds. You can specify addtionnal parameters as follows:

```shell
./run.sh NBCLIENTS NBMINCLIENTS NBFITCLIENTS NBROUNDS VBATCHSIZE BATCHSIZE LR NM MGN EPS
```

`NBCLIENTS` specifies the number of clients you want to launch at once, `NBMINCLIENTS` the minimum number of clients needed to launch a round, `NBFITCLIENTS` the number of clients sampled in a round and `NBROUNDS` the number of rounds you want to train for. `VBATCHSIZE` and `BATCHSIZE` are the batch sizes for Opacus with `VBATCHSIZE` % `BATCHSIZE` = 0, it allows containing memory usage during DP-SGD. `LR` is the learning rate. Finally `NM` (noise multiplier), `MGN` (max gradient norm) and `EPS` (epsilon) are hyperparameters for the Privacy Engine of Opacus. 


You will see that PyTorch is starting a federated training. If you have issues with clients not connecting, you can try uncommenting these lines in both `server.py` and `client.py`:

```python
import os
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]
```
