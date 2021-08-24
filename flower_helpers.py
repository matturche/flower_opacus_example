import flwr as fl
from collections import OrderedDict
from flwr.server.strategy import FedAvg
from flwr.common import Weights, Parameters, Scalar, FitRes
from flwr.server.server import shutdown
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_splits(
    number: int, split: int = 11, ratio: float = 0.75, seed: int = 42
):
    """Function used to create splits for federated datasets.

    Parameters
    ----------
    number
        The number to split.
    split
        Number of splits to do.
    ratio
        Determine the minimum and maximum size of one split.
    seed
        Random seed.
        
    Returns
    -------
    splits
        The list of elements per split.
    added_splits
        Number of elements per splits matched to index.
    """

    # Set the seed to always get the same splits for evaluation purposes
    np.random.seed(seed)
    # Contains number of elements per split
    splits = []
    # Contains cumulated sum of splits to match indexes
    added_splits = []
    entire_part = number // split
    # A single split cannot be lower than entire_part - min_split
    min_split = entire_part * ratio
    if number < split:
        return [number]
    for i in range(split):
        if number % split != 0 and i >= split - (number % split):
            splits.append(entire_part + 1)
        else:
            splits.append(entire_part)
    length = len(splits) if len(splits) % 2 == 0 else len(splits) - 1
    for s in range(0, length, 2):
        random_value = np.random.randint(low=0, high=min_split)
        splits[s] -= random_value
        added_splits.append(int(np.sum(splits[:s])))
        splits[s + 1] += random_value
        added_splits.append(int(np.sum(splits[: s + 1])))
    if len(splits) % 2 != 0:
        added_splits.append(np.sum(splits[:-1]))
    added_splits.append(np.sum(splits))
    return splits, added_splits

def load_data(
    client_share: int, nbc: int, batch_size: int, train: bool =True
    ):
    """Load CIFAR-10 (training and test set).

    Parameters
    ----------
    client_share : int
        The client id used for splitting dataset.
    nbc : int
        Total number of clients.
    batch_size : int
        Batch size.
    train : bool, optional
        Training or testing dataset, by default True

    Returns
    -------
    Dataloader, int
        Resulting Dataloader and length of the dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = CIFAR10("./dataset", train=train, download=True, transform=transform)
    # Get splits
    _, added_splits = create_splits(len(dataset), split=nbc, ratio=0.25)
    # Create subset
    client_dataset = Subset(dataset,
                            range(int(added_splits[client_share]), int(added_splits[client_share + 1])))
    # Create uniform sampler
    sampler = UniformWithReplacementSampler(
        num_samples=len(client_dataset),
        sample_rate=batch_size/len(client_dataset)
        ) if train else None
    dataloader = DataLoader(client_dataset, batch_sampler=sampler) if train else DataLoader(client_dataset, batch_size=batch_size)
    return dataloader, len(client_dataset)

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.gn1 = nn.GroupNorm(int(6 / 3), 6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(int(16 / 4), 16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.gn1(x)
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.gn2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to get the weights of a model
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Function to set the weights of a model
def set_weights(model, weights) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_target_delta(data_size: int) -> float:
    """Generate target delta given the size of a dataset. Delta should be
    less than the inverse of the datasize.

    Parameters
    ----------
    data_size : int
        The size of the dataset.

    Returns
    -------
    float
        The target delta value.
    """
    den = 1
    while data_size // den >= 1:
        den *= 10
    return 1 / den

def train(
    parameters,
    return_dict,
    config,
    client_share,
    nbc,
    vbatch_size,
    batch_size,
    lr,
    nm,
    mgn,
    state_dict,
    ):
    """Train the network on the training set."""
    train_loss = 0.0
    train_acc = 0.0
    # Define the number of cumulative steps
    assert(vbatch_size%batch_size==0)
    n_acc_steps = int(vbatch_size / batch_size)
    # Get data
    train_loader, len_dataset = load_data(client_share, nbc, batch_size, train=True)
    # Create model
    net = Net().to(DEVICE)
    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    net.train()
    # Get orders for RDP
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    # Get delta
    delta = get_target_delta(len_dataset)
    # Set the sample rate
    sample_rate = batch_size / len_dataset
    # Define the Privacy Engine
    privacy_engine = PrivacyEngine(
        net,
        sample_rate=sample_rate*n_acc_steps,
        alphas=alphas,
        noise_multiplier=nm,
        max_grad_norm=mgn,
        target_delta=delta,
    )
    # Load the state_dict if not None
    if state_dict is not None:
        privacy_engine.load_state_dict(state_dict)
    privacy_engine.to(DEVICE)
    # Attach PrivacyEngine after moving it to the same device as the model
    privacy_engine.attach(optimizer)
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = net(images)
        loss = criterion(out, labels)
        # Get preds
        _, pred_ids = out.max(1)
        # Compute accuracy
        acc = (pred_ids == labels).sum().item() / batch_size
        loss.backward()
        # Take a real optimizer step after n_virtual_steps
        if ((i + 1) % n_acc_steps == 0) or ((i + 1) == len(train_loader)):
            optimizer.step()  # real step
            optimizer.zero_grad()
        else:
            optimizer.virtual_step()  # take a virtual step
        # Detach loss to compute total loss
        train_loss += (loss.detach().item() - train_loss) / (i + 1)
        train_acc += (acc - train_acc) / (i + 1)
    else:
        print(
            f"Round Results:",
            f"Train Loss: {train_loss}",
            f"Train Accuracy: {train_acc}",
        )
        # print best alpha and epsilon
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print(f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
        # Prepare return values
        return_dict["eps"] = epsilon
        return_dict["alpha"] = best_alpha
        return_dict["parameters"] = get_weights(net)
        return_dict["data_size"] = len(train_loader)
        return_dict["state_dict"] = privacy_engine.state_dict()


def test(parameters, return_dict, client_share, nbc, batch_size):
    """Validate the network on the entire test set."""
    test_loss = 0.0
    test_acc = 0.0
    # Get data
    test_loader, _ = load_data(client_share, nbc, batch_size, train=False)
    # Create model
    net = Net().to(DEVICE)
    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        net.eval()
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = net(images)
            loss = criterion(out, labels)
            _, pred_ids = out.max(1)
            acc = (pred_ids == labels).sum().item() / batch_size
            test_loss += (loss.detach().item() - test_loss) / (i + 1)
            test_acc += (acc - test_acc) / (i + 1)
    print(
        f"Test Loss: {test_loss}",
        f"Test Accuracy: {test_acc}",
    )
    # Prepare return values
    return_dict["loss"] = test_loss
    return_dict["accuracy"] = test_acc
    return_dict["data_size"] = len(test_loader)



class FedAvgDp(FedAvg):
    """This class implements the FedAvg strategy for Differential Privacy context."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        FedAvg.__init__(
            self,
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
            accept_failures,
            initial_parameters,
        )
        # Keep track of the maximum possible privacy budget
        self.max_epsilon = 0.0

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Get the privacy budget"""
        if not results:
            return None
        # Get the privacy budget of each client
        accepted_results = []
        disconnect_clients = []
        epsilons = []
        for c, r in results:
            # Check if client can be accepted or not
            if r.metrics["accept"]:
                accepted_results.append([c, r])
                epsilons.append(r.metrics["epsilon"])
            else:
                disconnect_clients.append(c)
        # Disconnect clients if needed
        if disconnect_clients:
            shutdown(disconnect_clients)
        results = accepted_results
        if epsilons:
            self.max_epsilon = max(self.max_epsilon, max(epsilons))
        print(f"Privacy budget ε at round {rnd}: {self.max_epsilon}")
        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_fit(rnd, results, failures)

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Configure the next round of evaluation. Returns None since evaluation is made server side.
        You could comment this method if you want to keep the same behaviour as FedAvg."""
        if client_manager.num_available() < self.min_fit_clients:
            print(
                f"{client_manager.num_available()} client(s) available(s), waiting for {self.min_available_clients} availables to continue."
            )
        # rnd -1 is a special round for last evaluation when all rounds are over
        return None
