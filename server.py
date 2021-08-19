import flwr as fl
import multiprocessing as mp
import argparse
from flower_helpers import Net, FedAvgDp, get_weights, test


"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
# import os
# if os.environ.get("https_proxy"):
#     del os.environ["https_proxy"]
# if os.environ.get("http_proxy"):
#     del os.environ["http_proxy"]

batch_size = None

def get_eval_fn():
    """Get the evaluation function for server side.
    
    Returns
    -------
    evaluate
        The evaluation function
    """

    def evaluate(weights):
        """Evaluation function for server side.

        Parameters
        ----------
        weights
            Updated model weights to evaluate.

        Returns
        -------
        loss
            Loss on the test set.
        accuracy
            Accuracy on the test set.
        """
        global batch_size
        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        # 0 is for the share and 1 total number of clients, here for server test
        # we take the full test set
        p = mp.Process(target=test, args=(weights, return_dict, 0, 1, batch_size))
        # Start the process
        p.start()
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Couldn't close the evaluating process: {e}")
        # Get the return values
        loss = return_dict["loss"]
        accuracy = return_dict["accuracy"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate

def main():
    # Start Flower server for three rounds of federated learning
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "-nbc",
        type=int,
        default=2,
        help="Number of clients to keep track of dataset share",
    )
    parser.add_argument("-b", type=int, default=256, help="Batch size")
    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    args = parser.parse_args()
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    global batch_size
    batch_size = int(args.b)
    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")
    # Create a new fresh model to initialize parameters
    net = Net()
    init_weights = get_weights(net)
    # Convert the weights (np.ndarray) to parameters
    init_param = fl.common.weights_to_parameters(init_weights)
    # del the net as we don't need it anymore
    del net
    # Define the strategy
    strategy = FedAvgDp(
        fraction_fit=float(fc / ac),
        min_fit_clients=fc,
        min_available_clients=ac,
        eval_fn=get_eval_fn(),
        initial_parameters=init_param,
    )
    fl.server.start_server(
        "[::]:8080", config={"num_rounds": rounds}, strategy=strategy
    )
    
if __name__ == "__main__":
    main()
