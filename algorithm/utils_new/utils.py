
import numpy as np
from datetime import datetime
import os

def get_info_from_dqn_weights(weights, num_clients, dqn_list_epochs):
    client_dicts = {}
    for cli in range(num_clients):
        cli_dict = {}
        cli_dict["mean"] = weights[0, 2*cli]
        cli_dict["std"] = weights[0, 2*cli+1]
        cli_dict["epoch"] = dqn_list_epochs[cli]
        client_dicts[cli] = cli_dict
    return client_dicts



