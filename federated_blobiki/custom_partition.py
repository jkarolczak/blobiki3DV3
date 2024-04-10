import logging

import numpy as np

logger = logging.getLogger()


def assign_examples_to_clients(y_train, n_parties, partition):
    """
    Assign examples to clients based on the specified partition type.

    :param y_train: labels in the training set (distributed among clients)
    :param n_parties: number of clients
    :param partition: type of partition ("uniform", "mid_balanced", "mid_random", "small_balanced", "small_random",
        "large_balanced", "large_random")
    :return: dict with client indexes as keys and lists of indexes of assigned examples as values
    """

    n_train = len(y_train)
    idxs = np.random.permutation(n_train)

    if partition == "uniform":
        batch_idxs = np.array_split(idxs, n_parties)
    elif partition == "mid_balanced":
        class_counts = np.bincount(y_train)
        weights = np.zeros_like(y_train, dtype=float)
        weights[y_train == 0] = class_counts[0]
        weights[y_train == 1] = class_counts[1]

        batch_idxs = []
        for _ in range(n_parties):
            weights /= np.sum(weights)
            batch_size = n_train // n_parties
            idx = np.random.choice(np.arange(n_train), size=batch_size, replace=False, p=weights)
            batch_idxs.append(idx)
            weights = np.delete(weights, idx)
            n_train -= batch_size
        batch_idxs[-1] = np.concatenate((batch_idxs[-1], np.arange(n_train)))
    elif partition == "mid_random":
        batch_idxs = []
        for _ in range(n_parties):
            batch_size = n_train // n_parties
            idx = np.random.choice(np.arange(n_train), size=batch_size, replace=False)
            batch_idxs.append(idx)
            n_train -= batch_size
        batch_idxs[-1] = np.concatenate((batch_idxs[-1], np.arange(n_train)))
    elif partition == "large_balanced":
        total_examples = len(y_train)
        client_sizes = (np.random.beta(2, 5, size=n_parties) * total_examples).astype(int)
        client_sizes = np.round(client_sizes / np.sum(client_sizes) * total_examples).astype(int)
        class_counts = np.bincount(y_train)
        weights = np.zeros_like(y_train, dtype=float)
        weights[y_train == 0] = class_counts[0]
        weights[y_train == 1] = class_counts[1]
        batch_idxs = []
        for size in client_sizes:
            weights /= np.sum(weights)
            size = min(size, len(weights))
            idx = np.random.choice(np.arange(n_train), size=size, replace=False, p=weights)
            batch_idxs.append(idx)
            weights = np.delete(weights, idx)
            n_train -= size
        if np.sum(client_sizes) < total_examples:
            remaining_examples = np.setdiff1d(np.arange(n_train), np.concatenate(batch_idxs))
            batch_idxs[-1] = np.concatenate((batch_idxs[-1], remaining_examples))
    elif partition == "small_random":
        total_examples = len(y_train)
        client_sizes = np.random.beta(2, 5, size=n_parties) * total_examples
        client_sizes = np.round(client_sizes / np.sum(client_sizes) * total_examples).astype(int)
        batch_idxs = []
        for size in client_sizes:
            idx = np.random.choice(np.arange(n_train), size=size, replace=False)
            batch_idxs.append(idx)
        if np.sum(client_sizes) < total_examples:
            remaining_examples = np.setdiff1d(np.arange(n_train), np.concatenate(batch_idxs))
            batch_idxs[-1] = np.concatenate((batch_idxs[-1], remaining_examples))
        elif partition == "large_balanced":
            total_examples = len(y_train)
            client_sizes = (np.random.beta(5, 2, size=n_parties) * total_examples).astype(int)
            client_sizes = np.round(client_sizes / np.sum(client_sizes) * total_examples).astype(int)
            class_counts = np.bincount(y_train)
            weights = np.zeros_like(y_train, dtype=float)
            weights[y_train == 0] = class_counts[0]
            weights[y_train == 1] = class_counts[1]
            batch_idxs = []
            for size in client_sizes:
                weights /= np.sum(weights)
                size = min(size, len(weights))
                idx = np.random.choice(np.arange(n_train), size=size, replace=False, p=weights)
                batch_idxs.append(idx)
                weights = np.delete(weights, idx)
                n_train -= size
            if np.sum(client_sizes) < total_examples:
                remaining_examples = np.setdiff1d(np.arange(n_train), np.concatenate(batch_idxs))
                batch_idxs[-1] = np.concatenate((batch_idxs[-1], remaining_examples))
    elif partition == "large_random":
        total_examples = len(y_train)
        client_sizes = (np.random.beta(5, 2, size=n_parties) * total_examples).astype(int)
        client_sizes = np.round(client_sizes / np.sum(client_sizes) * total_examples).astype(int)
        batch_idxs = []
        for size in client_sizes:
            idx = np.random.choice(np.arange(n_train), size=size, replace=False)
            batch_idxs.append(idx)
        if np.sum(client_sizes) < total_examples:
            remaining_examples = np.setdiff1d(np.arange(n_train), np.concatenate(batch_idxs))
            batch_idxs[-1] = np.concatenate((batch_idxs[-1], remaining_examples))
    else:
        logger.warning(f"Unknown partition type: {partition}. Using uniform distribution.")
        batch_idxs = np.array_split(idxs, n_parties)

    net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    return net_dataidx_map
