import numpy as np
from sage import utils


def permutation_test(model, X, Y, loss, n_permutations=1):
    # Setup.
    N, input_size = X.shape
    loss_fn = utils.get_loss(loss, reduction='mean')
    model = utils.model_conversion(model, loss_fn)

    # Performance with all features.
    base_loss = loss_fn(model(X), Y)

    # More setup.
    scores = np.zeros(input_size)

    for ind in range(input_size):
        # Setup.
        score = 0
        n = 0
        arange = np.arange(N)

        for i in range(n_permutations):
            # Permute data.
            np.random.shuffle(arange)
            X_permute = X[arange]
            X_mod = np.copy(X)
            X_mod[:, ind] = X_permute[:, ind]

            # Calculate loss.
            new_score = loss_fn(model(X_mod), Y)

            # Update score.
            n += 1
            score += (new_score - score) / n

        # Clean up.
        scores[ind] = score - base_loss

    return scores


def mean_importance(model, X, Y, loss):
    # Setup.
    N, input_size = X.shape
    loss_fn = utils.get_loss(loss, reduction='mean')
    model = utils.model_conversion(model, loss_fn)

    # Performance with all features.
    base_loss = loss_fn(model(X), Y)

    # More setup.
    scores = np.zeros(input_size)

    for ind in range(input_size):
        # Setup.
        X_copy = np.copy(X)
        X_copy[:, ind] = np.mean(X[:, ind])
        scores[ind] = loss_fn(model(X_copy), Y) - base_loss

    return scores
