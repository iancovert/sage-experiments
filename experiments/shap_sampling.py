import numpy as np
from sage import utils, core
from tqdm.auto import tqdm


class SHAPEstimator:
    '''
    Estimate SHAP values by unrolling permutations of feature indices.

    Args:
      model: callable prediction model.
      imputer: for imputing held out values.
      loss: loss function ('mse', 'cross entropy').
    '''
    def __init__(self,
                 imputer,
                 loss='cross entropy'):
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction='none')

    def __call__(self,
                 X,
                 Y,
                 batch_size=512,
                 detect_convergence=True,
                 thresh=0.025,
                 n_permutations=None,
                 verbose=False,
                 bar=True):
        # Verify model.
        assert (len(X.shape) == 2) and (X.shape[0] == 1)
        x = X.repeat(batch_size, 0)
        y = Y.repeat(batch_size, 0)
        num_features = self.imputer.num_groups
        X, Y = utils.verify_model_data(
            self.imputer, x, y, self.loss_fn, batch_size)

        # For setting up bar.
        estimate_convergence = n_permutations is None
        if estimate_convergence and verbose:
            print('Estimating convergence time')

        # Possibly force convergence detection.
        if n_permutations is None:
            n_permutations = 1e20
            if not detect_convergence:
                detect_convergence = True
                if verbose:
                    print('Turning convergence detection on')

        if detect_convergence:
            assert 0 < thresh < 1

        # Print message explaining parameter choices.
        if verbose:
            print('Batch size = batch * samples = {}'.format(
                batch_size * self.imputer.samples))

        # Set up bar.
        n_loops = int(n_permutations / batch_size)
        if bar:
            if estimate_convergence:
                bar = tqdm(total=1)
            else:
                bar = tqdm(total=n_loops * batch_size * num_features)

        # Setup.
        arange = np.arange(batch_size)
        scores = np.zeros((batch_size, num_features))

        # Permutation sampling.
        tracker = utils.ImportanceTracker()
        for it in range(n_loops):
            # Sample permutations.
            S = np.zeros((batch_size, num_features), dtype=bool)
            permutations = np.tile(np.arange(num_features), (batch_size, 1))
            for i in range(batch_size):
                np.random.shuffle(permutations[i])

            # Make prediction with missing features.
            y_hat = self.imputer(x, S)
            prev_loss = self.loss_fn(y_hat, y)

            for i in range(num_features):
                # Add next feature.
                inds = permutations[:, i]
                S[arange, inds] = 1

                # Make prediction with missing features.
                y_hat = self.imputer(x, S)
                loss = self.loss_fn(y_hat, y)

                # Calculate delta sample.
                scores[arange, inds] = prev_loss - loss
                prev_loss = loss
                if bar and (not estimate_convergence):
                    bar.update(batch_size)

            # Update tracker.
            tracker.update(scores)

            # Calculate progress.
            std = np.max(tracker.std)
            gap = tracker.values.max() - tracker.values.min()
            ratio = std / gap

            # Print progress message.
            if verbose:
                if detect_convergence:
                    print('StdDev Ratio = {:.4f} (Converge at {:.4f})'.format(
                        ratio, thresh))
                else:
                    print('StdDev Ratio = {:.4f}'.format(ratio))

            # Check for convergence.
            if detect_convergence:
                if ratio < thresh:
                    if verbose:
                        print('Detected convergence')

                    # Skip bar ahead.
                    if bar:
                        bar.n = bar.total
                        bar.refresh()
                    break

            # Update convergence estimation.
            if bar and estimate_convergence:
                std_est = ratio * np.sqrt(it + 1)
                n_est = (std_est / thresh) ** 2
                bar.n = np.around((it + 1) / n_est, 4)
                bar.refresh()

        if bar:
            bar.close()

        expl = core.Explanation(tracker.values, tracker.std, 'LossSHAP')
        expl.n_permutations = (it + 1) * batch_size
        return expl
