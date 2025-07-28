from gpytorch import ExactMarginalLogLikelihood
import torch
from torch import Tensor
import numpy as np
from sklearn.model_selection import KFold
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch import fit_fully_bayesian_model_nuts
from botorch.fit import fit_gpytorch_mll
from gpytorch.distributions import MultivariateNormal
from sklearn.metrics import mean_squared_error, r2_score



def cross_validation(model_initializer, X, y, n_splits=5, random_state=None):
    """
    Perform k-fold cross-validation for a given model and feature set.
    Args:
        model_initializer: function (train_x, train_y) -> (mll, model)
        X: numpy array of features
        y: numpy array of targets
        n_splits: number of folds
        random_state: random seed
    Returns:
        dict: {'mse': avg_mse, 'r2': avg_r2, 'mse_list': [...], 'r2_list': [...],
               'y_true_folds': [...], 'y_pred_folds': [...]}
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mses = []
    r2s = []
    y_true_folds = []
    y_pred_folds = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_x = torch.tensor(X_train.astype(np.float64))
        train_y = torch.tensor(y_train)
        test_x = torch.tensor(X_test.astype(np.float64))
        train_x = train_x.cuda().float()
        train_y = train_y.cuda().float().squeeze()
        test_x = test_x.cuda().float()

        mll, model = model_initializer(train_x, train_y)

        if isinstance(mll, ExactMarginalLogLikelihood):
            # Fit the model using ExactMarginalLogLikelihood
            fit_gpytorch_mll(mll)
        elif hasattr(mll, '__call__'):
            # Assume is a train function: call with (model, train_x, train_y)
            mll(model, train_x, train_y)
        with torch.no_grad():
            model.eval()
            pred = model(test_x)
            # Handle different prediction types
            if isinstance(pred, MultivariateNormal):
                y_pred = pred.mean.cpu().numpy().squeeze()
            elif isinstance(pred, Tensor):
                y_pred = pred.mean(dim=0).cpu().numpy().squeeze()
        if y_pred.ndim > 1:
            # If shape is (n_samples, n_tasks), reduce to (n_samples,)
            if y_pred.shape[0] == y_test.shape[0]:
                y_pred = y_pred.mean(axis=1)  # mean over tasks
            elif y_pred.shape[1] == y_test.shape[0]:
                y_pred = y_pred.mean(axis=0)
            else:
                raise ValueError(f"y_pred shape {y_pred.shape} does not match y_test shape {y_test.shape}")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Fold MSE: {mse:.4f}, R2: {r2:.4f}")
        mses.append(mse)
        r2s.append(r2)
        y_true_folds.append(y_test)
        y_pred_folds.append(y_pred)
    return {'mse': float(np.mean(mses)), 'r2': float(np.mean(r2s)), 'mse_list': mses, 'r2_list': r2s,
            'y_true_folds': y_true_folds, 'y_pred_folds': y_pred_folds}
