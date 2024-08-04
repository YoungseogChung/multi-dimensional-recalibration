import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from uncertainty_toolbox.metrics_calibration import mean_absolute_calibration_error

""" Accuracy Metrics. """


def mae_per_dim(mean, data, return_list=False):
    assert mean.shape == data.shape
    out = mean_absolute_error(data, mean, multioutput="raw_values")
    mean_out = np.mean(out)
    if return_list:
        return mean_out, out
    return mean_out


def mse_per_dim(mean, data, return_list=False):
    assert mean.shape == data.shape
    out = mean_squared_error(data, mean, multioutput="raw_values")
    mean_out = np.mean(out)
    if return_list:
        return mean_out, out
    return mean_out


""" Calibration. """


def mace_per_dim(mean, std, data, return_list=False):
    """Measure the mean of miscalibration across all dimensions

    Args:
        mean: numpy array of mean predictions, size (num_pts, dim_y)
        std: numpy array of std predictions, size (num_pts, dim_y)
        data: numpy array of datapoints, size (num_pts, dim_y)

    Returns:
        a scalar, the mean of miscalibration error across dimension
    """

    num_pts, dim_y = mean.shape
    assert std.shape == data.shape == (num_pts, dim_y)

    mace_per_dim = []
    for d in range(dim_y):
        mean_for_d = mean[:, d]
        std_for_d = std[:, d]
        data_for_d = data[:, d]
        mace_for_d = mean_absolute_calibration_error(
            y_pred=mean_for_d,
            y_std=std_for_d,
            y_true=data_for_d,
            vectorized=True,
        )
        mace_per_dim.append(mace_for_d)

    mean_mace = np.mean(mace_per_dim)
    if return_list:
        return mean_mace, mace_per_dim

    return mean_mace
