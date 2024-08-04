import numpy as np
from scipy.optimize import minimize_scalar

from metrics.metrics_by_dim import mace_per_dim

""" Criterion function for optimization """


def miscal_by_dim_crit_fn(mean, std, data):
    return mace_per_dim(mean, std, data, return_list=False)


""" Mean bias by dimension, error correlation matrix """


def get_mean_bias_error_correlation(mean, data):
    """
    Return the mean bias by dimension and the error correlation matrix

    Args:
        mean: numpy array of mean predictions, size (num_pts, dim_y)
        data: numpy array of datapoints, size (num_pts, dim_y)

    Returns:
        dict:
            - mean_bias: numpy array, size (1, dim_y)
            - error_corr_mat: numpy array, size (dim_y, dim_y)
    """

    num_pts, dim_y = mean.shape
    assert data.shape == (num_pts, dim_y)
    error = mean - data
    mean_bias = np.mean(error, axis=0)
    error_corr_mat = np.corrcoef(error.T).astype(np.float32)

    out = {
        'mean_bias': mean_bias,
        'error_corr_mat': error_corr_mat
    }

    return out


""" Correlation matrix construction and reverse """


def make_corr_mat(corrs, dim_y):
    """ Given an array of corr values (all off diagonal values),
    make a correlation matrix with the values filled in, 
    the diagonals will be 1's
    """
    corr_mat = np.ones((dim_y, dim_y))
    num_corr_vals = int((dim_y * (dim_y - 1)) / 2)

    assert corrs.shape == (num_corr_vals,)

    vidx = 0
    for d1 in range(dim_y):
        for d2 in range(d1 + 1, dim_y):
            corr_mat[d1, d2] = corrs[vidx]
            corr_mat[d2, d1] = corrs[vidx]
            vidx += 1
    assert all(corr_mat.diagonal() == 1)

    return corr_mat


def extract_corrs_from_corr_mat(corr_mat, dim_y):
    """ Given the correlation matrix, extract just the off diagonals """

    assert corr_mat.shape == (dim_y, dim_y)
    values = []
    for d1 in range(dim_y):
        for d2 in range(d1 + 1, dim_y):
            values.append(corr_mat[d1, d2])
    values = np.array(values)

    return values


def extract_corr_from_cov_mat(cov_mat, dim_y):
    assert cov_mat.shape in [2, 3]
    if len(cov_mat.shape) == 2:  # (dim_y, dim_y)
        assert cov_mat.shape == (dim_y, dim_y)
        cov_mat = cov_mat[np.newaxis, :, :]

    # cov_mat now (num_pts, dim_y, dim_y)
    num_pts = cov_mat.shape[0]
    assert cov_mat.shape[1:] == (dim_y, dim_y)

    variances = np.diagonal(cov_mat, axis1=1, axis2=2)
    assert variances.shape == (num_pts, dim_y)
    stds = np.sqrt(variances)

    corrs = (cov_mat / stds[:, np.newaxis, :]) / stds[:, :,
                                                 np.newaxis]  # (num_pts, dim_y, dim_y)

    return corrs


""" Optimizer functions to find multiplicative constant for initial recalibration """


def optimize_ratio(
        criterion: str,
        mean: np.ndarray,
        data: np.ndarray,
        **kwargs
):
    """
    Optimize the multiplicative ratio applied to std or cov to minimize the criterion.
    Args:
        criterion: string in ['by_dim']
        mean: numpy array of mean predictions, size (num_pts, dim_y)
        data: numpy array of datapoints, size (num_pts, dim_y)
        **kwargs:
            [cov or std]: one of cov or std must be given
                cov: numpy array of cov predictions, size (num_pts, dim_y, dim_y)
                std: numpy array of cov predictions, size (num_pts, dim_y)


    Returns:
        a dictionary with
            - recal_ratio: scalar, how much to multiply given cov or std by
            - criterion: str, same as input
            - achieved criterion: scalar, the optimized criterion value
    """

    assert criterion in ['by_dim']
    assert 'std' in kwargs

    WORST_CAL = 0.5
    crit_fn = miscal_by_dim_crit_fn

    def make_crit_fn_input(ratio):
        if criterion == 'by_dim':
            in_dict = {'mean': mean, 'std': ratio * kwargs['std'], 'data': data}
        else:
            raise NotImplementedError

        return in_dict

    BEST_SOFAR = WORST_CAL

    def obj(ratio):
        nonlocal WORST_CAL
        nonlocal BEST_SOFAR

        # If ratio is 0, return worst-possible calibration metric
        if ratio == 0:
            return WORST_CAL

        in_dict = make_crit_fn_input(ratio)

        curr_crit = crit_fn(**in_dict)
        if curr_crit < BEST_SOFAR:
            BEST_SOFAR = curr_crit

        return curr_crit

    bounds = (1e-8, 3)
    result = minimize_scalar(fun=obj, bounds=bounds)
    opt_ratio = result.x

    ratio_crit_val = crit_fn(**(make_crit_fn_input(opt_ratio)))
    # result.success = False  # for debugging

    if not result.success:
        # raise Warning("Optimization did not succeed")
        original_crit_val = crit_fn(**(make_crit_fn_input(1.0)))
        if ratio_crit_val > original_crit_val:
            raise Warning(
                "No better calibration found, no recalibration performed and returning original uncertainties"
            )
            opt_ratio = 1.0

    out = {
        'ratio': opt_ratio,
        'crit': criterion,
        'val': ratio_crit_val,
    }

    return out
