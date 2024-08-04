import numpy as np
import torch

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

""" General array/tensor utils"""


def convert_to_tensor(in_item, device):
    if isinstance(in_item, np.ndarray):
        in_item = torch.from_numpy(in_item).float().to(device)
    elif isinstance(in_item, list):
        in_item = torch.tensor(in_item).float().to(device)
    elif isinstance(in_item, torch.Tensor):
        pass
    else:
        raise RuntimeError(
            f'No handling method to convert {type(in_item)} to tensor'
        )

    return in_item


def convert_to_numpy(in_item):
    if isinstance(in_item, np.ndarray):
        pass
    elif isinstance(in_item, list):
        in_item = np.array(in_item).astype(float)
    elif isinstance(in_item, torch.Tensor):
        in_item = in_item.detach().cpu().numpy()
    else:
        raise RuntimeError(
            f'No handling method to convert {type(in_item)} to tensor'
        )

    return in_item


def check_shape(in_item, in_shape, raise_error=False):
    """
    Check that the shape of in_item is in_shape

    Args:
        in_item: numpy array or torch tensor, must have .shape attribute
        in_shape: tuple or list

    Returns:
        None, raises error if shape does not match
    """
    assert hasattr(in_item, 'shape')
    if not in_item.shape == in_shape:
        if raise_error:
            raise RuntimeError(f'shape of {in_item.shape} does not match {in_shape}')
        else:
            return False
    else:
        return True


""" Distribution construction utils"""


def make_batch_independent_normal_torch(mean, std, device):
    """ Make a torch RV that has a batch of normal distributions.
    These will have no covariance at all, hence is a collection of independent 1D
    normal distributions, or equivalently, a diagonal multivariate normal distribution.

    Args:
        mean: mean predictions, size (num_pts, dim_y)
            can be dtype numpy array, torch tensor, list
        std: std predictions, size (num_pts, dim_y)
            can be dtype numpy array, torch tensor, list
        device:
            torch device, cuda or cpu

    Returns:
        A torch RV which has methods e.g. sample, log_prob, ...
    """
    num_pts, dim_y = mean.shape
    assert std.shape == (num_pts, dim_y)

    mean = convert_to_tensor(mean, device)
    std = convert_to_tensor(std, device)

    torch_rv = Normal(loc=mean, scale=std)

    return torch_rv


def make_batch_multivariate_normal_diagcov_torch(mean, std, device):
    """ Make a torch RV that has a batch of multivariate distributions.
    Given just standard deviations as inputs, a diagonal matrix will be
    constructed for inputs to the RV construction.
    Hence, there will be convariance, but they will all be diagonal.

    Args:
        mean: mean predictions, size (num_pts, dim_y)
            can be dtype numpy array, torch tensor, list
        std: std predictions, size (num_pts, dim_y)
            can be dtype numpy array, torch tensor, list
        device:
            torch device, cuda or cpu

    Returns:
        A torch RV which has methods e.g. sample, log_prob, ...
    """
    num_pts, dim_y = mean.shape
    assert std.shape == (num_pts, dim_y)

    mean = convert_to_tensor(mean, device)
    std = convert_to_tensor(std, device)

    batch_cov_mat = torch.tile(torch.eye(dim_y).unsqueeze(0), dims=(num_pts, 1, 1))
    batch_cov_mat = batch_cov_mat.to(device)
    assert batch_cov_mat.shape == (num_pts, dim_y, dim_y)
    batch_cov_mat = std.unsqueeze(1) * batch_cov_mat * std.unsqueeze(2)
    assert batch_cov_mat.shape == (num_pts, dim_y, dim_y)
    torch_rv = MultivariateNormal(
        loc=mean,
        covariance_matrix=batch_cov_mat
    )

    return torch_rv


def make_batch_multivariate_normal_torch(mean, cov, device):
    """ Make a torch RV that has a batch of normal distributions.
    These will have covariance, hence will not be isotropic.

    Args:
        mean: mean predictions, size (num_pts, dim_y)
            can be dtype numpy array, torch tensor, list
        cov: std predictions, size (num_pts, dim_y)
            can be dtype numpy array, torch tensor, list
        device:
            torch device, cuda or cpu

    Returns:
        A torch RV which has methods e.g. sample, log_prob, ...
    """

    num_pts, dim_y = mean.shape
    assert cov.shape == (num_pts, dim_y, dim_y)

    mean = convert_to_tensor(mean, device)
    cov = convert_to_tensor(cov, device)

    torch_rv = MultivariateNormal(
        loc=mean,
        covariance_matrix=cov
    )

    return torch_rv


""" HDR recal utils """


def torch_rv_fhyh_fhys(pred_rv, targets, num_samples, pred_has_std=False,
                       apply_distr_fn='pdf'):
    num_pts, dim_y = pred_rv.loc.shape

    yh = pred_rv.sample((num_samples,))  # (num_samples, num_pts, dim_y)
    assert yh.shape == (num_samples, num_pts, dim_y)
    yh_reshaped = yh.swapaxes(0, 1).numpy()
    assert yh_reshaped.shape == (num_pts, num_samples, dim_y)
    if apply_distr_fn == 'pdf':
        apply_pdf_out = torch_rv_apply_pdf(pred_rv=pred_rv, pred_has_std=pred_has_std,
                                           samples=yh_reshaped, targets=targets)
        out = {
            'fhys': apply_pdf_out['fhys'],
            'fhyh': apply_pdf_out['fhyh'],
        }
    elif apply_distr_fn == 'cdf':
        apply_cdf_out = torch_rv_apply_cdf(pred_rv=pred_rv, pred_has_std=pred_has_std,
                                           samples=yh_reshaped, targets=targets)
        out = {
            'fhys': apply_cdf_out['Fhys'],
            'fhyh': apply_cdf_out['Fhyh'],
        }
    out['yh'] = yh_reshaped

    return out


def torch_rv_apply_pdf(pred_rv, pred_has_std, samples, targets=None):
    num_pts, dim_y = pred_rv.loc.shape
    # get fhys
    fhys = None
    if targets is not None:
        assert targets.shape == (num_pts, dim_y)
        fhys = pred_rv.log_prob(torch.Tensor(targets)).exp().numpy()
        if pred_has_std:
            assert fhys.shape == (num_pts, dim_y)
            fhys = np.prod(fhys, axis=1)
        assert fhys.shape == (num_pts,)

    # get fhyh
    # yh = pred_rv.sample((num_samples,))  # (num_samples, num_pts, dim_y)
    assert samples.shape[0] == num_pts
    assert samples.shape[2] == dim_y
    _, num_samples, _ = samples.shape
    yh = samples.swapaxes(0, 1)
    assert yh.shape == (num_samples, num_pts, dim_y)
    if isinstance(yh, np.ndarray):
        yh = torch.Tensor(yh)
    fhyh = pred_rv.log_prob(yh).exp().numpy()  # (num_samples, num_pts)
    if pred_has_std:
        assert fhyh.shape == (num_samples, num_pts, dim_y)
        fhyh = np.prod(fhyh, axis=2)
    assert fhyh.shape == (num_samples, num_pts)
    # yh = yh.swapaxes(0, 1).numpy()  # (num_pts, num_samples, dim_y)
    fhyh = fhyh.T  # (num_pts, num_samples)

    out = {'fhys': fhys, 'fhyh': fhyh}

    return out


def torch_rv_apply_cdf(pred_rv, pred_has_std, samples, targets=None):
    num_pts, dim_y = pred_rv.loc.shape
    # get Fhys
    Fhys = None
    if targets is not None:
        assert targets.shape == (num_pts, dim_y)
        Fhys = pred_rv.cdf(torch.Tensor(targets)).numpy()
        if pred_has_std:
            assert Fhys.shape == (num_pts, dim_y)
            Fhys = np.prod(Fhys, axis=1)
        assert Fhys.shape == (num_pts,)

    # get Fhyh
    # yh = pred_rv.sample((num_samples,))  # (num_samples, num_pts, dim_y)
    assert samples.shape[0] == num_pts
    assert samples.shape[2] == dim_y
    _, num_samples, _ = samples.shape
    yh = samples.swapaxes(0, 1)
    assert yh.shape == (num_samples, num_pts, dim_y)
    if isinstance(yh, np.ndarray):
        yh = torch.Tensor(yh)
    Fhyh = pred_rv.cdf(yh).numpy()  # (num_samples, num_pts)
    if pred_has_std:
        assert Fhyh.shape == (num_samples, num_pts, dim_y)
        Fhyh = np.prod(Fhyh, axis=2)
    assert Fhyh.shape == (num_samples, num_pts)
    # yh = yh.swapaxes(0, 1).numpy()  # (num_pts, num_samples, dim_y)
    Fhyh = Fhyh.T  # (num_pts, num_samples)

    out = {'Fhys': Fhys, 'Fhyh': Fhyh}

    return out


""" Basis Recal utils """


def is_invertible(in_matrix):
    return np.abs(np.linalg.det(in_matrix)) > 1e-10
