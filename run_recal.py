import numpy as np
import torch

import torch_utils
from torch_utils import torch_rv_fhyh_fhys
from hdr_recal import conditional_hdr_recalibration
from recal_utils import optimize_ratio
from single_dim_recal import (
    iso_single_dim_recalibrator,
    sample_gaussian_with_quantiles,
)


""" Mean and Std Adjustment Functions """


def adjust_pred_mean(pred_means, targets):
    num_pts, dim_y = targets.shape
    assert pred_means.shape == (num_pts, dim_y)
    error = pred_means - targets

    # calculate bias and bias-adjusted-mean
    bias_by_dim = np.mean(error, axis=0)
    assert bias_by_dim.shape == (dim_y,)
    adj_means = pred_means - bias_by_dim
    adj_bias_by_dim = np.mean(adj_means - targets, axis=0)
    # tests that the bias adjusted mean actually has ~0 bias
    try:
        assert np.max(np.abs(adj_bias_by_dim)) < 1e-3
    except:
        import pdb; pdb.set_trace()

    info = {'bias_by_dim': bias_by_dim}
    return adj_means, info


def cov_from_error_corr_mat(means, stds, targets):
    num_pts, dim_y = targets.shape
    assert means.shape == (num_pts, dim_y)
    assert stds.shape == (num_pts, dim_y)

    error = means - targets
    error_corr_mat = np.corrcoef(error.T)  # (dim_y, dim_y)
    assert error_corr_mat.shape == (dim_y, dim_y)

    adj_cov = (
            stds[:, np.newaxis, :]
            * error_corr_mat
            * stds[:, :, np.newaxis]
    )

    info = {'error_corr_mat': error_corr_mat}

    return adj_cov, info


def adjust_std(means, stds, targets, std_opt_criterion, num_samples):
    num_pts, dim_y = targets.shape
    assert means.shape == (num_pts, dim_y)
    assert stds.shape == (num_pts, dim_y)

    opt_result = optimize_ratio(
        criterion=std_opt_criterion,
        mean=means, std=stds, data=targets, num_samples=num_samples
    )
    opt_ratio = opt_result['ratio']
    # single scalar to multiple to stds
    adj_std = stds * opt_ratio

    info = {
        'std_ratio': opt_ratio,
    }

    return adj_std, info


""" Handle function """


def handle_mean_adjustment(means, targets, use_mean_type):
    if use_mean_type == 'orig':
        use_mean = means
        mean_adj_fn = lambda pred_mean: pred_mean
    elif use_mean_type == 'bias_adj':
        use_mean, adj_mean_info = adjust_pred_mean(means, targets)
        mean_adj_fn = lambda pred_mean: pred_mean - adj_mean_info['bias_by_dim']
    else:
        raise RuntimeError()

    test_mean_fn_output = mean_adj_fn(means)
    np.testing.assert_allclose(use_mean, test_mean_fn_output)

    return use_mean, mean_adj_fn


def handle_scale_adjustment(means, stds, targets, use_scale_type, scale_opt_criterion,
                            num_scale_opt_samples):
    num_pts, dim_y = targets.shape

    pred_has_std = None
    use_std = use_cov = None
    if use_scale_type == 'orig':
        pred_has_std = True
        use_std = stds
        std_adj_fn = lambda pred_std: pred_std
        cov_adj_fn = None
    elif use_scale_type == 'std_scalar_opt':
        pred_has_std = True
        adj_std, adj_std_info = adjust_std(means, stds, targets, scale_opt_criterion,
                                           num_scale_opt_samples)
        use_std = adj_std
        std_adj_fn = lambda pred_std: pred_std * adj_std_info['std_ratio']
        cov_adj_fn = None
    elif use_scale_type == 'std_array_opt':
        pred_has_std = True
        adj_std_by_dim = []
        adj_std_ratio_by_dim = []
        for dim_idx in range(dim_y):
            dim_means = means[:, [dim_idx]]
            dim_stds = stds[:, [dim_idx]]
            dim_targets = targets[:, [dim_idx]]
            dim_adj_std, dim_adj_std_info = adjust_std(dim_means, dim_stds, dim_targets,
               scale_opt_criterion, num_scale_opt_samples)
            adj_std_by_dim.append(dim_adj_std)
            adj_std_ratio_by_dim.append(dim_adj_std_info['std_ratio'])
        adj_std = np.concatenate(adj_std_by_dim, axis=1)  # TODO: is this right?
        assert adj_std.shape == stds.shape
        adj_std_ratio = np.stack(adj_std_ratio_by_dim)

        use_std = adj_std
        std_adj_fn = lambda pred_std: pred_std * adj_std_ratio
        cov_adj_fn = None
    elif use_scale_type == 'error_corr_mat':
        pred_has_std = False
        adj_cov, adj_cov_info = cov_from_error_corr_mat(means, stds, targets)
        use_std = stds
        use_cov = adj_cov
        std_adj_fn = lambda pred_std: pred_std
        cov_adj_fn = lambda pred_std: (
                pred_std[:, np.newaxis, :]
                * adj_cov_info['error_corr_mat']
                * pred_std[:, :, np.newaxis]
        )
    elif use_scale_type == 'std_arr_opt_error_corr_mat':
        adj_std_by_dim = []
        adj_std_ratio_by_dim = []
        for dim_idx in range(dim_y):
            dim_means = means[:, [dim_idx]]
            dim_stds = stds[:, [dim_idx]]
            dim_targets = targets[:, [dim_idx]]
            dim_adj_std, dim_adj_std_info = adjust_std(dim_means, dim_stds, dim_targets,
                                                       scale_opt_criterion,
                                                       num_scale_opt_samples)
            adj_std_by_dim.append(dim_adj_std)
            adj_std_ratio_by_dim.append(dim_adj_std_info['std_ratio'])
        adj_std = np.concatenate(adj_std_by_dim, axis=1)
        assert adj_std.shape == stds.shape
        adj_std_ratio = np.stack(adj_std_ratio_by_dim)

        use_std = adj_std
        std_adj_fn = lambda pred_std: pred_std * adj_std_ratio

        ### then do error_corr_mat fitting
        pred_has_std = False
        adj_cov, adj_cov_info = cov_from_error_corr_mat(means, use_std, targets)
        use_cov = adj_cov
        cov_adj_fn = lambda pred_std: (
                std_adj_fn(pred_std)[:, np.newaxis, :]
                * adj_cov_info['error_corr_mat']
                * std_adj_fn(pred_std)[:, :, np.newaxis]
        )
    else:
        raise RuntimeError()

    assert use_std is not None and std_adj_fn is not None
    if not pred_has_std:
        assert use_cov is not None and cov_adj_fn is not None

    np.testing.assert_allclose(std_adj_fn(stds), use_std, rtol=1e-5)
    if not pred_has_std:
        np.testing.assert_allclose(cov_adj_fn(stds), use_cov, rtol=1e-5)

    return use_std, use_cov, pred_has_std, std_adj_fn, cov_adj_fn


""" Recalibration end to end """


def recalibrate(pred_means, pred_stds, targets, recal_type, num_samples,
                test_pred_means, test_pred_stds,
                use_mean_type='orig', use_scale_type='orig',
                hdr_delta_for_recal=0.01,
                scale_opt_criterion=None, num_scale_opt_samples=None):

    assert use_mean_type in ['orig', 'bias_adj']
    assert use_scale_type in ['orig', 'std_scalar_opt', 'std_array_opt',
                              'error_corr_mat', 'std_arr_opt_error_corr_mat']

    assert len(targets.shape) == 2
    num_pts, dim_y = targets.shape
    assert pred_means.shape == (num_pts, dim_y)
    assert pred_stds.shape == (num_pts, dim_y)
    print(f"Recalibrating with method {recal_type}, with {num_pts} recalibration datapoints")

    assert recal_type in ['sd', 'hdr']

    """ OPTION 1: original mean or bias-adjusted-mean """
    use_mean, mean_adj_fn = handle_mean_adjustment(pred_means, targets, use_mean_type)

    """ OPTION 2: original stds or optimized stds/covs """
    use_std, use_cov, pred_has_std, std_adj_fn, cov_adj_fn = handle_scale_adjustment(
        use_mean, pred_stds, targets, use_scale_type, scale_opt_criterion,
        num_scale_opt_samples)

    """ Make function to make RV's with adjustments """
    def make_rv_fn(pred_means, pred_stds):
        use_mean = mean_adj_fn(pred_means)
        use_std = std_adj_fn(pred_stds)
        use_cov = None
        if pred_has_std:
            # pred_rv = torch_utils.make_batch_multivariate_normal_diagcov_torch(
            #     mean=use_mean, std=use_std, device=torch.device('cpu'))
            pred_rv = torch_utils.make_batch_independent_normal_torch(
                mean=use_mean, std=use_std, device=torch.device('cpu'))
            out_rv_is_iso = True
        else:
            use_cov = cov_adj_fn(pred_stds)
            pred_rv = torch_utils.make_batch_multivariate_normal_torch(
                mean=use_mean, cov=use_cov, device=torch.device('cpu'))
            out_rv_is_iso = False
        info = {
            'adj_mean': use_mean, 'adj_std': use_std, 'adj_cov': use_cov,
            'pred_has_std': out_rv_is_iso
        }
        return pred_rv, info

    """ BEGIN: SAMPLE BASED RECAL """
    # just make the RV's to return
    if pred_has_std:
        pred_rv = torch_utils.make_batch_multivariate_normal_diagcov_torch(
            mean=use_mean, std=use_std, device=torch.device('cpu'))
    else:
        pred_rv = torch_utils.make_batch_multivariate_normal_torch(
            mean=use_mean, cov=use_cov, device=torch.device('cpu'))

    if recal_type == 'sd':
        """ recalibrate now """
        sd_recal = iso_single_dim_recalibrator(dim_y=dim_y, mus=use_mean, stds=use_std,
                                               ys=targets, init_train=True)
        # recal samples on same outputs
        sd_recal_samples = sd_recal.produce_recal_samples_from_mean_std(
            use_mean, use_std, num_samples)
        val_recal_samples = sd_recal_samples
        recal_obj = sd_recal
    elif recal_type == 'hdr':
        # first make the RV's
        if pred_has_std:
            pred_rv = torch_utils.make_batch_independent_normal_torch(
                mean=use_mean, std=use_std, device=torch.device('cpu'))
        else:
            pred_rv = torch_utils.make_batch_multivariate_normal_torch(
                mean=use_mean, cov=use_cov, device=torch.device('cpu'))

        rv_out = torch_rv_fhyh_fhys(
            pred_rv=pred_rv, targets=targets, num_samples=num_samples,
            pred_has_std=pred_has_std)
        fhys, fhyh, y_hat = rv_out['fhys'], rv_out['fhyh'], rv_out['yh']

        hdr_recal = conditional_hdr_recalibration(
            dim_y=dim_y, fhyh=fhyh, fhys=fhys,
            hdr_delta=hdr_delta_for_recal
        )
        # recal samples on same outputs
        hdr_recal_samples, _ = hdr_recal.recal_sample(y_hat=y_hat, f_hat_y_hat=fhyh)
        val_recal_samples = hdr_recal_samples

        recal_obj = hdr_recal
    else:
        raise RuntimeError()
    """ END: SAMPLE BASED RECAL """

    """ TESTING """
    te_rv, te_rv_info = make_rv_fn(test_pred_means, test_pred_stds)
    te_use_mean = te_rv_info['adj_mean']
    te_use_std = te_rv_info['adj_std']
    te_use_cov = te_rv_info['adj_cov']
    te_pred_has_std = te_rv_info['pred_has_std']
    if not pred_has_std:
        assert te_use_cov is not None  # just for checking

    ###
    te_rv_out = torch_rv_fhyh_fhys(pred_rv=te_rv, targets=None, num_samples=num_samples,
                                   pred_has_std=te_pred_has_std)
    te_fhyh, te_y_hat = te_rv_out['fhyh'], te_rv_out['yh']
    ###

    if recal_type == 'sd':
        te_sd_recal_quantiles = sd_recal.recal_sample_quantiles(
            sample_shape=(te_use_mean.shape[0], num_samples, dim_y))
        te_sd_recal_samples = sample_gaussian_with_quantiles(
            dim_y=dim_y, means=te_use_mean, stds=te_use_std,
            sample_quantiles=te_sd_recal_quantiles)

        te_recal_samples = te_sd_recal_samples
    elif recal_type == 'hdr':
        te_hdr_recal_samples, _ = hdr_recal.recal_sample(
            y_hat=te_y_hat,
            f_hat_y_hat=te_fhyh
        )

        te_recal_samples = te_hdr_recal_samples
    else:
        raise RuntimeError()

    out = {
        'val_adj_mean': use_mean,
        'val_adj_std': use_std,
        'val_adj_cov': use_cov,
        'val_rv': pred_rv,
        'val_recal_samples': val_recal_samples,
        'te_adj_mean': te_use_mean,
        'te_adj_std': te_use_std,
        'te_adj_cov': te_use_cov,
        'te_rv': te_rv,
        'te_recal_samples': te_recal_samples,
        'te_orig_samples': te_y_hat,
        'recal_obj': recal_obj
    }

    return out
