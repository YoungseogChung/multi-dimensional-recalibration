"""
Define function to get all metrics that will be reported.
"""
from typing import Any, Union, Optional

import numpy as np
import torch

from hdr_recal import conditional_hdr_recalibration
from torch_utils import (
    torch_rv_apply_pdf,
    check_shape,
)
from metrics.sample_metrics import (
    sample_based_mace_by_dim,
    energy_score_vectorized,
)


def get_all_sample_metrics_from_samples(
        dim_y: int,
        torch_rv: Any,
        pred_had_std: bool,
        pred_samples: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        hdr_mace_setting: dict,
        energy_score_setting: Optional[dict] = None,
):
    """

        Args:
            dim_y: int, dim of y
            torch_rv: torch_rv object (e.g. from torch_utils.make_***_normal_torch fn)
            pred_had_std: bool, whether torch_rv has std (if has cov, set to False)
            pred_samples: array or tensor, samples from predictive distribution
                shape (num_pts, num_pred_samples, dim_y)
            targets: array or tensor, target data from gt distribution,
                shape (num_pts, dim_y)
            hdr_mace_setting: must have "num_pred_samples" (int), "hdr_delta" (float)
                "num_pred_samples": int, number of samples to use from predictive distribution
                "hdr_delta": float, increment of HDR level sets, must be in (0, 1)
            energy_score_setting: must have "energy_beta" (float)
                "energy_beta": float, beta hp to calculate energy score, must be in (0, 2)
                "num_processes": int, number of processes to use to compute energy score

        Returns:
            a dictionary of metrics, including
                hdr mace, sd mace, energy score, divergence metrics
                (maybe) fit a kde and do nll
    """

    # check all shapes
    assert len(pred_samples.shape) == 3
    num_pts = pred_samples.shape[0]
    num_pred_samples = pred_samples.shape[1]
    check_shape(pred_samples, (num_pts, num_pred_samples, dim_y))

    check_shape(targets, (num_pts, dim_y))

    metrics_dict = {}  # this will be returned
    info = {}  # this will be returned

    # HDR MACE
    hdr_delta = hdr_mace_setting['hdr_delta']
    apply_f_out = torch_rv_apply_pdf(
        pred_rv=torch_rv, pred_has_std=pred_had_std, samples=pred_samples,
        targets=targets)
    fhyh = apply_f_out['fhyh']
    fhys = apply_f_out['fhys']
    hdr_recal_obj = conditional_hdr_recalibration(
        dim_y=dim_y, fhyh=fhyh, fhys=fhys, hdr_delta=hdr_delta)
    hdr_mace = hdr_recal_obj.mace
    metrics_dict["hdr_mace"] = hdr_mace

    # SD MACE
    sd_mace, sd_mace_info = sample_based_mace_by_dim(
        samples=pred_samples, y=targets, return_levels=True)
    metrics_dict["sd_mace"] = sd_mace
    info["sd_mace_info"] = sd_mace_info

    # Energy Score
    if energy_score_setting is not None:
        energy_beta = energy_score_setting["energy_beta"]
        energy_num_processes = energy_score_setting["num_processes"]
        energy_score = energy_score_vectorized(samples=pred_samples, y=targets,
                                               beta=energy_beta,
                                               num_processes=energy_num_processes)
        metrics_dict["energy_score"] = energy_score

    return metrics_dict, info
