import numpy as np
import pickle as pkl

from torch_utils import make_batch_independent_normal_torch
from run_recal import recalibrate
from metrics import get_all_sample_metrics_from_samples


def run_single_seed(seed):
    print("=" * 20 + f" Running Experiment for Seed={seed} " + "=" * 20)
    NUM_SAMPLES = 50000  # number of samples to draw for HDR recalibration
    NUM_TRIALS = 20  # number of trials
    HDR_DELTA = 0.01  # width of bins for HDR calibration
    ENERGY_BETA = 1.7  # beta parameter for energy score
    ENERGY_NUM_PROCESSES = 8  # number of processes to use for computing energy score

    # load the data and prediction for the dataset scpf
    pred_dict = pkl.load(open(f"data_predictions/scpf_pred_dict_seed{seed}.pkl", "rb"))
    dim_y = pred_dict["tr"]["y"].shape[1]

    # dictionaries to save the results
    methods_metrics = {
        "Prehoc": {},
        "SD": {},
        "HDR": {},
    }

    # for NUM_TRIALS times of sampling
    for trial_idx in range(NUM_TRIALS):
        print(f"\nRunning Trial {trial_idx} for Seed={seed}...")
        # 1) Evaluate HDR-recalibrate samples
        hdr_recal_out = recalibrate(
            pred_means=pred_dict["val"]["mean_pred"],
            pred_stds=pred_dict["val"]["std_pred"],
            targets=pred_dict["val"]["y"],
            recal_type="hdr",
            num_samples=NUM_SAMPLES,
            test_pred_means=pred_dict["te"]["mean_pred"],
            test_pred_stds=pred_dict["te"]["std_pred"],
            use_mean_type="bias_adj",
            use_scale_type="std_arr_opt_error_corr_mat",
            hdr_delta_for_recal=HDR_DELTA,
            scale_opt_criterion="by_dim",
            num_scale_opt_samples=0,
        )
        num_recal_samples = hdr_recal_out["te_recal_samples"].shape[1]
        print(f"Evaluating HDR-recalibrated samples")
        hdr_eval_metrics, hdr_eval_info = get_all_sample_metrics_from_samples(
            dim_y=dim_y,
            torch_rv=hdr_recal_out["te_rv"],
            pred_had_std=False,
            pred_samples=hdr_recal_out["te_recal_samples"],
            targets=pred_dict["te"]["y"],
            hdr_mace_setting={"hdr_delta": HDR_DELTA},
            energy_score_setting={
                "energy_beta": ENERGY_BETA,
                "num_processes": ENERGY_NUM_PROCESSES,
            },
        )

        # 2) Evaluate SD-recalibrated samples
        sd_recal_out = recalibrate(
            pred_means=pred_dict["val"]["mean_pred"],
            pred_stds=pred_dict["val"]["std_pred"],
            targets=pred_dict["val"]["y"],
            recal_type="sd",
            num_samples=num_recal_samples,
            test_pred_means=pred_dict["te"]["mean_pred"],
            test_pred_stds=pred_dict["te"]["std_pred"],
            use_mean_type="orig",
            use_scale_type="orig",
        )

        print(f"Evaluating SD-recalibrated samples")
        sd_eval_metrics, sd_eval_info = get_all_sample_metrics_from_samples(
            dim_y=sd_recal_out["te_rv"].loc.shape[1],
            torch_rv=sd_recal_out["te_rv"],
            pred_had_std=True,
            pred_samples=sd_recal_out["te_recal_samples"],
            targets=pred_dict["te"]["y"],
            hdr_mace_setting={"hdr_delta": HDR_DELTA},
            energy_score_setting={
                "energy_beta": ENERGY_BETA,
                "num_processes": ENERGY_NUM_PROCESSES,
            },
        )

        # 3) Evaluate Prehoc samples
        # first produce samples from the Prehoc distribution
        prehoc_pred_te_rv = make_batch_independent_normal_torch(
            mean=pred_dict["te"]["mean_pred"],
            std=pred_dict["te"]["std_pred"],
            device="cpu",
        )
        prehoc_pred_te_samples = prehoc_pred_te_rv.sample(
            (num_recal_samples,)
        ).swapaxes(0, 1)

        print(f"Evaluating Prehoc samples")
        prehoc_eval_metrics, prehoc_eval_info = get_all_sample_metrics_from_samples(
            dim_y=dim_y,
            torch_rv=prehoc_pred_te_rv,
            pred_had_std=True,
            pred_samples=prehoc_pred_te_samples.numpy(),
            targets=pred_dict["te"]["y"],
            hdr_mace_setting={"hdr_delta": HDR_DELTA},
            energy_score_setting={
                "energy_beta": ENERGY_BETA,
                "num_processes": ENERGY_NUM_PROCESSES,
            },
        )
        curr_trial_methods_metrics = {
            "Prehoc": prehoc_eval_metrics,
            "SD": sd_eval_metrics,
            "HDR": hdr_eval_metrics,
        }

        for method, curr_trial_metrics in curr_trial_methods_metrics.items():
            for k, v in curr_trial_metrics.items():
                if k not in methods_metrics[method]:
                    methods_metrics[method][k] = []
                if k == "sd_mace":
                    # if the metric is sd_mace (i.e. calibration error of each dimension
                    # separately), then we need to take the mean over the 3 dimensions
                    methods_metrics[method][k].append(np.mean(v))
                else:
                    methods_metrics[method][k].append(v)

    print("=" * 20 + f" Results for Seed={seed} " + "=" * 20)
    for method, metrics in methods_metrics.items():
        print(f"{method} metrics")
        for k, v in metrics.items():
            print(f"{k}: {np.mean(v):.4f} +- {np.std(v) / np.sqrt(len(v)):.4f}")

    return methods_metrics


def print_results_across_seeds(list_of_result_dicts):
    full_dict = {}
    first_seed_dict = list_of_result_dicts[0]
    for method, metrics in first_seed_dict.items():
        full_dict[method] = {}
        for k, v in metrics.items():
            full_dict[method][k] = []

    for seed_result in list_of_result_dicts:
        for method, metrics in seed_result.items():
            for k, v in metrics.items():
                full_dict[method][k] += v

    print("=" * 20 + " Results Across All Seeds " + "=" * 20)
    for method, metrics in full_dict.items():
        print(f"{method} metrics")
        for k, v in metrics.items():
            print(f"{k}: {np.mean(v):.4f} +- {np.std(v) / np.sqrt(len(v)):.4f}")


if __name__ == "__main__":
    seed_result_list = []
    for seed in range(5):
        seed_result = run_single_seed(seed)
        seed_result_list.append(seed_result)

    print_results_across_seeds(seed_result_list)
