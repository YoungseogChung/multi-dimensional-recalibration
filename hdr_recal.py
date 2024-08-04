import numpy as np
import torch
import tqdm

from torch_utils import (
    check_shape,
    make_batch_multivariate_normal_torch,
)

""" Utils """


def hdr_alpha(query_val, base_pdf_values):
    assert len(base_pdf_values.shape) == 1
    rank_among_pdfs = np.mean(base_pdf_values >= query_val)

    return rank_among_pdfs


""" Recalibration Classes/Functions """


class conditional_hdr_recalibration():
    def __init__(
            self,
            dim_y,
            fhyh,  # (num_data, num_samples, 1) or (num_data, num_samples)
            fhys,  # (num_data, 1) or (num_data, )
            hdr_delta=0.05,
    ):
        self.dim_y = int(dim_y)
        self.hdr_delta = hdr_delta
        num_data, num_samples = fhyh.shape[:2]
        assert fhys.shape[0] == num_data

        self.fhyh = fhyh.reshape(num_data, num_samples)
        self.fhys = fhys.reshape(num_data, 1)

        self.hdr_levels = np.sort(np.mean(self.fhyh >= self.fhys, axis=1))
        self.hdr_level_bounds = np.arange(0, 1 + hdr_delta, hdr_delta)
        self.num_bounds = len(self.hdr_level_bounds) - 1
        self.num_in_bucket = np.histogram(self.hdr_levels, bins=self.hdr_level_bounds)[
            0]
        self.prop_in_bucket = self.num_in_bucket / np.sum(self.num_in_bucket)
        self.sample_scale_constant = 1 / np.max(self.prop_in_bucket)
        self.sample_prop_per_bucket = self.sample_scale_constant * self.prop_in_bucket

        self.min_req_samples = int(
            (self.num_bounds - 1) / np.max(self.sample_prop_per_bucket))

    @property
    def mace(self):
        exp_props = np.linspace(0, 1, len(self.hdr_levels))
        obs_props = np.sort(np.array(self.hdr_levels).flatten())
        mace = np.mean(np.abs(exp_props - obs_props))
        return mace

    def check_torch_attributes(
            self,
            device,
    ):
        if not hasattr(self, 'hdr_level_bounds_torch'):
            self.hdr_level_bounds_torch = torch.from_numpy(self.hdr_level_bounds).to(
                device)
        if not hasattr(self, 'sample_prop_per_bucket_torch'):
            self.sample_prop_per_bucket_torch = torch.from_numpy(
                self.sample_prop_per_bucket).to(device)
        # if hasattr(self, 'mean_bias') and not hasattr(self, 'mean_bias_tensor'):
        #     self.mean_bias_tensor = torch.from_numpy(self.mean_bias).to(device)
        # if hasattr(self, 'error_corr_mat') and not hasattr(self, 'error_corr_mat_tensor'):
        #     self.error_corr_mat_tensor = torch.from_numpy(self.error_corr_mat).to(device)

    def get_rejection_prob(
            self,
            y_hat,  # (num_data, num_samples, dim_y)
            f_hat_y_hat,  # (num_data, num_samples)
    ):
        num_data = y_hat.shape[0]
        num_samples = y_hat.shape[1]
        assert f_hat_y_hat.shape == (num_data, num_samples)

        hdr_level_bound_idxs = (self.hdr_level_bounds * num_samples).astype(int)
        num_samples_per_bucket_arr = hdr_level_bound_idxs[1:] - hdr_level_bound_idxs[
                                                                :-1]

        order_fhyh = np.flip(np.argsort(f_hat_y_hat, axis=1),
                             axis=1)  # (num_data, num_samples)

        pt_idxs_per_hdr = []
        for bin_idx in range(self.num_bounds):
            pt_idxs_per_hdr.append(
                order_fhyh[:,
                hdr_level_bound_idxs[bin_idx]:hdr_level_bound_idxs[bin_idx + 1]]
                # (num_data, c_bin)
            )

        fhyh_sample_prob = -1 * np.ones_like(f_hat_y_hat)
        for bin_idx in range(self.num_bounds):
            for data_idx in range(num_data):
                curr_chunk = fhyh_sample_prob[
                    data_idx, pt_idxs_per_hdr[bin_idx][data_idx]]
                np.testing.assert_equal(curr_chunk, -1)
                fhyh_sample_prob[data_idx, pt_idxs_per_hdr[bin_idx][data_idx]] = \
                self.sample_prop_per_bucket[bin_idx]
        assert np.all(fhyh_sample_prob.flatten() > 0)

        return fhyh_sample_prob

    def recal_sample(
            self,
            y_hat,  # (num_data, num_samples, dim_y)
            f_hat_y_hat,  # (num_data, num_samples)
    ):
        num_data = y_hat.shape[0]
        num_samples = y_hat.shape[1]
        assert f_hat_y_hat.shape == (num_data, num_samples)

        hdr_level_bound_idxs = (self.hdr_level_bounds * num_samples).astype(int)
        num_samples_per_bucket_arr = hdr_level_bound_idxs[1:] - hdr_level_bound_idxs[
                                                                :-1]

        order_fhyh = np.flip(np.argsort(f_hat_y_hat, axis=1),
                             axis=1)  # (num_data, num_samples)

        pt_idxs_per_hdr = []
        for bin_idx in range(self.num_bounds):
            pt_idxs_per_hdr.append(
                order_fhyh[
                :,
                hdr_level_bound_idxs[bin_idx]:hdr_level_bound_idxs[bin_idx + 1]
                ]  # (num_data, c_bin)
            )

        num_collect_per_bucket_arr = self.sample_prop_per_bucket * num_samples_per_bucket_arr
        # array times array

        rand_idx_per_bucket = [
            np.random.choice(
                num_samples_per_bucket_arr[bin_idx],
                size=int(num_collect_per_bucket_arr[bin_idx]),
                replace=False
            )
            # the following line is only for testing
            # #np.arange(int(num_collect_per_bucket_arr[bin_idx]))
            for bin_idx in range(self.num_bounds)  # TODO: this is probably a class att
        ]

        recal_sample_idxs = [
            pt_idxs_per_hdr[bin_idx][:, rand_idx_per_bucket[bin_idx]]
            for bin_idx in range(self.num_bounds)  # TODO: this is probably a class att
        ]  # each element is (num_data, randselected_bin)

        recal_sample_idxs = np.concatenate(recal_sample_idxs,
                                           axis=1)  # (num_data, sum of randselected_bin)

        # import pdb; pdb.set_trace() #TODO: have a bit of a problem here with array slicing
        # y_hat is (num_data, num_samples, dim_y)
        recal_samples = np.stack(
            [y_hat[data_idx][recal_sample_idxs[data_idx]] for data_idx in
             range(num_data)])
        recal_samples_f_hat_y_hat = np.stack(
            [f_hat_y_hat[data_idx][recal_sample_idxs[data_idx]] for data_idx in
             range(num_data)])

        return recal_samples, recal_samples_f_hat_y_hat

    def torch_recal_sample(
            self,
            y_hat: torch.Tensor,  # (num_data, num_samples, dim_y)
            f_hat_y_hat: torch.Tensor,  # (num_data, num_samples),
            device='cpu'
    ):
        num_data = y_hat.shape[0]
        num_samples = y_hat.shape[1]
        assert f_hat_y_hat.shape == (num_data, num_samples)

        hdr_level_bound_idxs = (
                self.hdr_level_bounds_torch * num_samples).int()  # tensor
        num_samples_per_bucket_arr = hdr_level_bound_idxs[1:] - hdr_level_bound_idxs[
                                                                :-1]  # tensor

        order_fhyh = torch.flip(torch.argsort(f_hat_y_hat, dim=1),
                                dims=(1,))  # (num_data, num_samples), tensor

        pt_idxs_per_hdr = []  # list of tensors
        for bin_idx in range(self.num_bounds):
            pt_idxs_per_hdr.append(
                order_fhyh[
                :,
                hdr_level_bound_idxs[bin_idx]:hdr_level_bound_idxs[bin_idx + 1]
                ]  # (num_data, c_bin)
            )

        num_collect_per_bucket_arr = self.sample_prop_per_bucket_torch * num_samples_per_bucket_arr  # tensor
        # array times array

        rand_idx_per_bucket = [
            torch.randint(
                low=0,
                high=num_samples_per_bucket_arr[bin_idx],
                size=(int(num_collect_per_bucket_arr[bin_idx]),)
            )
            # the following line is only for testing
            # #torch.arange(int(num_collect_per_bucket_arr[bin_idx]))
            for bin_idx in range(self.num_bounds)
        ]  # list of tensors

        recal_sample_idxs = [
            pt_idxs_per_hdr[bin_idx][:, rand_idx_per_bucket[bin_idx]]
            for bin_idx in range(self.num_bounds)
        ]  # each element is (num_data, randselected_bin)

        recal_sample_idxs = torch.cat(recal_sample_idxs,
                                      dim=1)  # (num_data, sum of randselected_bin)

        # import pdb; pdb.set_trace() #TODO: have a bit of a problem here with array slicing
        # y_hat is (num_data, num_samples, dim_y)
        recal_samples = torch.stack(
            [
                y_hat[data_idx][recal_sample_idxs[data_idx]]
                for data_idx in range(num_data)
            ],
            dim=0
        )
        recal_samples_f_hat_y_hat = torch.stack(
            [
                f_hat_y_hat[data_idx][recal_sample_idxs[data_idx]]
                for data_idx in range(num_data)
            ],
            dim=0
        )

        return recal_samples, recal_samples_f_hat_y_hat

    def produce_recal_samples_from_mean_std(
            self,
            means,
            stds,
            mean_bias,
            std_ratio,
            error_corr_mat,
            num_samples,
            device,
    ):
        """
        Produce recalibrated samples
        Assumes TORCH!
        Args:
            dim_y:
            means: (num_pts, dim_y)
            stds: (num_pts, dim_y)
            mean_bias: (dim_y,)
            std_ratio:
            error_corr_mat: (dim_y, dim_y)
            num_samples:
            hdr_recal:
            device:

        Returns:

        """
        raise RuntimeError("do not use this function")
        num_pts = means.shape[0]
        # print(means.shape, stds.shape, mean_bias.shape, error_corr_mat.shape)
        check_shape(means, (num_pts, self.dim_y), raise_error=True)
        check_shape(stds, (num_pts, self.dim_y), raise_error=True)
        assert (
                check_shape(mean_bias, (self.dim_y,), raise_error=False)
                or check_shape(mean_bias, (self.dim_y, 1), raise_error=False)
        )
        check_shape(error_corr_mat, (self.dim_y, self.dim_y), raise_error=True)

        # checking for torch attributes
        self.check_torch_attributes(device)

        bias_adj_mean = means - mean_bias
        ratio_adj_std = std_ratio * stds
        corr_adj_cov = (
                ratio_adj_std[:, np.newaxis, :]
                * error_corr_mat
                * ratio_adj_std[:, :, np.newaxis]
        )
        rv = make_batch_multivariate_normal_torch(
            bias_adj_mean,
            corr_adj_cov,
            device
        )

        yh = rv.sample((num_samples,))  # (num_samples, num_pts, dim_y)
        fhyh = rv.log_prob(yh).exp()  # (num_samples, num_pts)

        yh = yh.swapaxes(0, 1)  # (num_pts, num_samples)
        fhyh = fhyh.T  # (num_pts, num_samples)

        recal_yh, recal_fhyh = self.torch_recal_sample(yh, fhyh)

        return recal_yh, recal_fhyh

    def orig_recal_sample(  ### DEPRECATED ###
            self,
            y_hat,  # (num_data, num_samples, dim_y)
            f_hat_y_hat,  # (num_data, num_samples)
    ):
        num_data = y_hat.shape[0]
        num_samples = y_hat.shape[1]
        assert f_hat_y_hat.shape == (num_data, num_samples)

        hdr_level_bound_idxs = (self.hdr_level_bounds * num_samples).astype(int)

        recal_samples_per_x = []
        recal_samples_f_hat_y_hat_per_x = []
        for x_idx in tqdm.tqdm(range(num_data)):
            order_fhyh = np.argsort(f_hat_y_hat[x_idx])[::-1]
            pt_idxs_per_hdr = []
            for bin_idx in range(self.num_bounds):
                pt_idxs_per_hdr.append(
                    order_fhyh[
                    hdr_level_bound_idxs[bin_idx]
                    :hdr_level_bound_idxs[bin_idx + 1]]
                )

            recal_sample_idxs = []
            for bin_idx, prop in enumerate(self.sample_prop_per_bucket):
                num_samples_in_bucket = pt_idxs_per_hdr[bin_idx].shape[0]
                curr_num_collect = int(prop * num_samples_in_bucket)
                rand_idx = np.random.choice(
                    num_samples_in_bucket,
                    size=curr_num_collect,
                    replace=False
                )
                # rand_idx = np.arange(curr_num_collect)
                recal_sample_idxs.append(pt_idxs_per_hdr[bin_idx][rand_idx])

            recal_sample_idxs = np.concatenate(recal_sample_idxs)
            recal_samples = y_hat[x_idx][recal_sample_idxs]
            recal_samples_f_hat_y_hat = f_hat_y_hat[x_idx][recal_sample_idxs]

            recal_samples_per_x.append(recal_samples)
            recal_samples_f_hat_y_hat_per_x.append(recal_samples_f_hat_y_hat)

        # at each x, number of recal samples are the same
        recal_samples_per_x = np.stack(recal_samples_per_x)
        recal_samples_f_hat_y_hat_per_x = np.stack(recal_samples_f_hat_y_hat_per_x)

        return recal_samples_per_x, recal_samples_f_hat_y_hat_per_x

    def rejection_sample(
            self,
            f_val_input_point,  # scalar
            f_hat_y_hat,  # (N,)
    ):
        num_samples = f_hat_y_hat.shape[0]
        hdr_level_bound_idxs = (self.hdr_level_bounds * num_samples).astype(int)
        order_fhyh = np.argsort(f_hat_y_hat)[::-1]

        input_point_percentile = np.mean(f_hat_y_hat >= f_val_input_point)

        bin_idx = np.digitize(input_point_percentile,
                              self.hdr_level_bounds)  # gets end index of bin
        bin_sample_prop = self.sample_prop_per_bucket[bin_idx - 1]
        return np.random.uniform() <= bin_sample_prop
