import numpy as np
import torch

from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from uncertainty_toolbox.metrics_calibration import (
    mean_absolute_calibration_error,
    get_proportion_lists_vectorized,
)

from torch_utils import check_shape


""" Util Functions """


def to_torch_tensors(in_arr):
    if isinstance(in_arr, torch.Tensor):
        out = in_arr
    elif isinstance(in_arr, np.ndarray):
        out = torch.from_numpy(in_arr)
    else:
        raise RuntimeError(f'Dunno what this type is {type(in_arr)}')
    
    return out


def normalize_basis(basis):
    norm_basis = basis / (np.linalg.norm(basis, axis=1) ** 2).reshape(-1,
                                                                      1)  # TODO: not sure if this is needed
    return norm_basis


""" Recalibration Functions """


def cal_by_dim(mus, stds, ys, mode='ermon'):
    
    """
    Args:
        mus: mean predictions, (N, dim_y)
        stds: std predictions, (N, dim_y)
        ys: true datapoints, (N, dim_y)
        
    
    """
    num_pts = mus.shape[0]
    dim_y = mus.shape[1]
    
    assert stds.shape == (num_pts, dim_y)
    assert ys.shape == (num_pts, dim_y)
    
    if mode=='ermon':
        recal_tr_x = np.zeros_like(ys)
        recal_tr_y = np.zeros_like(ys)
        
        for d in range(dim_y):
            mu = mus[:, d]
            std = stds[:, d]
            y = ys[:, d]

            cdf_pred = norm.cdf(y, loc=mu, scale=std)
            cdf_true = np.array([np.sum(cdf_pred < p)/len(cdf_pred) for p in cdf_pred])

            recal_tr_x[:, d] = cdf_pred
            recal_tr_y[:, d] = cdf_true

    elif mode=='ut':
        recal_tr_x = []
        recal_tr_y = []

        for d in range(dim_y):
            mu = mus[:, d]
            std = stds[:, d]
            y = ys[:, d]
            
            exp_props, obs_props = get_proportion_lists_vectorized(
                y_pred=mu.flatten(),
                y_std=std.flatten(),
                y_true=y.flatten(),
                prop_type='quantile'
            )
            
            recal_tr_x.append(obs_props)
            recal_tr_y.append(exp_props)

        recal_tr_x = np.stack(recal_tr_x).T
        recal_tr_y = np.stack(recal_tr_y).T
    
    return recal_tr_x, recal_tr_y


def train_torch_module(x, y, model, epochs, opt_type, lr):
    
    x = to_torch_tensors(x)
    y = to_torch_tensors(y)
    
    model.train()
    
    if opt_type=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    for _ in range(epochs):
        pred = model.forward(x)
        loss = model.loss_fn(pred, y)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    info = {
        'train_loss': losses,
        'optimizer': optimizer
    }
    
    return info
    

def sample_gaussian_with_quantiles(
    dim_y,
    means,
    stds,
    sample_quantiles,
):
    """ Produce samples from predictive conditional gaussian distributions,
    and evaluate the predictive pdf on the samples.
    If targets is not None, evaluate predictive pdf on targets too.
    
    Args:
        dim_y: dimension of conditional distribution
        means: array of size (num_pts, dim_y)
        stds:  array of size (num_pts, dim_y)
        sample_quantiles: quantiles to sample at from predictive distribution, array of size (num_pts, num_samples, dim_y)

    Returns:
        sample: array of size (num_pts, num_samples)
    """
    num_pts, num_samples, _ = sample_quantiles.shape  # (num_pts, num_samples, dim_y)
    assert sample_quantiles.shape[2] == dim_y
    
    def assert_shapes(check_arr):
        assert check_arr.shape == (num_pts, dim_y)
    
    for check_arr in [means, stds]:
        assert_shapes(check_arr)

    samples_by_dim = []

    for dim_idx in range(dim_y):
        
        dim_means = means[:, [dim_idx]]
        dim_stds = stds[:, [dim_idx]]
        dim_rv = norm(loc=dim_means, scale=dim_stds)
        # TODO: *** fix this part, I keep running into errors
        # import pdb; pdb.set_trace()
        sample_quantiles = np.clip(sample_quantiles, a_min=1e-3, a_max=1-1e-3)  # TODO: clipping thresh is hardcoded
        dim_samples = dim_rv.ppf(sample_quantiles[:, :, dim_idx])
        assert dim_samples.shape == (num_pts, num_samples)
        if not np.isfinite(dim_samples).all():
            raise ValueError(f'Samples for dimension {dim_idx} are non-finite')

        samples_by_dim.append(dim_samples)
    
    out = np.transpose(np.stack(samples_by_dim), (1, 2, 0))
    assert out.shape == (num_pts, num_samples, dim_y)
    
    return out


class iso_single_dim_recalibrator():
    def __init__(
        self,
        dim_y,
        mus=None,
        stds=None,
        ys=None,
        init_train=False
    ):
        """
        Args:
            mus, stds, ys: np.ndarray of size (num_datapoints, dim_y)
        """
        self.dim_y = dim_y
        self.model = [IsotonicRegression(out_of_bounds='clip') for _ in range(self.dim_y)] 
        self.trained = False
        if all([x is not None for x in [mus, stds, ys]]):
            self.mus = mus
            self.stds = stds
            self.ys = ys
            self.base_data_given = True
            
            if init_train:
                recal_tr_x, recal_tr_y = cal_by_dim(mus=self.mus, stds=self.stds, ys=self.ys)
                self.train(recal_tr_x, recal_tr_y)
                print('Trained isotonic single dimension recalibrator')
                
        else:
            self.mus = self.stds = self.ys = None
            self.base_data_given = False
            
    def train(self, x, y):
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        assert x.shape[1] == self.dim_y
        assert y.shape[1] == self.dim_y
        
        for dim_idx in range(self.dim_y):
            self.model[dim_idx].fit(
                y[:, dim_idx].flatten(), 
                x[:, dim_idx].flatten()
            )
        
        self.trained = True
        
        return None
        
    def recal_sample_quantiles(self, sample_shape):
        """
        Args:
            sample_shape is a tuple of ints, (num_pts, num_samples, dim_y)
        """
        assert len(sample_shape) == 3
        assert sample_shape[2] == self.dim_y
        
        num_pts, num_samples, _ = sample_shape
        input_quantiles = np.random.uniform(size=sample_shape)
        
        output_quantiles = self.inv_call(input_quantiles)
        
        return output_quantiles
    
        
    def inv_call(self, y):
        """
        Args:
            # y: of shape (num_pts, dim_y) or (num_samples, num_pts, dim_y) 
            y: of shape (num_pts, dim_y) or (num_pts, num_samples, dim_y) 
        """
        if not self.trained:
            raise RuntimeError('Recalibrator is not trained')
            
        if len(y.shape) == 2:
            assert y.shape[1] == self.dim_y
            num_samples = 1
            num_pts = y.shape[0]
            y = np.expand_dims(y, axis=1)
        elif len(y.shape) == 3:
            assert y.shape[2] == self.dim_y
            num_pts = y.shape[0]
            num_samples = y.shape[1]
        
        pred_by_dim = [
            self.model[dim_idx].predict(
                y[:, :, dim_idx].flatten()
            ).reshape(num_pts, num_samples) 
            for dim_idx in range(self.dim_y)
        ]
        
        pred = np.transpose(np.stack(pred_by_dim), (1, 2, 0))
        assert pred.shape == (num_pts, num_samples, self.dim_y)
        if num_samples==1:
            pred = np.squeeze(pred)
        
        return pred

    def call(self, x):
        raise RuntimeError("Not allowed to call isotonic recalibrator, try 'inv_cal'")

    def produce_recal_samples_from_mean_std(
            self,
            means,
            stds,
            num_samples,
            device=None,  #TODO do something about this?
    ):
        num_pts = means.shape[0]
        check_shape(means, (num_pts, self.dim_y))
        check_shape(stds, (num_pts, self.dim_y))

        recal_sample_shape = (num_pts, num_samples, self.dim_y)
        recal_quantiles = self.recal_sample_quantiles(recal_sample_shape)
        recal_samples = sample_gaussian_with_quantiles(
            dim_y=self.dim_y,
            means=means,
            stds=stds,
            sample_quantiles=recal_quantiles
        )

        return recal_samples

    @property
    def mace(self):
        if not self.base_data_given:
            raise RuntimeError('Base data must have been given to calculate mace')

        # measure miscalibration
        #TODO maybe move this check block up to the constructor
        num_samples = self.ys.shape[0]
        for x in [self.mus, self.stds, self.ys]:
            assert x.shape == (num_samples, self.dim_y)
            
        mace_per_dim = []
        calplot_per_dim = []
        for d in range(self.dim_y):
            std_for_d = np.ones(num_samples) * self.stds[:, d]
            mean_for_d = np.ones(num_samples) * self.mus[:, d]
            mace_for_d = mean_absolute_calibration_error(
                y_pred=mean_for_d,
                y_std=std_for_d,
                y_true=self.ys[:, d],
                vectorized=True,
            )
            mace_per_dim.append(mace_for_d)
            # calplot_for_d = plot_calibration(
            #     y_pred=mean_for_d,
            #     y_std=std_for_d,
            #     y_true=gt_samples[:, d]
            # )
            # calplot_per_dim.append(calplot_for_d)
        
        return mace_per_dim


class torch_single_dim_recalibrator(torch.nn.Module):
    
    def __init__(
        self,
        dim_y,
        activation=True,
    ):
        super(torch_single_dim_recalibrator, self).__init__()
        
        self.dim_y = dim_y
        self.activation = activation
        
        self.trained = False
        self.A = torch.nn.Parameter(torch.empty(1, self.dim_y))
        self.B = torch.nn.Parameter(torch.empty(1, self.dim_y))
        torch.nn.init.uniform_(self.A)
        torch.nn.init.uniform_(self.B)
        
        self.A.requires_grad = True
        self.B.requires_grad = True
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.dim_y
        
        breakpoint()
        
        x = to_torch_tensors(x)
        
        out = x * self.A + self.B
        if self.activation:
            sigm = torch.nn.Sigmoid()
            out = sigm(out)
            
        assert out.shape == x.shape
           
        return out
    
    def train(x, y, epochs, opt_type, lr):
        train_info = train_torch_module(x, y, self, epochs, opt_type, lr)
        
        return train_info
        
    def call(self, x):
        if not self.trained:
            raise RuntimeError('Recalibrator is not trained')
        
        return self.forward(x)
        
    def inv_call(self, y):
        if not self.trained:
            raise RuntimeError('Recalibrator is not trained')
            
        y = to_torch_tensors(y)
        out = y
        if self.activation:
            out = torch.log(y / (1-y))
        
        return (out - self.B) / self.A


def sample_recalibrated_gaussian(means, stds, recalibrator, num_samples=1000):
    """
    Args:
        means: means, (N, dim_y)
        stds: stds, (N, dim_y)
        recalibrator: recalibration function which has method 'inv_call'
        
    """
    
    recal_samples = []

    ps = np.random.uniform(size=[num_samples] + list(means.shape)) 
    # (num_samples, num_pts, dim_y)
    ps = recalibrator.inv_call(ps) 
    ps = np.clip(ps, 1e-6, 1-1e-6)

    dist = norm(loc=means, scale=stds)
    out = dist.ppf(ps) # (num_samples, num_pts, dim_y)
    out = np.transpose(out, axes=(1, 0, 2))
    assert out.shape == (means.shape[0], num_samples, means.shape[1])
    # out shape is (num_pts, num_samples, dim_y)

    return out


def marginal_sd_recal(
    data_pred_dict,
    recal_mode='ermon',
    use_alt_rv=False
):
    """
    Args:
        data_pred_dict: a dictionary, e.g. from the synth_data.py file
        sd_recal_mode: one of ['ermon', 'ut']
    """
    dim_y = data_pred_dict['dim']
    if use_alt_rv:
        assert 'alt_pred_mean' in data_pred_dict
        pred_mean_key = 'alt_pred_mean'
        pred_std_key = 'alt_pred_std'
        pred_cov_key = 'alt_pred_cov'
    else:
        pred_mean_key = 'pred_mean'
        pred_std_key = 'pred_std'
        pred_cov_key = 'pred_cov'

    pred_mean = np.array(data_pred_dict[pred_mean_key]).reshape(1, dim_y)
    if dim_y == 1:
        pred_std = np.array(data_pred_dict[pred_std_key]).reshape(1, dim_y)
    else:
        pred_std = np.sqrt(np.diag(data_pred_dict[pred_cov_key])).reshape(1, dim_y)

    gt_samples = data_pred_dict['gt'].reshape(-1, dim_y)
    num_samples = gt_samples.shape[0]

    recal_tr_x, recal_tr_y = cal_by_dim(
        mus=np.ones_like(gt_samples) * pred_mean,
        stds=np.ones_like(gt_samples) * pred_std,
        ys=gt_samples,
        mode=recal_mode
    )

    recal_obj = iso_single_dim_recalibrator(dim_y=dim_y)
    recal_obj.train(x=recal_tr_x, y=recal_tr_y)

    # measure miscalibration
    mace_per_dim = []
    calplot_per_dim = []
    for d in range(dim_y):
        std_for_d = np.ones(num_samples) * pred_std[:, d]
        mean_for_d = np.ones(num_samples) * pred_mean[:, d]
        mace_for_d = mean_absolute_calibration_error(
            y_pred=mean_for_d,
            y_std=std_for_d,
            y_true=gt_samples[:, d]
        )
        mace_per_dim.append(mace_for_d)
        # calplot_for_d = plot_calibration(
        #     y_pred=mean_for_d,
        #     y_std=std_for_d,
        #     y_true=gt_samples[:, d]
        # )
        # calplot_per_dim.append(calplot_for_d)

    info = {
        'recal_tr_x': recal_tr_x,
        'recal_tr_y': recal_tr_y,
        'mace_per_dim': mace_per_dim,
        'calplot_per_dim': calplot_per_dim,
    }

    return recal_obj, info
    

    

