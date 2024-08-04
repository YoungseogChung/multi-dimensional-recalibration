import numpy as np
import tqdm
import timeit

from multiprocessing import (
    RawArray,
    Pool,
)


def sample_based_mace_by_dim(samples, y, return_levels=False):
    """

    Args:
        samples: samples from predictive distribution,
          shape (num_pts, num_samples, dim_y)
        y: data from gt distribution
          shape (num_pts, dim_y)

    Returns: list of mace levels by dim, and optionally list of levels as well


    """
    num_pts, num_samples, dim_y = samples.shape
    assert y.shape == (num_pts, dim_y)

    by_dim_mace_list = []
    by_dim_levels_list = []
    for dim_idx in range(dim_y):
        dim_samples = samples[:, :, dim_idx]  # (num_pts, num_samples)
        dim_y = y[:, [dim_idx]]  # (num_pts, 1)
        dim_emp_quantiles = np.mean(dim_samples <= dim_y, axis=1)
        exp_props = np.linspace(0, 1, dim_emp_quantiles.shape[0])
        obs_props = np.sort(dim_emp_quantiles.flatten())
        by_dim_levels_list.append(obs_props)
        dim_sample_based_mace = np.mean(np.abs(exp_props - obs_props))
        by_dim_mace_list.append(dim_sample_based_mace)

    if return_levels:
        info = {"levels": by_dim_levels_list}
        return by_dim_mace_list, info

    return by_dim_mace_list


def energy_score(samples: np.ndarray, y: np.ndarray, beta: float,
                 rand_set_splits: bool = False, verbose: bool = False):
    """Energy score.

    Args:
        samples: samples from predictive distribution,
          shape (num_pts, num_samples, dim_y)
        y: data from gt distribution
          shape (num_pts, dim_y)
        beta: float in (0, 2)
        verbose: True to print for debugging purposes

    Returns:
        single scalar for energy score

    """
    num_pts, num_samples, dim_y = samples.shape
    assert y.shape == (num_pts, dim_y)

    # check that beta in (0, 2)
    assert (0 < beta) and (beta < 2)

    if verbose:
        for i in range(num_pts):
            print(f'samples at pt_{i}: {samples[i]}')
    # first, split samples into 2 groups
    set_1_size = set_2_size = num_samples // 2
    if rand_set_splits:
        idx_order = np.random.permutation(num_samples)
    else:
        idx_order = np.arange(num_samples)
    set_1_idxs = idx_order[: set_1_size]
    set_2_idxs = idx_order[set_1_size: set_1_size + set_2_size]
    set_1_samples = samples[:, set_1_idxs, :]
    set_2_samples = samples[:, set_2_idxs, :]
    assert set_1_samples.shape == (num_pts, set_1_size, dim_y)
    assert set_2_samples.shape == (num_pts, set_2_size, dim_y)

    if verbose:
        print(f'set_1_samples: {set_1_samples}')
        print(f'set_2_samples: {set_2_samples}')

    score_per_pt = []
    for pt_idx in tqdm.tqdm(range(num_pts)):
        curr_set_1 = set_1_samples[pt_idx]
        curr_set_2 = set_2_samples[pt_idx]
        curr_y = y[pt_idx]
        if verbose:
            print(f'curr_set_1: {curr_set_1}')
            print(f'curr_set_2: {curr_set_2}')
            print(f'curr_y: {curr_y}')

        assert curr_y.shape == (dim_y,)
        term_1_list = []
        term_2_list = []
        for s_1_sample_idx in range(set_1_size):
            curr_set_1_sample = curr_set_1[s_1_sample_idx]
            assert curr_set_1_sample.shape == (dim_y,)
            if verbose:
                print(f'curr_set_1_sample: {curr_set_1_sample}')
            for s_2_sample_idx in range(set_2_size):
                curr_set_2_sample = curr_set_2[s_2_sample_idx]
                assert curr_set_2_sample.shape == (dim_y,)
                if verbose:
                    print(f'  curr_set_2_sample: {curr_set_2_sample}')

                # term_1
                term_1 = np.linalg.norm(curr_set_1_sample - curr_set_2_sample,
                                        ord=2) ** (beta)
                diff = curr_set_1_sample - curr_set_2_sample
                np.testing.assert_almost_equal(term_1,
                                               np.sqrt(np.sum(diff ** 2)) ** beta)
                term_1_list.append(term_1)
                if verbose:
                    print(f'  term_1: {term_1}')
            # using set 1 to calculate the second term
            term_2 = np.linalg.norm(curr_set_1_sample - curr_y, ord=2) ** (
                beta)
            term_2_list.append(term_2)
            if verbose:
                print(f'term_2: {term_2}')

        assert len(term_1_list) == set_1_size * set_2_size
        assert len(term_2_list) == set_1_size
        # energy score at curent pt_idx
        curr_pt_score = (-1 / 2) * np.mean(term_1_list) + np.mean(term_2_list)
        score_per_pt.append(curr_pt_score)

    assert len(score_per_pt) == num_pts

    score = np.mean(score_per_pt)

    return score


var_dict = {}


def init_worker(set_1_Array, set_2_Array, y_Array, set_1_shape, set_2_shape, y_shape):
    var_dict['set_1_Array'] = set_1_Array
    var_dict['set_2_Array'] = set_2_Array
    var_dict['y_Array'] = y_Array
    var_dict['set_1_shape'] = set_1_shape
    var_dict['set_2_shape'] = set_2_shape
    var_dict['y_shape'] = y_shape


def worker_func(pt_idx, beta, dim_y):
    set_1_samples = np.frombuffer(var_dict['set_1_Array']).reshape(
        var_dict['set_1_shape'])
    set_2_samples = np.frombuffer(var_dict['set_2_Array']).reshape(
        var_dict['set_2_shape'])
    y = np.frombuffer(var_dict['y_Array']).reshape(var_dict['y_shape'])

    curr_set_1 = set_1_samples[pt_idx]  # (set_1_size, dim_y)
    curr_set_2 = set_2_samples[pt_idx]  # (set_2_size, dim_y)
    curr_y = y[pt_idx]  # (dim_y,)

    assert curr_y.shape == (dim_y,)
    set_samples_diff = curr_set_1[:, np.newaxis, :] - curr_set_2[np.newaxis, :, :]
    term_1_list = np.sqrt(np.sum(set_samples_diff ** 2, axis=2)) ** beta
    set_1_y_diff = curr_set_1 - curr_y[np.newaxis, :]
    term_2_list = np.sqrt(np.sum(set_1_y_diff ** 2, axis=1)) ** beta

    curr_pt_score = (-1 / 2) * np.mean(term_1_list) + np.mean(term_2_list)
    return curr_pt_score


def energy_score_vectorized(samples: np.ndarray, y: np.ndarray, beta: float,
                            rand_set_splits: bool = False, verbose: bool = False,
                            num_processes: int = 4):
    """Energy score.

    Args:
        samples: samples from predictive distribution,
          shape (num_pts, num_samples, dim_y)
        y: data from gt distribution
          shape (num_pts, dim_y)
        beta: float in (0, 2)
        verbose: True to print for debugging purposes

    Returns:
        single scalar for energy score

    """
    num_pts, num_samples, dim_y = samples.shape
    assert y.shape == (num_pts, dim_y)

    # check that beta in (0, 2)
    assert (0 < beta) and (beta < 2)

    if verbose:
        for i in range(num_pts):
            print(f'samples at pt_{i}: {samples[i]}')
    # first, split samples into 2 groups
    set_1_size = set_2_size = num_samples // 2
    if rand_set_splits:
        idx_order = np.random.permutation(num_samples)
    else:
        idx_order = np.arange(num_samples)
    set_1_idxs = idx_order[: set_1_size]
    set_2_idxs = idx_order[set_1_size: set_1_size + set_2_size]
    set_1_samples = samples[:, set_1_idxs, :]
    set_2_samples = samples[:, set_2_idxs, :]
    assert set_1_samples.shape == (num_pts, set_1_size, dim_y)
    assert set_2_samples.shape == (num_pts, set_2_size, dim_y)

    if verbose:
        print(f'set_1_samples: {set_1_samples}')
        print(f'set_2_samples: {set_2_samples}')

    set_1_Array = RawArray('d', set_1_samples.size)
    set_1_shape = set_1_samples.shape
    set_1_Array_np = np.frombuffer(set_1_Array, dtype=float).reshape(set_1_shape)
    np.copyto(set_1_Array_np, set_1_samples)

    set_2_Array = RawArray('d', set_2_samples.size)
    set_2_shape = set_2_samples.shape
    set_2_Array_np = np.frombuffer(set_2_Array, dtype=float).reshape(set_2_shape)
    np.copyto(set_2_Array_np, set_2_samples)

    y_Array = RawArray('d', y.size)
    y_shape = y.shape
    y_Array_np = np.frombuffer(y_Array, dtype=float).reshape(y_shape)
    np.copyto(y_Array_np, y)

    print("Computing energy score with multiprocessing...")
    t_0 = timeit.default_timer()
    with Pool(processes=num_processes, initializer=init_worker,
              initargs=(set_1_Array, set_2_Array, y_Array, set_1_shape, set_2_shape,
                        y_shape)) as pool:
        result = pool.starmap(
            worker_func,
            [(pt_idx, beta, dim_y) for pt_idx in range(num_pts)]
        )
        assert len(result) == num_pts

        t_1 = timeit.default_timer()
        # print(f"  took {(t_1 - t_0)}s")

        score = np.mean(result)

        return score

