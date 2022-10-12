import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from typing import Optional, Union, Iterable
import pandas as pd

from ..photon_source import PhotonSource, PhotonSourceType
from functools import partial
from time import time


def bucket_fn(bucket_size, use_bayesian_blocks=True):
    """
    Decorator for builing a bucket function.

    Takes a function f(x, *args) and pads the first dimension of x
    to sime discrete bucket size.
    """

    class _decorator:
        def __init__(self, fn, use_bayesian_blocks):
            self._fn = fn

            self._past_sizes = []
            self._past_times = []
            self._buckets = None
            self._use_bblocks = use_bayesian_blocks
            self._block_calc_at = 50

        def __call__(self, x, *args):

            data_size = x.shape[0]
            if data_size == 0:
                return self._fn(x, *args)

            if self._use_bblocks:
                self._past_sizes.append(data_size)

                # Analyze bucket_size
                if len(self._past_sizes) == self._block_calc_at:
                    """
                    print(
                        f"Analyzing bucket-size. Past time stats {np.median(self._past_times)} / {np.percentile(self._past_times, 90)}"
                    )
                    """
                    ps = np.asarray(self._past_sizes)
                    ps = ps[ps > 0]
                    logsize = np.log2(ps)
                    blocks = bayesian_blocks(logsize)
                    if len(blocks) > 3:
                        self._buckets = bayesian_blocks(logsize)
                        # print(f"New buckets: {self._buckets}")
                elif len(self._past_sizes) > self._block_calc_at and (
                    (len(self._past_sizes) % 10) == 0
                ):
                    pass
                    """
                    print(
                        f"Past time stats {np.median(self._past_times[-10:])} / {np.percentile(self._past_times[-10:], 90)}"
                    )
                    print(
                        f"Pre-blocks {np.median(self._past_times[:self._block_calc_at])} / {np.percentile(self._past_times[:self._block_calc_at], 90)}"
                    )
                    """

            if self._buckets is None:
                log_cnt = np.log(data_size) / np.log(bucket_size)
                pad_len = int(np.power(bucket_size, np.ceil(log_cnt)))
            else:
                lds = np.log2(data_size)
                bucket_ix = np.digitize(lds, self._buckets)
                if bucket_ix == 0:
                    log2_pad_len = self._buckets[1]
                elif bucket_ix == len(self._buckets):
                    log2_pad_len = np.ceil(lds)
                else:
                    log2_pad_len = self._buckets[bucket_ix]
                pad_len = int(np.ceil(np.power(2, log2_pad_len)))

            padded = jnp.pad(
                x, [(0, pad_len - data_size)] + [(0, 0)] * (len(x.shape) - 1)
            )

            start = time()
            retval = self._fn(padded, *args)[:data_size]
            self._past_times.append(time() - start)
            return retval

    return partial(_decorator, use_bayesian_blocks=use_bayesian_blocks)


def source_to_model_input_per_module(
    module_coords, source_pos, source_dir, source_t0, c_medium
):
    """
    Convert photon source and module coordinates into neural net input.

    Calculates the distance and viewing angle between the source and the module.
    The viewing angle is the angle of the vector between module and source and the direction
    vector of the source.
    Also calculates the geometric time (expected arrival time for a direct photon).

    Returns the viewing angle and log10(distance) the geometric time.

    """

    source_targ_vec = module_coords - source_pos

    dist = jnp.linalg.norm(source_targ_vec)
    # angles = jnp.arccos(jnp.einsum("ak, k -> a", source_targ_vec, source_dir) / dist)

    angle = jnp.arccos(jnp.sum(source_targ_vec * source_dir) / dist)

    time_geo = dist / c_medium + source_t0

    inp_pars = jnp.asarray([jnp.log10(dist), angle])

    return inp_pars, time_geo


# Vectorize across modules
source_to_model_input = vmap(
    source_to_model_input_per_module, in_axes=(0, None, None, None, None)
)

# Vectorize across sources and jit
sources_to_model_input = jit(vmap(source_to_model_input, in_axes=(None, 0, 0, 0, None)))

# Vectorize across sources and jit
sources_to_model_input_per_module = vmap(
    source_to_model_input_per_module, in_axes=(None, 0, 0, 0, None)
)


def sources_to_array(sources):
    source_pos = np.empty((len(sources), 3))
    source_dir = np.empty((len(sources), 3))
    source_time = np.empty((len(sources), 1))
    source_photons = np.empty((len(sources), 1))

    for i, source in enumerate(sources):
        if source.type != PhotonSourceType.STANDARD_CHERENKOV:
            raise ValueError(
                f"Only Cherenkov-like sources are supported. Got {source.type}."
            )
        source_pos[i] = source.position
        source_dir[i] = source.direction
        source_time[i] = source.time
        source_photons[i] = source.n_photons
    return source_pos, source_dir, source_time, source_photons


def source_array_to_sources(source_pos, source_dir, source_time, source_nphotons):
    sources = []
    for i in range(source_pos.shape[0]):
        source = PhotonSource(
            np.asarray(source_pos[i]),
            np.asarray(source_nphotons[i]),
            np.asarray(source_time[i]),
            np.asarray(source_dir[i]),
        )
        sources.append(source)
    return sources


class Prior(object):
    """Helper class for calculating the prior on the fitness function."""

    def __init__(self, p0: float = 0.05, gamma: Optional[float] = None):
        """
        Args:
            p0: False-positive rate, between 0 and 1.  A lower number places a stricter penalty
              against creating more bin edges, thus reducing the potential for false-positive bin edges. In general,
              the larger the number of bins, the small the p0 should be to prevent the creation of spurious, jagged
              bins. Defaults to 0.05.

            gamma: If specified, then use this gamma to compute the general prior form,
              :math:`p \\sim \\gamma^N`. If gamma is specified, p0 is ignored. Defaults to None.
        """

        self.p0 = p0
        self.gamma = gamma

    def calc(self, N: int) -> float:
        """
        Computes the prior.

        Args:
            N: N-th change point.

        Returns:
            the prior.
        """
        if self.gamma is not None:
            return -np.log(self.gamma)
        else:
            # eq. 21 from Scargle 2012
            return 4 - np.log(73.53 * self.p0 * (N**-0.478))


# From hepstats.modeling.bayesian_blocks
def bayesian_blocks(
    data: Union[Iterable, np.ndarray],
    weights: Union[Iterable, np.ndarray, None] = None,
    p0: float = 0.05,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """Bayesian Blocks Implementation.

    This is a flexible implementation of the Bayesian Blocks algorithm described in :cite:`Scargle_2013`.
    It has been modified to natively accept weighted events, for ease of use in HEP applications.

    Args:
        data: Input dataset values (one dimensional, length N). Repeat values are allowed.

        weights: Weights for dataset (otherwise assume all dataset points have a weight of 1).
          Must be same length as dataset. Defaults to None.

        p0: False-positive rate, between 0 and 1.  A lower number places a stricter penalty
          against creating more bin edges, thus reducing the potential for false-positive bin edges. In general,
          the larger the number of bins, the small the p0 should be to prevent the creation of spurious, jagged
          bins. Defaults to 0.05.

        gamma: If specified, then use this gamma to compute the general prior form,
          :math:`p \\sim \\gamma^N`. If gamma is specified, p0 is ignored. Defaults to None.

    Returns:
         Array containing the (N+1) bin edges

    Examples:
        Unweighted dataset:

        >>> d = np.random.normal(size=100)
        >>> bins = bayesian_blocks(d, p0=0.01)

        Unweighted dataset with repeats:

        >>> d = np.random.normal(size=100)
        >>> d[80:] = d[:20]
        >>> bins = bayesian_blocks(d, p0=0.01)

        Weighted dataset:

        >>> d = np.random.normal(size=100)
        >>> w = np.random.uniform(1,2, size=100)
        >>> bins = bayesian_blocks(d, w, p0=0.01)

    """
    # validate input dataset
    data = np.asarray(data, dtype=float)
    assert data.ndim == 1

    # validate input weights
    if weights is not None:
        weights = np.asarray(weights)
    else:
        # set them to 1 if not given
        weights = np.ones_like(data)

    # initialize the prior
    prior = Prior(p0, gamma)

    # Place dataset and weights into a DataFrame.
    # We want to sort the dataset array (without losing the associated weights), and combine duplicate
    # dataset points by summing their weights together.  We can accomplish all this with `groupby`

    df = pd.DataFrame({"dataset": data, "weights": weights})
    gb = df.groupby("dataset").sum()
    data = gb.index.values
    weights = gb.weights.values

    N = weights.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([data[:1], 0.5 * (data[1:] + data[:-1]), data[-1:]])
    block_length = data[-1] - edges

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    # -----------------------------------------------------------------
    # Start with first dataset cell; add one cell at each iteration
    # -----------------------------------------------------------------
    # last = core_loop(N, block_length, weights, fitfunc, best, last)
    for R in range(N):
        # Compute fit_vec : fitness of putative last block (end at R)

        # T_k: width/duration of each block
        T_k = block_length[: R + 1] - block_length[R + 1]

        # N_k: number of elements in each block
        N_k = np.cumsum(weights[: R + 1][::-1])[::-1]

        # evaluate fitness function
        fit_vec = N_k * (np.log(N_k / T_k))

        # penalize function with prior
        A_R = fit_vec - prior.calc(R + 1)
        A_R[1:] += best[:R]

        i_max = np.argmax(A_R)
        last[R] = i_max
        best[R] = A_R[i_max]

    # -----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    # -----------------------------------------------------------------
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]
