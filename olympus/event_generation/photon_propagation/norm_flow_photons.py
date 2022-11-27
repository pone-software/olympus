import pickle
from typing import List

import awkward as ak
import jax
import jax.numpy as jnp
from jax.lax import cond
import numpy as np

from ananke.models.detector import Detector
from ananke.models.event import Hit, SourceRecord, SourceRecordCollection, HitCollection
from hyperion.models.photon_arrival_time_nflow.net import (
    make_counts_net_fn,
    make_shape_conditioner_fn,
    sample_shape_model,
    traf_dist_builder,
    eval_log_prob,
)
from jax import random

from .interface import AbstractPhotonPropagator
from .utils import sources_to_model_input, sources_to_model_input_per_module, bucket_fn


def make_generate_norm_flow_photons(
        shape_model_path, counts_model_path, c_medium, padding_base=4
):
    """
    Sample photon arrival times using a normalizing flow and a counts model.

    Paramaters:
        shape_model_path: str
        counts_model_path: str
        c_medium: float
            Speed of light in medium (ns). Used to calculate
            the direct propagation time between source and module.
            Make sure it matches the value used in generating the PDF.
        padding_base: int
            Logarithmic base used to calculate bucket size when compiling the
            sampling function. bucket_size = padding_base**N

    """
    shape_config, shape_params = pickle.load(open(shape_model_path, "rb"))
    counts_config, counts_params = pickle.load(open(counts_model_path, "rb"))

    shape_conditioner = make_shape_conditioner_fn(
        shape_config["mlp_hidden_size"],
        shape_config["mlp_num_layers"],
        shape_config["flow_num_bins"],
        shape_config["flow_num_layers"],
    )

    dist_builder = traf_dist_builder(
        shape_config["flow_num_layers"],
        (shape_config["flow_rmin"], shape_config["flow_rmax"]),
        return_base=True,
    )

    counts_net = make_counts_net_fn(counts_config)

    @bucket_fn(bucket_size=8)
    @jax.jit
    def apply_fn(x):
        return shape_conditioner.apply(shape_params, x)

    @bucket_fn(bucket_size=8)
    @jax.jit
    def counts_net_apply_fn(x):
        return counts_net.apply(counts_params, x)

    @bucket_fn(bucket_size=padding_base)
    @jax.jit
    def sample_model(traf_params, key):
        return sample_shape_model(dist_builder, traf_params, traf_params.shape[0], key)

    def generate_norm_flow_photons(
            module_coords,
            module_efficiencies,
            source_pos,
            source_dir,
            source_time,
            source_nphotons,
            seed=31337,
    ):

        # TODO: Reimplement using padding / bucket compile (jax.mask???)

        if isinstance(seed, int):
            key = random.PRNGKey(seed)
        else:
            key = seed

        inp_pars, time_geo = sources_to_model_input(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            c_medium,
        )

        inp_pars = jnp.swapaxes(inp_pars, 0, 1)
        time_geo = jnp.swapaxes(time_geo, 0, 1)

        # flatten [densely pack [modules, sources] in 1D array]
        inp_pars = inp_pars.reshape(
            (source_pos.shape[0] * module_coords.shape[0], inp_pars.shape[-1])
        )
        time_geo = time_geo.reshape(
            (source_pos.shape[0] * module_coords.shape[0], time_geo.shape[-1])
        )
        source_photons = jnp.tile(source_nphotons, module_coords.shape[0]).T.ravel()
        mod_eff_factor = jnp.repeat(module_efficiencies, source_pos.shape[0])

        # Normalizing flows only built up to 300
        # TODO: Check lower bound as well
        distance_mask = inp_pars[:, 0] < np.log10(300)

        inp_params_masked = inp_pars[distance_mask]
        time_geo_masked = time_geo[distance_mask]
        source_photons_masked = source_photons[distance_mask]
        mod_eff_factor_masked = mod_eff_factor[distance_mask]

        # Eval count net to obtain survival fraction
        ph_frac = jnp.power(10, counts_net_apply_fn(inp_params_masked)).squeeze()

        # Sample number of detected photons
        n_photons_masked = ph_frac * source_photons_masked * mod_eff_factor_masked

        key, subkey = random.split(key)
        n_photons_masked = random.poisson(
            subkey, n_photons_masked, shape=n_photons_masked.shape
        ).squeeze()

        if jnp.all(n_photons_masked == 0):
            times = [] * module_coords.shape[0]
            return ak.Array(times)

        # Obtain flow parameters and repeat them for each detected photon
        traf_params = apply_fn(inp_params_masked)
        traf_params_rep = jnp.repeat(traf_params, n_photons_masked, axis=0)
        # Also repeat the geometric time for each detected photon
        time_geo_rep = jnp.repeat(time_geo_masked, n_photons_masked, axis=0).squeeze()

        # Calculate number of photons per module
        # Start with zero array and fill in the poisson samples using distance mask
        n_photons = jnp.zeros(
            source_pos.shape[0] * module_coords.shape[0], dtype=jnp.int32
        )
        n_photons = n_photons.at[distance_mask].set(n_photons_masked)
        n_photons = n_photons.reshape(module_coords.shape[0], source_pos.shape[0])
        n_ph_per_mod = np.sum(n_photons, axis=1)

        # Sample times from flow
        key, subkey = random.split(key)
        samples = sample_model(traf_params_rep, subkey)
        times = np.atleast_1d(np.asarray(samples.squeeze() + time_geo_rep))

        if len(times) == 1:
            ix = np.argwhere(n_ph_per_mod).squeeze()
            times = [[] if i != ix else times for i in range(module_coords.shape[0])]
        else:
            # Split per module and covnert to awkward array
            times = np.split(times, np.cumsum(n_ph_per_mod)[:-1])

        return ak.Array(times)

    return generate_norm_flow_photons


class NormFlowPhotonLHPerModule(object):
    def __init__(self, shape_model_path, counts_model_path, noise_window_len, c_medium):
        shape_config, self._shape_params = pickle.load(open(shape_model_path, "rb"))
        counts_config, self._counts_params = pickle.load(open(counts_model_path, "rb"))

        self._shape_conditioner_fn = make_shape_conditioner_fn(
            shape_config["mlp_hidden_size"],
            shape_config["mlp_num_layers"],
            shape_config["flow_num_bins"],
            shape_config["flow_num_layers"],
        )

        self._dist_builder = traf_dist_builder(
            shape_config["flow_num_layers"],
            (shape_config["flow_rmin"], shape_config["flow_rmax"]),
        )

        self._counts_net_fn = make_counts_net_fn(counts_config)
        self._c_medium = c_medium
        self._noise_window_len = noise_window_len

    def _get_shape_net_params(self, x):
        return self._shape_conditioner_fn.apply(self._shape_params, x)

    def _eval_shape_log_prob(self, net_inp_params, x):
        traf_params = self._get_shape_net_params(net_inp_params)
        traf_params = traf_params.reshape(
            (net_inp_params.shape[0], 1, traf_params.shape[-1])
        )
        return eval_log_prob(self._dist_builder, traf_params, x)

    def _eval_counts_net(self, x):
        return self._counts_net_fn.apply(self._counts_params, x)

    def expected_photons(self, module_efficiency, net_inp_params, source_photons):
        ph_frac = jnp.power(10, self._eval_counts_net(net_inp_params)).reshape(
            source_photons.shape[0]
        )

        n_photons = jnp.reshape(
            ph_frac * source_photons.squeeze(), (source_photons.shape[0],)
        )
        n_ph_pred_per_source = n_photons * module_efficiency

        return n_ph_pred_per_source

    def expected_noise_photons(self, module_noise_rate):
        return module_noise_rate * self._noise_window_len

    def expected_photons_for_sources(
            self,
            module_coords,
            module_efficiency,
            source_pos,
            source_dir,
            source_time,
            source_photons,
    ):
        inp_pars, _ = sources_to_model_input_per_module(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            self._c_medium,
        )
        return self.expected_photons(module_efficiency, inp_pars, source_photons)

    def per_module_shape_llh(
            self, t_res, net_inp_pars, src_photons_pred, noise_photons_pred
    ):
        n_photons_pred = src_photons_pred + noise_photons_pred

        source_weight = n_photons_pred / jnp.sum(n_photons_pred)

        distance_mask = (net_inp_pars[..., 0] < np.log10(300))[:, np.newaxis]

        finite_times = jnp.isfinite(t_res)
        physical = t_res > -4

        mask = distance_mask & finite_times & physical

        # Sanitize likelihood evaluation to avoid nans.
        sanitized_times = jnp.where(mask, t_res, jnp.zeros_like(t_res))
        shape_llh = self._eval_shape_log_prob(net_inp_pars, sanitized_times)

        # Mask the scale factor. This will remove unwanted source-time pairs from the logsumexp
        scale_factor = source_weight[:, np.newaxis] * mask  # + 1e-15

        # Sanitize scale factor. Rows will all zeros lead to nan in logsumexp.
        all_zeros = jnp.all(~mask, axis=0)
        scale_factor += jnp.full(all_zeros.shape[0], 1e-15) * all_zeros

        # logsumexp is log( sum_i b_i * (exp (a_i)))
        shape_llh = jax.scipy.special.logsumexp(shape_llh, b=scale_factor, axis=0)

        return shape_llh

    def per_module_shape_lh_with_noise(
            self,
            t_res,
            net_inp_params,
            src_photons_pred,
            noise_photons_pred,
    ):
        shape_lh = self.per_module_shape_llh(
            t_res, net_inp_params, src_photons_pred, noise_photons_pred
        )
        noise_lh = -jnp.log(self._noise_window_len)

        src_photons_sum = src_photons_pred.sum()

        n_photons_pred = src_photons_sum + noise_photons_pred

        total_shape_lh = jnp.logaddexp(
            noise_lh + jnp.log(noise_photons_pred / n_photons_pred),
            shape_lh + jnp.log(src_photons_sum / n_photons_pred),
        )

        return total_shape_lh

    def per_module_shape_lh_with_noise_for_sources(
            self,
            times,
            module_coords,
            source_pos,
            source_dir,
            source_time,
            src_photons_pred,
            noise_photons_pred,
    ):
        net_inp_params, t_geo = sources_to_model_input_per_module(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            self._c_medium,
        )

        t_res = times - t_geo

        return self.per_module_shape_lh_with_noise(
            t_res, net_inp_params, src_photons_pred, noise_photons_pred
        )

    @staticmethod
    def poisson_llh(predicted, measured):
        return -predicted + measured * jnp.log(predicted)

    def per_module_poisson_llh(
            self,
            module_noise_rate,
            module_efficiency,
            n_measured,
            source_photons,
            net_inp_pars,
    ):
        n_ph_pred_per_mod = self.expected_photons(
            module_efficiency, net_inp_pars, source_photons
        ).sum()

        n_p_pred_noise = self.expected_noise_photons(module_noise_rate)

        return self.poisson_llh(n_ph_pred_per_mod + n_p_pred_noise, n_measured)

    def per_module_poisson_llh_for_sources(
            self,
            n_measured,
            module_coords,
            module_noise_rate,
            module_efficiency,
            source_pos,
            source_dir,
            source_time,
            source_photons,
    ):
        net_inp_pars, _ = sources_to_model_input_per_module(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            self._c_medium,
        )

        return self.per_module_poisson_llh(
            module_noise_rate,
            module_efficiency,
            n_measured,
            source_photons,
            net_inp_pars,
        )

    def per_module_full_llh(
            self,
            times,
            counts,
            source_pos,
            source_dir,
            source_time,
            source_photons,
            module_coords,
            module_noise_rate,
            module_efficiency,
    ):
        net_inp_pars, t_geo = sources_to_model_input_per_module(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            self._c_medium,
        )

        t_res = times - t_geo

        src_photons_pred = self.expected_photons(
            module_efficiency, net_inp_pars, source_photons
        )
        noise_photons_pred = self.expected_noise_photons(module_noise_rate)

        n_photons_total = src_photons_pred.sum() + noise_photons_pred

        shape_llh = self.per_module_shape_lh_with_noise(
            t_res, net_inp_pars, src_photons_pred, noise_photons_pred
        )
        poisson_llh = self.poisson_llh(n_photons_total, counts)

        return shape_llh, poisson_llh

    def per_module_full_llh_sum(self, *args):
        shape, poisson = self.per_module_full_llh(*args)
        return shape.sum() + poisson

    def per_module_tfirst_llh(
            self,
            time,
            counts,
            source_pos,
            source_dir,
            source_time,
            source_photons,
            module_coords,
            module_noise_rate,
            module_efficiency,
    ):
        net_inp_pars, t_geo = sources_to_model_input_per_module(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            self._c_medium,
        )

        tfirst = time

        window_start = 0
        tvanilla = jnp.linspace(window_start, tfirst, 1000).squeeze()
        # tsamples = tvanilla / 5000 * (tfirst + 1000) - 1000
        tsamples = tvanilla - t_geo

        src_photons_pred = self.expected_photons(
            module_efficiency, net_inp_pars, source_photons
        )
        noise_photons_pred = self.expected_noise_photons(module_noise_rate)

        n_photons_total = src_photons_pred.sum() + noise_photons_pred

        shape_llh = self.per_module_shape_lh_with_noise(
            tsamples, net_inp_pars, src_photons_pred, noise_photons_pred
        )

        poisson_llh = self.poisson_llh(n_photons_total, counts)

        cumul = jnp.trapz(jnp.exp(shape_llh), x=tvanilla)

        llh = (
                jnp.log(counts)
                + self.per_module_shape_lh_with_noise(
            (jnp.atleast_1d(tfirst) - t_geo),
            net_inp_pars,
            src_photons_pred,
            noise_photons_pred,
        )
                + jnp.log(1 - cumul) * (counts - 1)
        )

        return llh, poisson_llh


def make_nflow_photon_likelihood_per_module(
        shape_model_path, counts_model_path, mode="full", noise_window_len=5000
):
    shape_config, shape_params = pickle.load(open(shape_model_path, "rb"))
    counts_config, counts_params = pickle.load(open(counts_model_path, "rb"))

    shape_conditioner = make_shape_conditioner_fn(
        shape_config["mlp_hidden_size"],
        shape_config["mlp_num_layers"],
        shape_config["flow_num_bins"],
        shape_config["flow_num_layers"],
    )

    def apply_fn(params, x):
        return shape_conditioner.apply(params, x)

    dist_builder = traf_dist_builder(
        shape_config["flow_num_layers"],
        (shape_config["flow_rmin"], shape_config["flow_rmax"]),
    )

    counts_net = make_counts_net_fn(counts_config)

    def counts_net_apply_fn(params, x):
        return counts_net.apply(params, x)

    def eval_l_p(traf_params, samples):
        return eval_log_prob(dist_builder, traf_params, samples)

    def per_module_shape_lh(t_res, inp_pars, source_weight):
        traf_params = apply_fn(shape_params, inp_pars)
        traf_params = traf_params.reshape((inp_pars.shape[0], traf_params.shape[-1]))

        distance_mask = (inp_pars[..., 0] < np.log10(300))[:, np.newaxis]
        finite_times = jnp.isfinite(t_res)
        physical = t_res > -4

        mask = distance_mask & finite_times & physical

        traf_params = traf_params.reshape(
            (traf_params.shape[0], 1, traf_params.shape[1])
        )

        # Sanitize likelihood evaluation to avoid nans.
        sanitized_times = jnp.where(mask, t_res, jnp.zeros_like(t_res))
        shape_lh = eval_l_p(traf_params, sanitized_times)

        # Mask the scale factor. This will remove unwanted source-time pairs from the logsumexp
        scale_factor = source_weight[:, np.newaxis] * mask + 1e-15

        # logsumexp is log( sum_i b_i * (exp (a_i)))
        shape_lh = jax.scipy.special.logsumexp(shape_lh, b=scale_factor, axis=0)

        return shape_lh

    def eval_per_module_likelihood(
            time,
            n_measured,
            module_coords,
            module_efficiencies,
            source_pos,
            source_dir,
            source_time,
            source_photons,
            c_medium,
            noise_rate,
    ):

        inp_pars, time_geo = sources_to_model_input_per_module(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            c_medium,
        )

        inp_pars = inp_pars.reshape((source_pos.shape[0], inp_pars.shape[-1]))

        ph_frac = jnp.power(10, counts_net_apply_fn(counts_params, inp_pars)).reshape(
            source_pos.shape[0]
        )

        noise_photons = noise_rate * noise_window_len

        n_photons = (
                jnp.reshape(ph_frac * source_photons.squeeze(), (source_pos.shape[0],))
                * module_efficiencies
        )

        n_ph_pred_per_mod = jnp.sum(n_photons)
        n_ph_pred_per_mod_total = n_ph_pred_per_mod + noise_photons

        counts_lh = jnp.sum(
            -n_ph_pred_per_mod_total + n_measured * jnp.log(n_ph_pred_per_mod_total)
        )

        if mode == "counts":
            return counts_lh

        time_geo = time_geo.reshape((source_pos.shape[0], time_geo.shape[-1]))
        t_res = time - time_geo

        def total_shape_lh(t_res):
            source_weight = n_photons / jnp.sum(n_photons)

            shape_lh = per_module_shape_lh(t_res, inp_pars, source_weight)
            noise_lh = -jnp.log(noise_window_len)

            total_shape_lh = jnp.logaddexp(
                noise_lh + jnp.log(noise_photons / n_ph_pred_per_mod_total),
                shape_lh + jnp.log(n_ph_pred_per_mod / n_ph_pred_per_mod_total),
            )

            total_shape_lh = total_shape_lh

            return total_shape_lh

        if mode == "full":
            return total_shape_lh(t_res), counts_lh, n_ph_pred_per_mod_total

        elif mode == "tfirst":
            tfirst = time
            tvanilla = jnp.linspace(-1000, tfirst, 4000)
            # tsamples = tvanilla / 5000 * (tfirst + 1000) - 1000
            tsamples = tvanilla - time_geo

            cumul = jnp.trapz(jnp.exp(total_shape_lh(tsamples)), x=tvanilla)

            llh = (
                    jnp.log(n_measured)
                    + total_shape_lh(tfirst - time_geo)
                    + jnp.log(1 - cumul) * n_measured
            )

            return llh.sum(), counts_lh, n_ph_pred_per_mod_total

    return eval_per_module_likelihood


def make_nflow_photon_likelihood(shape_model_path, counts_model_path):
    raise RuntimeError("Add noise")

    shape_config, shape_params = pickle.load(open(shape_model_path, "rb"))
    counts_config, counts_params = pickle.load(open(counts_model_path, "rb"))

    shape_conditioner = make_shape_conditioner_fn(
        shape_config["mlp_hidden_size"],
        shape_config["mlp_num_layers"],
        shape_config["flow_num_bins"],
        shape_config["flow_num_layers"],
    )

    @jax.jit
    def apply_fn(params, x):
        return shape_conditioner.apply(params, x)

    dist_builder = traf_dist_builder(
        shape_config["flow_num_layers"],
        (shape_config["flow_rmin"], shape_config["flow_rmax"]),
    )

    counts_net = make_counts_net_fn(counts_config)

    @jax.jit
    def counts_net_apply_fn(params, x):
        return counts_net.apply(params, x)

    @jax.jit
    def eval_l_p(traf_params, samples):
        return eval_log_prob(dist_builder, traf_params, samples)

    def eval_likelihood(
            event,
            module_coords,
            source_pos,
            source_dir,
            source_time,
            source_photons,
            c_medium,
    ):
        inp_pars, time_geo = sources_to_model_input(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            c_medium,
        )

        distance_mask = inp_pars[..., 0] < np.log10(300)
        inp_pars = inp_pars.reshape(
            (source_pos.shape[0] * module_coords.shape[0], inp_pars.shape[-1])
        )

        traf_params = apply_fn(shape_params, inp_pars)
        traf_params = traf_params.reshape(
            (source_pos.shape[0], module_coords.shape[0], traf_params.shape[-1])
        )

        hits_per_mod = jnp.asarray(ak.count(event, axis=1))

        flat_ev = jnp.asarray(ak.ravel(event))
        traf_params_rep = jnp.repeat(traf_params, hits_per_mod, axis=1)
        time_geo_rep = jnp.repeat(time_geo, hits_per_mod, axis=1).squeeze()
        distance_mask_rep = jnp.repeat(distance_mask, hits_per_mod, axis=1)

        t_res = flat_ev - time_geo_rep

        mask = distance_mask_rep & (t_res >= -4)
        shape_lh = jnp.where(
            mask, eval_l_p(traf_params_rep, t_res), jnp.zeros_like(distance_mask_rep)
        )

        ph_frac = jnp.power(10, counts_net_apply_fn(counts_params, inp_pars)).reshape(
            source_pos.shape[0], module_coords.shape[0]
        )

        n_photons = ph_frac * source_photons
        n_ph_pred_per_mod = jnp.sum(n_photons, axis=0)

        counts_lh = -n_ph_pred_per_mod + hits_per_mod * jnp.log(n_ph_pred_per_mod)

        return shape_lh.sum() + counts_lh.sum()

        lhsum = 0
        for imod in range(module_coords.shape[0]):
            if ak.count(event[imod]) == 0:
                continue

            dist_pars = traf_params[:, imod]
            mask = distance_mask[:, imod]

            if jnp.all(~mask):
                continue
            masked_pars = dist_pars[mask]

            t_res = jnp.asarray(event[imod]) - time_geo[:, imod][mask]

            per_mod_lh = eval_l_p(masked_pars, t_res.T)
            t_res_mask = t_res > -4

            zero_fill = jnp.zeros_like(per_mod_lh)

            lhsum += jnp.sum(jnp.where(t_res_mask.T, per_mod_lh, zero_fill))

            # lhsum += jnp.sum(per_mod_lh[t_res_mask.T])

        return lhsum

    return eval_likelihood


class NormalFlowPhotonPropagator(AbstractPhotonPropagator):
    """Photon propagator based on the normal flow method."""

    def __init__(
            self,
            detector: Detector,
            shape_model_path: str,
            counts_model_path: str,
            c_medium: float,
            padding_base: int = 4
    ):
        """
        Sample photon arrival times using a normalizing flow and a counts model.

        Args:
            shape_model_path: path of the shape model
            counts_model_path: path of the counts model
            c_medium:
                Speed of light in medium (ns). Used to calculate
                the direct propagation time between source and module.
                Make sure it matches the value used in generating the PDF.
            padding_base:
                Logarithmic base used to calculate bucket size when compiling the
                sampling function. bucket_size = padding_base**N

        """
        super().__init__(detector=detector)

        self.generate_norm_flow_photons = make_generate_norm_flow_photons(
            shape_model_path,
            counts_model_path,
            c_medium,
            padding_base
        )

    def propagate(
            self, sources: SourceRecordCollection,
            seed: int = 1337
    ) -> HitCollection:
        """Propagates using the normal flow propagator.

        Args:
            sources: Photon source to propagate
            seed: seed by which to propagate

        Returns:
            List of hits containing propagation result
        """
        source_df = sources.to_pandas()

        if len(source_df):
            hits = self.generate_norm_flow_photons(
                self.detector_df[['module_x', 'module_y', 'module_z', ]].to_numpy(dtype=np.float32),
                self.detector_df['pmt_efficiency'].to_numpy(dtype=np.float32),
                source_df[['location_x', 'location_y', 'location_z']].to_numpy(dtype=np.float32),
                source_df[['orientation_x', 'orientation_y', 'orientation_z']].to_numpy(dtype=np.float32),
                np.expand_dims(source_df['time'].to_numpy(dtype=np.float32), axis=1),
                np.expand_dims(source_df['number_of_photons'].to_numpy(dtype=np.float32), axis=1),
                seed
            )
        else:
            hits = []

        hit_collection = HitCollection()

        for index, module in enumerate(hits):
            for hit in module:
                hit_collection.append(
                    Hit(
                        pmt_id=self.detector_df.loc[index, "pmt_id"],
                        module_id=self.detector_df.loc[index, "module_id"],
                        string_id=self.detector_df.loc[index, "string_id"],
                        time=hit
                    )
                )

        return hit_collection
