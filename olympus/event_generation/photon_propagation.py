"""Implements photon propagation."""
import awkward as ak
import numpy as np
import torch
from hyperion.models.photon_arrival_time.pdf import sample_exp_exp_exp
from numba import float64, jit
from numba.experimental import jitclass

from .constants import Constants

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class PhotonSource_(object):
    """
    A source of photons.

    This class is the pure python prototype for the jitclass `PhotonSource`.

    Parameters:
        pos: float[3]
        amp: float
        t0: float
        dir: float[3]

    """

    pos: float64[:]
    amp: float
    t0: float
    dir: float64[:]

    def __init__(self, pos, amp, t0, dir):
        """Initialize PhotonSource_."""
        self.pos = pos
        self.amp = amp
        self.t0 = t0
        self.dir = dir


PhotonSource = jitclass(PhotonSource_)


def dejit_sources(sources):
    """Convert numba jitclass PhotonSources into pure python objects."""
    return [
        PhotonSource_(source.pos, source.amp, source.t0, source.dir)
        for source in sources
    ]


class BiolumiSource_(object):
    """
    A source of biolumi photons.

    This class is the pure python prototype for the jitclass `BiolumiSource_`.
    """

    pos: float64[:]
    amp: float
    t0: float
    tspread: float

    def __init__(self, pos, amp, t0, tspread):
        """Initialize PhotonSource_."""
        self.pos = pos
        self.amp = amp
        self.t0 = t0
        self.tspread = tspread


BiolumiSource = jitclass(BiolumiSource_)


def dejit_biolumi_sources(sources):
    """Convert numba jitclass PhotonSources into pure python objects."""
    return [
        BiolumiSource(source.pos, source.amp, source.t0, source.tspread)
        for source in sources
    ]


@jit(nopython=True)
def generate_photons(
    module_coords,
    module_efficiencies,
    sources,
    c_vac=Constants.c_vac,
    n_gr=Constants.n_gr,
    pandel_lambda=Constants.pandel_lambda,
    theta_cherenkov=Constants.theta_cherenkov,
    pandel_rho=Constants.pandel_rho,
    photocathode_area=400e-4,
    lambda_abs=Constants.lambda_abs,
    lambda_sca=Constants.lambda_sca,
    seed=31337,
):
    """
    Generate photons for a list of sources.

    The amplitude (== number of photon) at each detector module is modeled as exponential decay based on
    the distance to `pos` with decay constant `d0`. The detection process is modelled as poisson process.
    The photon arrival times are modeled using the
    `Pandel`-PDF (https://www.sciencedirect.com/science/article/pii/S0927650507001260), which is a gamma distribution
    with distance-dependent scaling of the shape parameters.
    """
    all_times_det = []
    np.random.seed(seed)

    lambda_p = np.sqrt(lambda_abs * lambda_sca / 3)
    xi = np.exp(-lambda_sca / lambda_abs)
    lambda_c = lambda_sca / (3 * xi)

    for idom in range(module_coords.shape[0]):

        this_times = []
        total_length = 0
        for source in sources:
            dist = np.linalg.norm(source.pos - module_coords[idom])

            """
      # model photon emission as point-like

      detected_flux = source.amp * np.exp(-dist/d0) / (4*np.pi* dist**2)
      detected_photons = detected_flux * photocathode_area

      # from https://arxiv.org/pdf/1311.4767.pdf
      """
            detected_photons = (
                source.amp
                * photocathode_area
                / (4 * np.pi)
                * np.exp(-dist / lambda_p)
                * 1
                / (lambda_c * dist * np.tanh(dist / lambda_c))
            )

            amp_det = np.random.poisson(detected_photons * module_efficiencies[idom])

            time_geo = dist / (c_vac / n_gr) + source.t0
            pandel_xi = dist / (pandel_lambda * np.sin(theta_cherenkov))

            times_det = (
                np.random.gamma(pandel_xi, scale=1 / pandel_rho, size=amp_det)
                + time_geo
            )
            this_times.append(times_det)
            total_length += amp_det

        this_times_arr = np.empty(total_length)
        i = 0
        for tt in this_times:
            this_times_arr[i : i + tt.shape[0]] = tt  # noqa: E203
            i += tt.shape[0]

        all_times_det.append(this_times_arr)

    return all_times_det


@jit(nopython=True)
def generate_biolumi_photons(
    module_coords,
    module_efficiencies,
    sources,
    c_vac=Constants.c_vac,
    n_gr=Constants.n_gr,
    photocathode_area=400e-4,
    lambda_abs=Constants.lambda_abs,
    lambda_sca=Constants.lambda_sca,
    seed=31337,
):
    """
    Generate photons for a list of biolumi sources.

    The amplitude (== number of photon) at each detector module is modeled as exponential decay based on
    the distance to `pos` with decay constant `d0`. The detection process is modelled as poisson process.

    """
    all_times_det = []
    np.random.seed(seed)

    lambda_p = np.sqrt(lambda_abs * lambda_sca / 3)
    xi = np.exp(-lambda_sca / lambda_abs)
    lambda_c = lambda_sca / (3 * xi)

    for idom in range(module_coords.shape[0]):

        this_times = []
        total_length = 0
        for source in sources:
            dist = np.linalg.norm(source.pos - module_coords[idom])

            """
      # model photon emission as point-like

      detected_flux = source.amp * np.exp(-dist/d0) / (4*np.pi* dist**2)
      detected_photons = detected_flux * photocathode_area

      # from https://arxiv.org/pdf/1311.4767.pdf
      """
            detected_photons = (
                source.amp
                * photocathode_area
                / (4 * np.pi)
                * np.exp(-dist / lambda_p)
                * 1
                / (lambda_c * dist * np.tanh(dist / lambda_c))
            )

            amp_det = np.random.poisson(detected_photons * module_efficiencies[idom])

            time_geo = dist / (c_vac / n_gr) + source.t0

            times_det = np.random.uniform(0, source.tspread, size=amp_det) + time_geo
            this_times.append(times_det)
            total_length += amp_det

        this_times_arr = np.empty(total_length)
        i = 0
        for tt in this_times:
            this_times_arr[i : i + tt.shape[0]] = tt  # noqa: E203
            i += tt.shape[0]

        all_times_det.append(this_times_arr)

    return all_times_det


exp_exp_exp_sampler = jit(sample_exp_exp_exp)


@jit(nopython=True)
def sample_times(pdf_params, sources, module_coords, module_efficiencies, time_geo):

    all_times_det = []
    for idom in range(module_coords.shape[0]):

        this_times = []
        total_length = 0
        for isource in range(len(sources)):
            pars = pdf_params[isource, idom]
            usf = 1 - 10 ** (-pars[5])
            surv_ratio = 10 ** pars[6]
            n_ph_tot = np.random.poisson(
                surv_ratio * sources[isource].amp * module_efficiencies[idom]
            )

            n_direct, n_indirect = np.random.multinomial(n_ph_tot, [usf, 1 - usf])

            all_samples = np.empty(n_ph_tot)

            """
            Can't use this with numba yet
            expon_samples = sampler(*pars[:-2], size=n_indirect, rstate=rstate) + 2
            """
            expon_samples = (
                exp_exp_exp_sampler(
                    pars[0], pars[1], pars[2], pars[3], pars[4], n_indirect
                )
                + 2
            )
            uni_samples = np.random.uniform(0, 2, size=n_direct)

            all_samples[:n_direct] = uni_samples
            all_samples[n_direct:] = expon_samples

            all_samples += time_geo[isource, idom]
            this_times.append(all_samples)
            total_length += n_ph_tot

        this_times_arr = np.empty(total_length)
        i = 0
        for tt in this_times:
            this_times_arr[i : i + tt.shape[0]] = tt  # noqa: E203
            i += tt.shape[0]

        all_times_det.append(this_times_arr)

    return all_times_det


def source_to_model_input(module_coords, sources, c_vac, n_gr):
    """Convert photon sources an module coordinates into neural net input."""
    source_pos = np.empty((len(sources), 3))
    source_amp = np.empty(len(sources))
    source_dir = np.empty((len(sources), 3))
    source_t0 = np.empty((len(sources)))

    for i in range(len(sources)):
        source_pos[i] = sources[i].pos
        source_amp[i] = sources[i].amp
        source_dir[i] = sources[i].dir
        source_t0[i] = sources[i].t0

    source_targ_vec = module_coords[np.newaxis, ...] - source_pos[:, np.newaxis, :]
    dist = np.linalg.norm(source_targ_vec, axis=-1)
    angles = np.arccos(np.einsum("abk, ak -> ab", source_targ_vec, source_dir) / dist)

    time_geo = dist / (c_vac / n_gr) + source_t0[:, np.newaxis]

    inp_pars = torch.stack(
        [
            torch.tensor(angles.ravel(), dtype=torch.float, device=device),
            torch.tensor(np.log10(dist.ravel()), dtype=torch.float, device=device),
        ],
        dim=-1,
    )

    return inp_pars, time_geo


def make_generate_photons_nn(model_path):
    """
    Build an arival time sampling function.

    This function uses a pytorch model to predict the pdf parameters
    of a triple-exponential mixture model fitted to the arrival time distributions.
    """

    model = torch.load(model_path).to(device)
    model.eval()

    def generate_photons_nn(
        module_coords,
        module_efficiencies,
        sources,
        seed=31337,
        c_vac=Constants.c_vac,
        n_gr=Constants.n_gr,
    ):

        all_times_det = []
        np.random.seed(seed)

        inp_pars, time_geo = source_to_model_input(module_coords, sources, c_vac, n_gr)

        pdf_params = (model(inp_pars).cpu().detach().numpy()).reshape(
            [len(sources), module_coords.shape[0], 9]
        )[..., :7]

        all_times_det = sample_times(
            pdf_params, sources, module_coords, module_efficiencies, time_geo
        )

        all_times_det = ak.sort(ak.Array(all_times_det))
        return all_times_det

    return generate_photons_nn


def make_generate_bin_amplitudes_nn(model_path, prediction=False):
    """
    Build a binned arival time sampling function.

    This function uses a pytorch model to predict the bin contents of the arrival time distribution.

    Parameters:
        model_path: str
            Path to pytorch model
        prediction: bool
            If True, return predicted amplitudes instead of poisson samples
    """

    model, binning = torch.load(model_path)
    nbins = len(binning) - 1
    model = model.to(device)
    binning = binning.to(device)
    bin_width = binning[1] - binning[0]
    model.eval()

    def generate_bin_amp_nn(
        module_coords,
        module_efficiencies,
        sources,
        seed=31337,
        c_vac=Constants.c_vac,
        n_gr=Constants.n_gr,
    ):

        np.random.seed(seed)
        rng = torch.Generator(device)
        rng.manual_seed(seed)

        source_amps = torch.tensor(
            [source.amp for source in sources], device=device, dtype=torch.float
        )

        inp_pars, time_geo = source_to_model_input(module_coords, sources, c_vac, n_gr)

        time_geo = torch.tensor(time_geo, device=device)
        pred_frac_log = model(inp_pars).reshape(
            [len(sources), module_coords.shape[0], nbins]
        )

        # We have the predicted amplitudes relative to each source's time.
        # Now construct a binning that covers the full range spanned by all light sources.

        tmin = torch.min(time_geo, axis=0)[0] + binning[0]
        tmax = torch.max(time_geo, axis=0)[0] + binning[-1]

        samples = []

        for imod in range(module_coords.shape[0]):
            new_binning = torch.arange(tmin[imod], tmax[imod] + bin_width, bin_width)
            new_binning_offsets_bws = (
                time_geo[:, imod] + binning[0] - tmin[imod]
            ) / bin_width

            new_binning_offsets = torch.floor(new_binning_offsets_bws).int()
            new_binning_offsets_frac = new_binning_offsets_bws - new_binning_offsets
            # print(new_binning_offsets, new_binning_offsets_frac)
            interpolated = (
                pred_frac_log[:, imod, :-1]
                + torch.diff(pred_frac_log[:, imod, :], axis=1)
                / bin_width
                * new_binning_offsets_frac[..., np.newaxis]
            )

            interpolated_amps = torch.exp(interpolated) * source_amps[:, np.newaxis]

            this_pred_amplitudes = torch.zeros(new_binning.shape[0] - 1, device=device)

            for isource in range(len(sources)):
                this_pred_amplitudes[
                    new_binning_offsets[isource] : new_binning_offsets[isource]
                    + nbins
                    - 1
                ] += interpolated_amps[isource]

            if prediction:
                samples.append((this_pred_amplitudes, new_binning))
            else:
                samples.append((torch.poisson(this_pred_amplitudes, rng), new_binning))

        return (samples, torch.tensor(time_geo, device=device))

    return generate_bin_amp_nn
