"""Event Generators."""
import logging

import numpy as np
from jax import numpy as jnp, random

from olympus.constants import Constants, defaults
from .lightyield import make_pointlike_cascade_source, make_realistic_cascade_source

logger = logging.getLogger(__name__)

_default_rng = defaults['rng']



def generate_muon_energy_losses(
        propagator,
        energy,
        track_len,
        position,
        direction,
        time,
        key,
        loss_resolution=0.2,
        cont_resolution=1
):
    try:
        import proposal as pp
    except ImportError as e:
        logger.critical("Could not import proposal!")
        raise e

    init_state = pp.particle.ParticleState()
    init_state.energy = energy * 1e3  # initial energy in MeV
    init_state.position = pp.Cartesian3D(
        position[0] * 100, position[1] * 100, position[2] * 100
    )
    init_state.direction = pp.Cartesian3D(direction[0], direction[1], direction[2])
    track = propagator.propagate(init_state, track_len * 100)  # cm

    aspos = []
    asdir = []
    astime = []
    asph = []

    loss_map = {
        "brems": 11,
        "epair": 11,
        "hadrons": 211,
        "ioniz": 11,
        "photonuclear": 211,
    }

    # harvest losses
    for loss in track.stochastic_losses():
        # dist = loss.position.z / 100
        e_loss = loss.energy / 1e3

        """
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        
        p = position + dist * direction
        t = dist / Constants.c_vac + time
        """

        p = np.asarray([loss.position.x, loss.position.y, loss.position.z]) / 100
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        t = np.linalg.norm(p - position) / Constants.c_vac + time

        loss_type_name = pp.particle.Interaction_Type(loss.type).name
        ptype = loss_map[loss_type_name]

        if e_loss < 1e3:
            sresult = make_pointlike_cascade_source(
                p, t, dir, e_loss, ptype
            )
            spos, sdir, stime, sph = sresult
        else:
            key, subkey = random.split(key)
            sresult = make_realistic_cascade_source(
                p,
                t,
                dir,
                e_loss,
                ptype,
                subkey,
                resolution=loss_resolution,
                moliere_rand=True
            )
            spos, sdir, stime, sph = sresult

        aspos.append(spos)
        asdir.append(sdir)
        astime.append(stime)
        asph.append(sph)

    # distribute continuous losses uniformly along track
    # TODO: check if thats a good approximation
    # TODO: track segments

    cont_loss_sum = sum([loss.energy for loss in track.continuous_losses()]) / 1e3
    total_dist = track.track_propagated_distances()[-1] / 100
    loss_dists = np.arange(0, total_dist, cont_resolution)
    e_loss = cont_loss_sum / len(loss_dists)

    for ld in loss_dists:
        p = ld * direction + position
        t = np.linalg.norm(p - position) / Constants.c_vac + time

        sresult = make_pointlike_cascade_source(
            p, t, direction, e_loss, 11
        )
        spos, sdir, stime, sph = sresult

        aspos.append(spos)
        asdir.append(sdir)
        astime.append(stime)
        asph.append(sph)

    if not aspos:
        return_value = (None, None, None, None,)
        return_value = return_value + (total_dist, )
        return return_value

    return_value = (
        jnp.concatenate(aspos),
        jnp.concatenate(asdir),
        jnp.concatenate(astime),
        jnp.concatenate(asph),
    )

    return_value = return_value + (total_dist,)

    return return_value

