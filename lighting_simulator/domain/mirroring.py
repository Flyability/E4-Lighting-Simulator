"""XZ-plane mirroring of LED panel / LED config dicts."""

from .geometry import mirror_xz_vecs
from .guides import mirror_guide_xz


def mirror_group_config_xz(cfg):
    """Return an XZ-mirrored copy of a custom-group LED config dict.

    Dynamic configs carry absolute world-space led_positions/directions, so
    reflecting them is a pure data transform. For standard (non-dynamic)
    groups the extrinsic R = Rz@Ry@Rx convention gives M R M =
    Rz(-yaw) Ry(pitch) Rx(-roll), hence the negated X/Z angles.
    """
    m = dict(cfg)
    m['owner'] = None  # derived copy: not clickable/selectable
    pos = cfg.get('position')
    if pos is not None:
        m['position'] = (pos[0], -pos[1], pos[2])
    m['rotation_x'] = -cfg.get('rotation_x', 0.0)
    m['rotation_z'] = -cfg.get('rotation_z', 0.0)
    if cfg.get('led_positions'):
        m['led_positions'] = mirror_xz_vecs(cfg['led_positions'])
    if cfg.get('led_rotations'):
        m['led_rotations'] = mirror_xz_vecs(cfg['led_rotations'])
    if cfg.get('led_row_directions'):
        m['led_row_directions'] = mirror_xz_vecs(cfg['led_row_directions'])
    if cfg.get('led_beam_tilts'):
        m['led_beam_tilts'] = [-float(t) for t in cfg['led_beam_tilts']]
    if cfg.get('led_states') is not None:
        m['led_states'] = list(cfg['led_states'])
    if cfg.get('guide'):
        m['guide'] = mirror_guide_xz(cfg['guide'])
    return m


def mirror_led_config_xz(cfg):
    """Return an XZ-mirrored copy of an individual-LED config dict.

    Direction convention d = Rx(rx)@Ry(ry)@Rz(rz)@[1,0,0]: reflecting across
    XZ negates rx and rz (keeps ry), and flips roll/beam-tilt handedness.
    """
    m = dict(cfg)
    m['owner'] = None
    m['pos_y'] = -cfg.get('pos_y', 0.0)
    m['rot_x'] = -cfg.get('rot_x', 0.0)
    m['rot_z'] = -cfg.get('rot_z', 0.0)
    m['square_roll'] = -cfg.get('square_roll', 0.0)
    m['beam_tilt'] = -cfg.get('beam_tilt', 0.0)
    return m


def expand_mirror_configs(custom_groups_configs, individual_leds_configs, primary_owner):
    """Append XZ-mirrored copies of every config owned by ``primary_owner`` (in place)."""
    if primary_owner is None:
        return
    for cfg in list(custom_groups_configs):
        if cfg.get('owner') == primary_owner:
            custom_groups_configs.append(mirror_group_config_xz(cfg))
    for cfg in list(individual_leds_configs):
        if cfg.get('owner') == primary_owner:
            individual_leds_configs.append(mirror_led_config_xz(cfg))
