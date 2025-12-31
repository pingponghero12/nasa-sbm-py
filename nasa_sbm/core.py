"""Core Python API for NASA Standard Breakup Model."""

import numpy as np
import xarray as xr
from datetime import datetime
from typing import Optional, Literal

try:
    from nasa_sbm import _core
except ImportError:
    from . import _core


def explosion(
    mass: float,
    sat_type: Literal["spacecraft", "rocket_body"] = "spacecraft",
    cutoff: float = 0.01,
    seed: Optional[int] = None,
    enforce_mass_conservation: bool = False,
) -> xr.Dataset:
    """Generate debris cloud from satellite explosion."""
    sim = _core.BreakupSimulation()

    sim.run_explosion(
        mass=mass,
        sat_type=sat_type,
        min_lc=cutoff,
        seed=seed,
        enforce_mass_conservation=enforce_mass_conservation
    )

    fragments = sim.get_fragments()
    n_fragments = sim.get_fragment_count()

    delta_v = generate_isotropic_velocities(fragments["velocity"], seed)

    coords = {
        'fragment': np.arange(n_fragments),
        'xyz': ['x', 'y', 'z']
    }

    data_vars = {
        'fragment_size': (['fragment'], fragments['characteristic_length']),
        'area_to_mass_ratio': (['fragment'], fragments['area_to_mass_ratio']),
        'cross_sectional_area': (['fragment'], fragments['area']),
        'fragment_mass': (['fragment'], fragments['mass']),
        'initial_position': (['fragment', 'xyz'], fragments['position']),
        'delta_velocity': (['fragment', 'xyz'], delta_v),
        'parent_velocity': (['fragment', 'xyz'], fragments['velocity']),
    }

    attrs = {
        'title':  'NASA SBM Explosion Debris Cloud',
        'simulation_type': 'explosion',
        'parent_mass_kg':  mass,
        'satellite_type': sat_type,
        'size_cutoff_m': cutoff,
        'n_fragments': n_fragments,
        'creation_time': datetime.utcnow().isoformat(),
        'enforce_mass_conservation':  int(enforce_mass_conservation),
        'random_seed': seed if seed is not None else -1,
    }

    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def collision(
    mass1: float,
    mass2: float,
    velocity:  float,
    sat_type1: Literal["spacecraft", "rocket_body"] = "spacecraft",
    sat_type2: Literal["spacecraft", "rocket_body"] = "spacecraft",
    cutoff: float = 0.01,
    seed: Optional[int] = None,
    enforce_mass_conservation: bool = False,
) -> xr.Dataset:
    """Generate debris cloud from satellite collision."""
    sim = _core.BreakupSimulation()

    sim.run_collision(
        mass1=mass1,
        mass2=mass2,
        velocity=velocity,
        sat_type1=sat_type1,
        sat_type2=sat_type2,
        min_lc=cutoff,
        seed=seed,
        enforce_mass_conservation=enforce_mass_conservation
    )

    fragments = sim.get_fragments()
    n_fragments = sim.get_fragment_count()

    delta_v = generate_isotropic_velocities(fragments["velocity"], seed)

    coords = {
        'fragment': np.arange(n_fragments),
        'xyz': ['x', 'y', 'z']
    }

    data_vars = {
        'fragment_size': (['fragment'], fragments['characteristic_length']),
        'area_to_mass_ratio': (['fragment'], fragments['area_to_mass_ratio']),
        'cross_sectional_area': (['fragment'], fragments['area']),
        'fragment_mass': (['fragment'], fragments['mass']),
        'initial_position': (['fragment', 'xyz'], fragments['position']),
        'delta_velocity':  (['fragment', 'xyz'], delta_v),
        'parent_velocity': (['fragment', 'xyz'], fragments['velocity']),
    }

    attrs = {
        'title': 'NASA SBM Collision Debris Cloud',
        'simulation_type': 'collision',
        'mass1_kg': mass1,
        'mass2_kg': mass2,
        'relative_velocity_km_s': velocity,
        'satellite_type1': sat_type1,
        'satellite_type2': sat_type2,
        'size_cutoff_m': cutoff,
        'n_fragments': n_fragments,
        'creation_time': datetime.utcnow().isoformat(),
        'enforce_mass_conservation': int(enforce_mass_conservation),
        'random_seed': seed if seed is not None else -1,
    }

    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def generate_isotropic_velocities(parent_velocities:  np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Generate isotropic velocity directions on unit sphere."""
    rng = np.random.default_rng(seed)
    n = parent_velocities.shape[0]

    magnitudes = np.linalg.norm(parent_velocities, axis=1)

    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)

    directions = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

    return directions * magnitudes[: , np.newaxis]
