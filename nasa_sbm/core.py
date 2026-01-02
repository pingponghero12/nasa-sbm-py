"""Core Python API for NASA Standard Breakup Model."""

import numpy as np
import xarray as xr
from datetime import datetime
from typing import Optional, Literal, Tuple

try:
    from nasa_sbm import _core
except ImportError:
    from .  import _core


def explosion(
    mass: float,
    sat_type: Literal["spacecraft", "rocket_body"] = "spacecraft",
    cutoff: float = 0.01,
    seed: Optional[int] = None,
    enforce_mass_conservation:  bool = False,
    orbit_altitude: Optional[float] = None,
    position: Optional[Tuple[float, float, float]] = None,
    velocity: Optional[Tuple[float, float, float]] = None,
) -> xr.Dataset:
    """Generate debris cloud from satellite explosion."""
    sim = _core. BreakupSimulation()

    # Prepare optional arguments for C++
    cpp_kwargs = {
        'mass': mass,
        'sat_type': sat_type,
        'min_lc': cutoff,
        'seed': seed,
        'enforce_mass_conservation': enforce_mass_conservation,
    }

    # Handle position/velocity inputs
    if position is not None and velocity is not None:
        # Convert km to m, km/s to m/s for C++
        cpp_kwargs['position'] = tuple(p * 1000.0 for p in position)
        cpp_kwargs['velocity'] = tuple(v * 1000.0 for v in velocity)
    elif orbit_altitude is not None:
        cpp_kwargs['orbit_altitude'] = orbit_altitude

    sim. run_explosion(**cpp_kwargs)

    fragments = sim.get_fragments()
    n_fragments = sim.get_fragment_count()
    stored_altitude = sim. get_orbit_altitude()

    coords = {
        'fragment':  np.arange(n_fragments),
        'xyz': ['x', 'y', 'z']
    }

    data_vars = {
        'fragment_size': (['fragment'], fragments['characteristic_length']),
        'area_to_mass_ratio':  (['fragment'], fragments['area_to_mass_ratio']),
        'cross_sectional_area': (['fragment'], fragments['area']),
        'fragment_mass': (['fragment'], fragments['mass']),
        'position': (['fragment', 'xyz'], fragments['position']),
        'velocity': (['fragment', 'xyz'], fragments['velocity']),
        'ejection_velocity': (['fragment', 'xyz'], fragments['ejection_velocity']),
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

    # Add orbital context to attributes if available
    if stored_altitude > 0:
        attrs['orbit_altitude_km'] = stored_altitude
    if position is not None: 
        attrs['custom_position_km'] = position
    if velocity is not None:
        attrs['custom_velocity_km_s'] = velocity

    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def collision(
    mass1: float,
    mass2: float,
    velocity_relative: float,
    sat_type1: Literal["spacecraft", "rocket_body"] = "spacecraft",
    sat_type2: Literal["spacecraft", "rocket_body"] = "spacecraft",
    cutoff: float = 0.01,
    seed: Optional[int] = None,
    enforce_mass_conservation:  bool = False,
    orbit_altitude: Optional[float] = None,
    position: Optional[Tuple[float, float, float]] = None,
    velocity1: Optional[Tuple[float, float, float]] = None,
    velocity2: Optional[Tuple[float, float, float]] = None,
) -> xr.Dataset:
    """Generate debris cloud from satellite collision."""
    sim = _core.BreakupSimulation()

    # Prepare optional arguments for C++
    cpp_kwargs = {
        'mass1': mass1,
        'mass2': mass2,
        'velocity_relative': velocity_relative,
        'sat_type1':  sat_type1,
        'sat_type2': sat_type2,
        'min_lc': cutoff,
        'seed': seed,
        'enforce_mass_conservation': enforce_mass_conservation,
    }

    # Handle position/velocity inputs
    if position is not None and velocity1 is not None and velocity2 is not None: 
        # Convert km to m, km/s to m/s for C++
        cpp_kwargs['position'] = tuple(p * 1000.0 for p in position)
        cpp_kwargs['velocity1'] = tuple(v * 1000.0 for v in velocity1)
        cpp_kwargs['velocity2'] = tuple(v * 1000.0 for v in velocity2)
    elif orbit_altitude is not None: 
        cpp_kwargs['orbit_altitude'] = orbit_altitude

    sim.run_collision(**cpp_kwargs)

    fragments = sim. get_fragments()
    n_fragments = sim.get_fragment_count()
    stored_altitude = sim.get_orbit_altitude()

    coords = {
        'fragment': np.arange(n_fragments),
        'xyz': ['x', 'y', 'z']
    }

    data_vars = {
        'fragment_size': (['fragment'], fragments['characteristic_length']),
        'area_to_mass_ratio': (['fragment'], fragments['area_to_mass_ratio']),
        'cross_sectional_area': (['fragment'], fragments['area']),
        'fragment_mass': (['fragment'], fragments['mass']),
        'position': (['fragment', 'xyz'], fragments['position']),
        'velocity': (['fragment', 'xyz'], fragments['velocity']),
        'ejection_velocity':  (['fragment', 'xyz'], fragments['ejection_velocity']),
    }

    attrs = {
        'title': 'NASA SBM Collision Debris Cloud',
        'simulation_type':  'collision',
        'mass1_kg': mass1,
        'mass2_kg': mass2,
        'relative_velocity_km_s': velocity_relative,
        'satellite_type1': sat_type1,
        'satellite_type2':  sat_type2,
        'size_cutoff_m': cutoff,
        'n_fragments': n_fragments,
        'creation_time': datetime.utcnow().isoformat(),
        'enforce_mass_conservation':  int(enforce_mass_conservation),
        'random_seed': seed if seed is not None else -1,
    }

    # Add orbital context to attributes if available
    if stored_altitude > 0:
        attrs['orbit_altitude_km'] = stored_altitude
    if position is not None:
        attrs['custom_position_km'] = position
    if velocity1 is not None and velocity2 is not None:
        attrs['custom_velocity1_km_s'] = velocity1
        attrs['custom_velocity2_km_s'] = velocity2

    return xr.Dataset(data_vars, coords=coords, attrs=attrs)
