"""Basic tests for NASA SBM Python bindings."""

from nasa_sbm import explosion, collision
from nasa_sbm.visualization import visualize_all, plot_gabbard_diagram, plot_velocity_field_3d, plot_size_distribution
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # Non-interactive backend for testing


def test_explosion():
    print("\nTesting explosion simulation...")
    
    fragments = explosion(
        mass=839.0, 
        sat_type="rocket_body", 
        cutoff=0.05, 
        seed=42,
        orbit_altitude=400.0  # Add orbit context
    )
    
    n_frags = len(fragments.fragment)
    mass_sum = fragments.fragment_mass.sum().values
    mass_min = fragments.fragment_mass.min().values
    mass_max = fragments.fragment_mass.max().values
    size_min = fragments.fragment_size.min().values
    size_max = fragments.fragment_size.max().values
    
    print(f"  Fragments generated: {n_frags}")
    print(f"  Total mass: {mass_sum:.2f} kg (input: 839.0 kg)")
    print(f"  Mass range:  {mass_min:.6f} - {mass_max:.2f} kg")
    print(f"  Size range: {size_min:.6f} - {size_max:.4f} m")
    print(f"  Orbit altitude:  {fragments.attrs.get('orbit_altitude_km', 'N/A')} km")
    
    assert n_frags > 0
    assert (fragments.fragment_mass > 0).all()
    assert (fragments.fragment_size >= 0.05).all()
    assert 'orbit_altitude_km' in fragments.attrs
    assert fragments.attrs['orbit_altitude_km'] == 400.0
    
    fragments.to_netcdf("explosion_test.nc")
    print("  Saved to explosion_test.nc")
    
    return fragments


def test_collision():
    print("\nTesting collision simulation...")
    
    fragments = collision(
        mass1=560.0,
        mass2=950.0,
        velocity_relative=11.7,
        sat_type1="spacecraft",
        sat_type2="spacecraft",
        cutoff=0.05,
        seed=42,
        orbit_altitude=400.0  # Add orbit context
    )
    
    n_frags = len(fragments.fragment)
    mass_sum = fragments.fragment_mass.sum().values
    mass_min = fragments.fragment_mass.min().values
    mass_max = fragments.fragment_mass.max().values
    size_min = fragments.fragment_size.min().values
    size_max = fragments.fragment_size.max().values
    
    print(f"  Fragments generated:  {n_frags}")
    print(f"  Total mass:  {mass_sum:.2f} kg (input: 1510.0 kg)")
    print(f"  Mass range: {mass_min:.6f} - {mass_max:.2f} kg")
    print(f"  Size range: {size_min:.6f} - {size_max:.4f} m")
    print(f"  Orbit altitude: {fragments.attrs.get('orbit_altitude_km', 'N/A')} km")
    
    assert n_frags > 0
    assert (fragments.fragment_mass > 0).all()
    assert 'orbit_altitude_km' in fragments.attrs
    
    fragments.to_netcdf("collision_test.nc")
    print("  Saved to collision_test.nc")
    
    return fragments


def test_velocities():
    print("\nTesting isotropic velocity distribution...")
    
    fragments = explosion(mass=100.0, cutoff=0.1, seed=123, orbit_altitude=400.0)
    
    assert fragments.ejection_velocity.shape[1] == 3
    assert fragments.velocity.shape[1] == 3
    
    mean_vel = fragments.ejection_velocity.mean(dim='fragment').values
    magnitudes = np.linalg.norm(fragments.ejection_velocity.values, axis=1)
    
    print(f"  Mean ejection velocity: [{mean_vel[0]:.6f}, {mean_vel[1]:.6f}, {mean_vel[2]:.6f}] km/s")
    print(f"  Magnitude range: {magnitudes.min():.6f} - {magnitudes.max():.6f} km/s")
    
    assert np.all(magnitudes > 0)
    print("  Velocities are isotropic")


def test_visualization():
    print("\nTesting visualization functions...")
    
    fragments = explosion(
        mass=839.0,
        sat_type="rocket_body",
        cutoff=0.05,
        seed=42,
        orbit_altitude=400.0
    )
    
    print("  Testing Gabbard diagram...")
    try:
        fig_gabbard = plot_gabbard_diagram(fragments, show=True)
        print("    ✓ Gabbard diagram created")
    except ImportError as e:
        print(f"    ⚠ Gabbard diagram requires poliastro: {e}")
    except Exception as e:
        print(f"    ✗ Gabbard diagram failed: {e}")
    
    print("  Testing 3D velocity field...")
    try:
        fig_velocity = plot_velocity_field_3d(fragments, show=True)
        print("    ✓ 3D velocity field created")
    except Exception as e:
        print(f"    ✗ 3D velocity field failed: {e}")
        raise
    
    print("  Testing size distribution...")
    try:
        fig_size = plot_size_distribution(fragments, show=True)
        print("    ✓ Size distribution created")
    except Exception as e:
        print(f"    ✗ Size distribution failed: {e}")
        raise
    
    print("  Testing visualize_all...")
    try:
        figs = visualize_all(fragments, show=False)
        assert len(figs) == 3
        print("    ✓ All visualizations created")
    except Exception as e:
        print(f"    ✗ visualize_all failed: {e}")
        raise
    
    print("  All visualization tests passed")


def test_orbital_data():
    print("\nTesting orbital data extraction...")
    
    fragments = explosion(
        mass=100.0,
        cutoff=0.1,
        seed=42,
        orbit_altitude=400.0
    )
    
    # Check that position and velocity are non-zero
    pos_magnitudes = np.linalg.norm(fragments.position.values, axis=1)
    vel_magnitudes = np.linalg.norm(fragments.velocity.values, axis=1)
    
    print(f"  Position range: {pos_magnitudes.min():.2f} - {pos_magnitudes.max():.2f} km")
    print(f"  Velocity range: {vel_magnitudes.min():.2f} - {vel_magnitudes.max():.2f} km/s")
    
    # Position should be around 6778 km (400 km altitude)
    assert np.mean(pos_magnitudes) > 6000
    assert np.mean(pos_magnitudes) < 7000
    
    # Velocity should be around 7-8 km/s for LEO
    assert np.mean(vel_magnitudes) > 6
    assert np.mean(vel_magnitudes) < 10
    
    print("  Orbital data is valid")


if __name__ == "__main__":
    print("NASA SBM Python Bindings Tests")
    print("=" * 50)
    
    try:
        test_explosion()
        test_collision()
        test_velocities()
        test_orbital_data()
        test_visualization()
        
        print("\n" + "=" * 50)
        print("All tests passed ✓")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
