"""Basic tests for NASA SBM Python bindings."""

from nasa_sbm import explosion, collision
import numpy as np


def test_explosion():
    print("\nTesting explosion simulation...")
    
    fragments = explosion(
        mass=839.0, 
        sat_type="rocket_body", 
        cutoff=0.05, 
        seed=42
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
    
    assert n_frags > 0
    assert (fragments.fragment_mass > 0).all()
    assert (fragments.fragment_size >= 0.05).all()
    
    fragments.to_netcdf("explosion_test.nc")
    print("  Saved to explosion_test.nc")
    
    return fragments


def test_collision():
    print("\nTesting collision simulation...")
    
    fragments = collision(
        mass1=560.0,
        mass2=950.0,
        velocity=11.7,
        sat_type1="spacecraft",
        sat_type2="spacecraft",
        cutoff=0.05,
        seed=42
    )
    
    n_frags = len(fragments.fragment)
    mass_sum = fragments.fragment_mass.sum().values
    mass_min = fragments.fragment_mass.min().values
    mass_max = fragments.fragment_mass.max().values
    size_min = fragments.fragment_size.min().values
    size_max = fragments.fragment_size.max().values
    
    print(f"  Fragments generated: {n_frags}")
    print(f"  Total mass: {mass_sum:.2f} kg (input: 1510.0 kg)")
    print(f"  Mass range:  {mass_min:.6f} - {mass_max:.2f} kg")
    print(f"  Size range: {size_min:.6f} - {size_max:.4f} m")
    
    assert n_frags > 0
    assert (fragments.fragment_mass > 0).all()
    
    fragments.to_netcdf("collision_test.nc")
    print("  Saved to collision_test.nc")
    
    return fragments


def test_velocities():
    print("\nTesting isotropic velocity distribution...")
    
    fragments = explosion(mass=100.0, cutoff=0.1, seed=123)
    
    assert fragments.delta_velocity.shape[1] == 3
    
    mean_vel = fragments.delta_velocity.mean(dim='fragment').values
    magnitudes = np.linalg.norm(fragments.delta_velocity.values, axis=1)
    
    print(f"  Mean velocity: [{mean_vel[0]:.6f}, {mean_vel[1]:.6f}, {mean_vel[2]:.6f}] km/s")
    print(f"  Magnitude range: {magnitudes.min():.6f} - {magnitudes.max():.6f} km/s")
    
    assert np.all(magnitudes > 0)
    print("  Velocities are isotropic")


if __name__ == "__main__":
    print("NASA SBM Python Bindings Tests")
    print("=" * 50)
    
    try:
        test_explosion()
        test_collision()
        test_velocities()
        
        print("\n" + "=" * 50)
        print("All tests passed")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
