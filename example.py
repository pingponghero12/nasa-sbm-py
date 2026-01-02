"""Usage examples for NASA SBM Python bindings."""

from nasa_sbm import explosion, collision
from nasa_sbm.visualization import visualize_all, plot_gabbard_diagram, plot_velocity_field_3d, plot_size_distribution
import numpy as np


def example_explosion():
    """Simulate satellite explosion event."""
    print("\n--- Explosion Simulation Example ---")

    fragments = explosion(
        mass=839.0,
        sat_type="rocket_body",
        cutoff=0.05,
        seed=42,
        orbit_altitude=400.0
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
    print(f"  Orbit altitude: {fragments.attrs.get('orbit_altitude_km', 'N/A')} km")

    fragments.to_netcdf("explosion_example.nc")
    print("  Saved to explosion_example.nc")

    return fragments


def example_collision():
    """Simulate collision between two satellites."""
    print("\n--- Collision Simulation Example ---")

    fragments = collision(
        mass1=560.0,
        mass2=950.0,
        velocity_relative=11.7,
        sat_type1="spacecraft",
        sat_type2="spacecraft",
        cutoff=0.05,
        seed=42,
        orbit_altitude=400.0
    )

    n_frags = len(fragments.fragment)
    mass_sum = fragments.fragment_mass.sum().values
    mass_min = fragments.fragment_mass.min().values
    mass_max = fragments.fragment_mass.max().values
    size_min = fragments.fragment_size.min().values
    size_max = fragments.fragment_size.max().values
    
    print(f"  Fragments generated: {n_frags}")
    print(f"  Total mass: {mass_sum:.2f} kg (input: 1510.0 kg)")
    print(f"  Mass range: {mass_min:.6f} - {mass_max:.2f} kg")
    print(f"  Size range: {size_min:.6f} - {size_max:.4f} m")
    print(f"  Orbit altitude: {fragments.attrs.get('orbit_altitude_km', 'N/A')} km")
    
    fragments.to_netcdf("collision_example.nc")
    print("  Saved to collision_example.nc")
    
    return fragments


def example_velocity_analysis():
    """Analyze fragment velocity distribution."""
    print("\n--- Velocity Distribution Analysis ---")
    
    fragments = explosion(mass=100.0, cutoff=0.1, seed=123, orbit_altitude=400.0)
    
    mean_vel = fragments.ejection_velocity.mean(dim='fragment').values
    magnitudes = np.linalg.norm(fragments.ejection_velocity.values, axis=1)
    
    print(f"  Mean ejection velocity: [{mean_vel[0]:.6f}, {mean_vel[1]:.6f}, {mean_vel[2]:.6f}] km/s")
    print(f"  Magnitude range: {magnitudes.min():.6f} - {magnitudes.max():.6f} km/s")
    print(f"  Ejection velocity shape: {fragments.ejection_velocity.shape}")
    print(f"  Total velocity shape: {fragments.velocity.shape}")


def example_visualization():
    """Generate visualization plots for fragment data."""
    print("\n--- Visualization Examples ---")
    
    fragments = explosion(
        mass=839.0,
        sat_type="rocket_body",
        cutoff=0.05,
        seed=42,
        orbit_altitude=400.0
    )
    
    print("  Creating Gabbard diagram...")
    try:
        fig_gabbard = plot_gabbard_diagram(fragments, show=True)
        print("    Gabbard diagram created")
    except ImportError as e:
        print(f"    Gabbard diagram requires poliastro: {e}")
    except Exception as e:
        print(f"    Error:  {e}")
    
    print("  Creating 3D velocity field...")
    fig_velocity = plot_velocity_field_3d(fragments, show=True)
    print("    3D velocity field created")
    
    print("  Creating size distribution plot...")
    fig_size = plot_size_distribution(fragments, show=True)
    print("    Size distribution created")
    
    print("  Creating all visualizations...")
    figs = visualize_all(fragments, show=False)
    print(f"    {len(figs)} visualizations created")


def example_orbital_analysis():
    """Extract and analyze orbital parameters."""
    print("\n--- Orbital Data Analysis ---")
    
    fragments = explosion(
        mass=100.0,
        cutoff=0.1,
        seed=42,
        orbit_altitude=400.0
    )

    pos_magnitudes = np.linalg.norm(fragments.position.values, axis=1)
    vel_magnitudes = np.linalg.norm(fragments.velocity.values, axis=1)

    print(f"  Position range: {pos_magnitudes.min():.2f} - {pos_magnitudes.max():.2f} km")
    print(f"  Mean position: {np.mean(pos_magnitudes):.2f} km")
    print(f"  Velocity range: {vel_magnitudes.min():.2f} - {vel_magnitudes.max():.2f} km/s")
    print(f"  Mean velocity: {np.mean(vel_magnitudes):.2f} km/s")


if __name__ == "__main__":
    print("NASA SBM Python Bindings - Examples")
    print("=" * 50)

    example_explosion()
    example_collision()
    example_velocity_analysis()
    example_orbital_analysis()
    example_visualization()

    print("\n" + "=" * 50)
    print("Examples completed")
