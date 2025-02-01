import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.image import imread
from matplotlib.cm import viridis
from matplotlib.widgets import Slider, Button
from scipy.spatial.distance import pdist
import pyopencl as cl
from numba import njit, prange

#Creation date 01/02/2025
# OpenCL setup
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Constants
G = 1.0  # Gravitational constant (scaled for simplicity)
dt = 0.01  # Time step
num_steps = 1000  # Number of simulation steps
num_particles = 1000  # Number of stars/particles
theta = 0.7  # Barnes-Hut opening angle

# Initialize particle positions, velocities, and masses
def initialize_particles(num_particles):
    r = np.linspace(0, 50, num_particles)  # Radial distance
    theta_spiral = np.linspace(0, 10 * np.pi, num_particles)  # Angle
    x = r * np.cos(theta_spiral)
    y = r * np.sin(theta_spiral)
    z = np.zeros_like(x)

    positions = np.column_stack((x, y, z)).astype(np.float32)
    velocities = np.zeros_like(positions)  # Initial velocities
    masses = np.ones(num_particles, dtype=np.float32)  # Masses (all particles have mass = 1 for simplicity)

    # Add a central supermassive black hole
    masses[0] = 1000  # Central mass
    positions[0] = np.zeros(3, dtype=np.float32)  # Center of the galaxy
    velocities[0] = np.zeros(3, dtype=np.float32)

    # Give particles initial velocities for rotation (circular motion)
    radii = np.linalg.norm(positions[1:], axis=1)
    v = np.sqrt(G * masses[0] / radii)[:, np.newaxis]
    velocities[1:] = np.cross([0, 0, 1], positions[1:]) * v / radii[:, np.newaxis]

    return positions, velocities, masses

positions, velocities, masses = initialize_particles(num_particles)

# Add sliders for dt, G, and num_particles
ax_dt = plt.axes([0.2, 0.9, 0.6, 0.03])
dt_slider = Slider(ax_dt, 'Time Step', 0.001, 0.1, valinit=dt)

ax_G = plt.axes([0.2, 0.85, 0.6, 0.03])
G_slider = Slider(ax_G, 'Gravitational Constant', 0.1, 10.0, valinit=G)

ax_particles = plt.axes([0.2, 0.8, 0.6, 0.03])
particles_slider = Slider(ax_particles, 'Number of Particles', 100, 5000, valinit=num_particles, valstep=100)

def compute_min_distance(positions):
    if len(positions) < 2:
        return float('inf')
    return np.min(pdist(positions))

# Simulation loop
trail_length = 100
trails = [positions.copy() for _ in range(trail_length)]

def update_dt(val):
    global dt
    dt = val

def update_G(val):
    global G
    G = val

def update_particles(val):
    global num_particles, positions, velocities, masses, star_colors, star_sizes, subset
    num_particles = int(val)
    positions, velocities, masses = initialize_particles(num_particles)
    
    # Recalculate star colors and sizes
    star_colors = viridis(masses / np.max(masses))
    star_sizes = 10 * masses / np.max(masses)
    
    # Update the scatter plot
    scatter._offsets3d = (positions[subset, 0], positions[subset, 1], positions[subset, 2])
    scatter.set_color(star_colors[subset])
    scatter.set_sizes(star_sizes[subset])

dt_slider.on_changed(update_dt)
G_slider.on_changed(update_G)
particles_slider.on_changed(update_particles)

# OpenCL kernel for force calculation
kernel_code = """
__kernel void compute_forces(__global float4* positions, __global float* masses, __global float4* forces, float G) {
    int i = get_global_id(0);
    float4 force = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pos_i = positions[i];
    float mass_i = masses[i];

    for (int j = 0; j < get_global_size(0); j++) {
        if (i == j) continue;
        float4 pos_j = positions[j];
        float4 r = pos_j - pos_i;
        float distance = length(r);
        if (distance > 0) {
            force += G * mass_i * masses[j] * r / (distance * distance * distance);
        }
    }
    forces[i] = force;
}
"""
prg = cl.Program(ctx, kernel_code).build()

def compute_forces_opencl(positions, masses, G):
    mf = cl.mem_flags
    positions_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=positions)
    masses_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses)
    forces_buf = cl.Buffer(ctx, mf.WRITE_ONLY, positions.nbytes)

    prg.compute_forces(queue, positions.shape, None, positions_buf, masses_buf, forces_buf, np.float32(G))
    forces = np.empty_like(positions)
    cl.enqueue_copy(queue, forces, forces_buf)

    return forces

# Simulation loop
trail_length = 100
trails = [positions.copy() for _ in range(trail_length)]

# Add collision detection and star formation
@njit(parallel=True)
def handle_collisions(positions, velocities, masses):
    num_particles = len(positions)
    to_remove = np.zeros(num_particles, dtype=np.bool_)

    for i in prange(num_particles):
        for j in range(i + 1, num_particles):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1.0:  # Collision threshold
                # Merge particles
                masses[i] += masses[j]
                velocities[i] = (masses[i] * velocities[i] + masses[j] * velocities[j]) / (masses[i] + masses[j])
                to_remove[j] = True

    # Remove merged particles
    keep = ~to_remove
    positions = positions[keep]
    velocities = velocities[keep]
    masses = masses[keep]

    return positions, velocities, masses

# Example usage:
positions, velocities, masses = initialize_particles(num_particles)
positions, velocities, masses = handle_collisions(positions, velocities, masses)
# Dark matter halo parameters
dark_matter_mass = 1e4  # Total mass of the dark matter halo
dark_matter_radius = 100  # Radius of the dark matter halo

def compute_dark_matter_force(positions):
    num_particles = len(positions)
    forces = np.zeros_like(positions)
    
    for i in range(num_particles):
        r = np.linalg.norm(positions[i])
        if r > 0:
            force_magnitude = G * dark_matter_mass * masses[i] / (r**2 + dark_matter_radius**2)
            forces[i] = -force_magnitude * positions[i] / r
    
    return forces

def simulate():
    global positions, velocities, masses, trails
    
    for _ in range(num_steps):
        # Adjust time step dynamically
        min_distance = compute_min_distance(positions)
        dt = min(0.01, min_distance * 0.1)
        
        forces = compute_forces_opencl(positions, masses, G)
        forces += compute_dark_matter_force(positions)
        velocities += forces * dt / masses[:, np.newaxis]
        positions += velocities * dt
        
        # Handle collisions and star formation
        positions, velocities, masses = handle_collisions(positions, velocities, masses)
        
        # Update trails
        trails.pop(0)
        trails.append(positions.copy())
        
        yield positions

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add galaxy background
galaxy_image = imread('galaxy_background.jpg')  # Replace with your image path
ax.imshow(galaxy_image, extent=[-50, 50, -50, 50], aspect='auto', alpha=0.5)

# Assign colors and sizes to stars
star_colors = viridis(masses / np.max(masses))  # Color based on mass
star_sizes = 10 * masses / np.max(masses)  # Size based on mass

# Plot every 10th particle
subset = slice(None, None, 10)
scatter = ax.scatter(
    positions[subset, 0], 
    positions[subset, 1], 
    positions[subset, 2], 
    c=star_colors[subset], 
    s=star_sizes[subset]
)

# Add central black hole glow
black_hole_glow = ax.scatter([0], [0], [0], c='white', s=500, alpha=0.3, edgecolors='none')

# Add gas clouds using a 3D scatter plot
gas_cloud_positions = np.random.uniform(-50, 50, (100, 3))  # Random positions for gas clouds
gas_cloud_sizes = np.random.uniform(10, 50, 100)  # Random sizes for gas clouds
gas_clouds = ax.scatter(
    gas_cloud_positions[:, 0], 
    gas_cloud_positions[:, 1], 
    gas_cloud_positions[:, 2], 
    c='blue', 
    s=gas_cloud_sizes, 
    alpha=0.2
)

# Set plot limits and labels
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(-50, 50)
ax.set_title("Galaxy Simulation", fontsize=16)
ax.set_xlabel("X (kpc)", fontsize=12)
ax.set_ylabel("Y (kpc)", fontsize=12)
ax.set_zlabel("Z (kpc)", fontsize=12)

# Add a colorbar for star masses
cbar = plt.colorbar(scatter, ax=ax, label="Star Mass")

# Global counter for the glow effect
glow_counter = 0

# Update function for animation
def update(positions):
    global glow_counter
    
    # Update scatter plot positions
    scatter._offsets3d = (
        positions[subset, 0], 
        positions[subset, 1], 
        positions[subset, 2]
    )
    
    # Update star colors and sizes
    scatter.set_color(star_colors[subset])
    scatter.set_sizes(star_sizes[subset])
    
    # Plot trails
    for i in range(len(trails) - 1):
        ax.plot(
            trails[i][subset, 0], 
            trails[i][subset, 1], 
            trails[i][subset, 2], 
            color='white', 
            alpha=0.1
        )
    
    # Update black hole glow
    black_hole_glow.set_sizes([500 + 100 * np.sin(glow_counter * 0.1)])
    glow_counter += 1  # Increment the counter
    
    return scatter, black_hole_glow

# Run animation
ani = FuncAnimation(
    fig, 
    update, 
    frames=simulate, 
    blit=True, 
    interval=10, 
    save_count=num_steps, 
    cache_frame_data=False
)

# Function to restart the animation
def restart_animation(event):
    global ani
    ani.event_source.stop()
    ani = FuncAnimation(
        fig, 
        update, 
        frames=simulate, 
        blit=True, 
        interval=10, 
        save_count=num_steps, 
        cache_frame_data=False
    )

# Attach the restart function to slider changes
dt_slider.on_changed(restart_animation)
G_slider.on_changed(restart_animation)
particles_slider.on_changed(restart_animation)

plt.show()
