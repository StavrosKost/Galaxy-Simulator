A galaxy simulator uses the libraies matplotlib, numpy,numba,pyopencl and scipy. Since i have an AMD gpu i use pyopencl, if you have a nvidia gpu you can use cuba instead.
You can import it like this:
import cupy as cp

In compute_forces_opencl you can replace it with this and i think it will probably work(haven't tried it)

def compute_forces_barnes_hut(positions, masses, theta):
    positions_gpu = cp.asarray(positions)  # Move data to GPU
    masses_gpu = cp.asarray(masses)
    num_particles = len(positions)
    forces_gpu = cp.zeros_like(positions_gpu)
    
    for i in range(num_particles):
        r = positions_gpu - positions_gpu[i]
        distances = cp.linalg.norm(r, axis=1)
        mask = distances > 0
        forces_gpu[i] = cp.sum(G * masses_gpu[i] * masses_gpu[mask, None] * r[mask] / (distances[mask, None]**3), axis=0)
    
    return cp.asnumpy(forces_gpu)  # Move data back to CPU
