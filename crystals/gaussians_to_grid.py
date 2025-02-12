# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
# cimport numpy as np

DTYPE_FLOAT = np.float64
# ctypedef np.float64_t DTYPE_FLOAT_t
DTYPE_INT = np.int
# ctypedef np.int_t DTYPE_INT_t


def gaussian(r: float, a: float, norm_factor: float, scaling: float):
    result = scaling * norm_factor * np.exp(-0.5*(r/a)**2)
    return result


# If grid index is out of bounds, finds one inside the bounds using periodic boundary conditions
def pbc_index(i_grid: int, n_grid: int):
    if i_grid < 0:
        return i_grid + n_grid
    elif i_grid >= n_grid:
        return i_grid - n_grid
    else:
        return i_grid


def collocate_gaussians_to_grid(n_grid: np.ndarray, grid_spacing: np.ndarray,
                                gaussian_coordinates: np.ndarray,
                                scaling_factors: np.ndarray,
                                gaussian_width: float, gaussian_cutoff: float):
    # cdef int i, ix, iy, iz, ia, ix_pbc, iy_pbc, iz_pbc
    # cdef double x, y, z, R, gaussian_norm_factor, dx_square, dy_square
    n_local_grid: np.ndarray = np.zeros(3, dtype=DTYPE_INT)
    nn_grid_point: np.ndarray = np.zeros(3, dtype=DTYPE_INT)
    charge_grid: np.ndarray = np.zeros(n_grid, dtype=DTYPE_FLOAT)
    
    # Pre-calculate the normalization factor of the Gaussians
    gaussian_norm_factor = 0.5/(gaussian_width**3*np.pi*np.sqrt(2.0*np.pi))
    
    # Determine number of grid points inside gaussian_cutoff
    # for i in range(3):
    #     n_local_grid[i] = int(gaussian_cutoff*gaussian_width / grid_spacing[i])
    n_local_grid = gaussian_cutoff*gaussian_width / grid_spacing
    
    # Assign Gaussians to grid points
    for ia in range(gaussian_coordinates.shape[0]):
        
        # for i in range(3):
        #     nn_grid_point[i] = np.rint(gaussian_coordinates[ia, i]/grid_spacing[i])
        nn_grid_point = np.rint(gaussian_coordinates[ia] / grid_spacing)
            
        for ix in range(nn_grid_point[0]-n_local_grid[0], nn_grid_point[0]+n_local_grid[0]+1):
            x = ix*grid_spacing[0]
            dx_square = (x-gaussian_coordinates[ia, 0])**2
            ix_pbc = pbc_index(ix, n_grid[0])
            for iy in range(nn_grid_point[1]-n_local_grid[1], nn_grid_point[1]+n_local_grid[1]+1):
                y = iy*grid_spacing[1]
                dy_square = (y-gaussian_coordinates[ia, 1])**2
                iy_pbc = pbc_index(iy, n_grid[1])
                for iz in range(nn_grid_point[2]-n_local_grid[2], nn_grid_point[2]+n_local_grid[2]+1):
                    z = iz*grid_spacing[2]
                    iz_pbc = pbc_index(iz, n_grid[2])
                    
                    R = np.sqrt(dx_square + dy_square +
                                (z-gaussian_coordinates[ia, 2])**2)
                    if R/gaussian_width < gaussian_cutoff:
                        charge_grid[ix_pbc, iy_pbc, iz_pbc] = charge_grid[ix_pbc, iy_pbc, iz_pbc] + \
                                                    gaussian(R, gaussian_width, gaussian_norm_factor, scaling_factors[ia])
    
    return charge_grid

