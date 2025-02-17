# https://github.com/SINGROUP/Potential_solver
# -*- coding: utf-8 -*-
#!/usr/bin/python
import pathlib
import os, sys
root = pathlib.Path(__file__).parent.parent
sys.path.append(root)

# import pyximport; pyximport.install()
import numpy as np
import time

import crystals.gaussians_to_grid as gaussians_to_grid

eps_0 = 8.854187817620e-22/1.6021766208e-19 # e/(V*Å)
#eps_0 = 1.0

class ElectrostaticPotentialSolver(object):
    def __init__(self, atom_coordinates, atom_charges, sim_cell):
        # Check that input is sensible
        assert type(atom_coordinates) == np.ndarray and atom_coordinates.shape[1] == 3, \
            "atom_coordinates must be a NumPy array of shape (n_atoms, 3)"
        assert type(atom_charges) == np.ndarray and atom_charges.shape == (atom_coordinates.shape[0],), \
            "atom_charges must be a NumPy array of shape (n_atoms,)"
        assert type(sim_cell) == np.ndarray and sim_cell.shape == (3,), \
            "sim_cell must be a NumPy array of shape (3,), {}".format(sim_cell.shape)
        
        # The solver does not work if the system is not charge neutral
        if abs(atom_charges.sum()) > 1.0e-10:
            raise Exception('System is not charge neutral. Atom charges sum up to {}.'.format(atom_charges.sum()))
        
        self.atom_coordinates = atom_coordinates
        self.atom_charges = atom_charges
        self.sim_cell = sim_cell
        
        self.is_solver_run = False
        self.n_grid = np.zeros(3, dtype=np.int)
        self.grid_spacing = np.zeros(3, dtype=np.float)
        self.kx = 0.0
        self.ky = 0.0
        self.kz = 0.0
        self.charge_grid = 0.0
        self.pot_grid = 0.0
        
        # Initialize the computational parameters for the solver
        self.solver_parameter_names = ['k_cutoff', 'gaussian_width', 'gaussian_cutoff']
        self.solver_parameters = {}
        for parameter_name in self.solver_parameter_names:
            self.solver_parameters[parameter_name] = 0.0
    
    
    def set_parameter(self, parameter_name, parameter_value):
        if parameter_name in self.solver_parameter_names:
            self.solver_parameters[parameter_name] = parameter_value
        else:
            print("Parameter name '{}' is not included in the set of solver parameters:\n{}".format(parameter_name,
                                                                                        self.solver_parameter_names))
    
    
    def solve_potential(self):
        for param_name, param_value in self.solver_parameters.items():
            if param_value == 0.0:
                raise Exception("Solver parameter '{}' is not set yet".format(param_name))
        
        self.pot_grid = np.zeros(self.n_grid, dtype=np.float)
        total_time = 0.0
        
        start = time.time()
        self._initialize_grid()
        end = time.time()
        total_time = total_time + end - start
        print ('| Grid initialization took {} s'.format(end-start))
        
        start = time.time()
        #self._collocate_gaussians_to_grid()
        self.charge_grid = gaussians_to_grid.collocate_gaussians_to_grid(self.n_grid, self.grid_spacing, self.atom_coordinates,
                                                                        self.atom_charges, self.solver_parameters['gaussian_width'],
                                                                        self.solver_parameters['gaussian_cutoff'])
        end = time.time()
        total_time = total_time + end - start
        print ('| Collocation of Gaussians to grid took {} s'.format(end-start)
        )
        start = time.time()
        self._solve_fft_poisson()
        end = time.time()
        total_time = total_time + end - start
        print ('| Solving of the potential using FFT took {} s'.format(end-start))
        
        print ('Total time taken to find the solution: {} s'.format(total_time))
        self.is_solver_run = True

    def __repr__(self, ):
        print("Parameters of solver: ")
        return self.__dict__
    
    def get_data_grid(self, data_type):
        if not self.is_solver_run:
            raise Exception('You have not run the solver yet, so you cannot get any data. Run solve_potential method first.')
        
        if data_type == 'charge':
            data_grid = self.charge_grid
        elif data_type == 'potential':
            data_grid = self.pot_grid
        elif data_type == 'efield_x':
            data_grid, trash_1, trash_2 = np.gradient(-self.pot_grid, self.grid_spacing[0],
                                                    self.grid_spacing[1], self.grid_spacing[2])
            del trash_1
            del trash_2
        elif data_type == 'efield_y':
            trash_1, data_grid, trash_2 = np.gradient(-self.pot_grid, self.grid_spacing[0],
                                                    self.grid_spacing[1], self.grid_spacing[2])
            del trash_1
            del trash_2
        elif data_type == 'efield_z':
            trash_1, trash_2, data_grid = np.gradient(-self.pot_grid, self.grid_spacing[0],
                                                    self.grid_spacing[1], self.grid_spacing[2])
            del trash_1
            del trash_2
        else:
            print ("Available types of data are: 'charge', 'potential', 'efield_x', 'efield_y' and 'efield_z'")
        
        xs = np.linspace(0.0, 1.0, num=self.n_grid[0], endpoint=False)*self.sim_cell[0]
        ys = np.linspace(0.0, 1.0, num=self.n_grid[1], endpoint=False)*self.sim_cell[1]
        zs = np.linspace(0.0, 1.0, num=self.n_grid[2], endpoint=False)*self.sim_cell[2]
        return xs, ys, zs, data_grid
    
    
    def get_data_slice(self, data_type, pos_along_normal, normal_direction='z'):
        xs, ys, zs, data_grid = self.get_data_grid(data_type)
        
        if normal_direction == 'x':
            normal_coord = 0
        elif normal_direction == 'y':
            normal_coord = 1
        else:
            normal_coord = 2
        
        i_normal_lower = int(pos_along_normal/self.grid_spacing[normal_coord])
        i_normal_higher = i_normal_lower + 1
        interp_factor = (pos_along_normal - i_normal_lower*self.grid_spacing[normal_coord])/self.grid_spacing[normal_coord]
        
        if normal_coord == 0:
            data_slice = np.zeros((self.n_grid[1], self.n_grid[2]))
            for iy in range(self.n_grid[1]):
                for iz in range(self.n_grid[2]):
                    data_slice[iy, iz] = (1-interp_factor)*data_grid[i_normal_lower, iy, iz] + interp_factor*data_grid[i_normal_higher, iy, iz]
            return ys, zs, data_slice
            
        elif normal_coord == 1:
            data_slice = np.zeros((self.n_grid[0], self.n_grid[2]))
            for ix in range(self.n_grid[0]):
                for iz in range(self.n_grid[2]):
                    data_slice[ix, iz] = (1-interp_factor)*data_grid[ix, i_normal_lower, iz] + interp_factor*data_grid[ix, i_normal_higher, iz]
            return xs, zs, data_slice
            
        else:
            data_slice = np.zeros((self.n_grid[0], self.n_grid[1]))
            for ix in range(self.n_grid[0]):
                for iy in range(self.n_grid[1]):
                    data_slice[ix, iy] = (1-interp_factor)*data_grid[ix, iy, i_normal_lower] + interp_factor*data_grid[ix, iy, i_normal_higher]
            return xs, ys, data_slice
    
    
    def _initialize_grid(self):
        desired_grid_spacing = 0.5/self.solver_parameters['k_cutoff']
        for i in range(3):
            self.n_grid[i] = int(self.sim_cell[i]/desired_grid_spacing)+1
            self.grid_spacing[i] = self.sim_cell[i]/float(self.n_grid[i])
        
        self.kx = np.fft.fftfreq(self.n_grid[0], self.grid_spacing[0])
        self.ky = np.fft.fftfreq(self.n_grid[1], self.grid_spacing[1])
        self.kz = np.fft.fftfreq(self.n_grid[2], self.grid_spacing[2])
        
        print ("""
         'Grid parameters:' \n
         '| cell_x = {}, cell_y = {}, cell_z = {}'.format(self.sim_cell[0], self.sim_cell[1], self.sim_cell[2]) \n
         '| nx = {}, ny = {}, nz = {}'.format(self.n_grid[0], self.n_grid[1], self.n_grid[2]) \n
         '| dx = {}, dy = {}, dz = {}'.format(self.grid_spacing[0], self.grid_spacing[1], self.grid_spacing[2]) \n
         '| kx_max = {}, ky_max = {}, kz_max = {}'.format(self.kx.max(), self.ky.max(), self.kz.max()) \n
         """)
    
    
    def _solve_fft_poisson(self):
        start = time.time()
        charge_k_space = np.fft.fftn(self.charge_grid)
        end = time.time()
        print ('| | FFT took {} s'.format(end-start))
        
        start = time.time()
        pot_k_space = np.zeros(self.n_grid, dtype=np.complex) #zeros array of n_grid shape!
        
        for ix in range(1, self.n_grid[0]):
            for iy in range(self.n_grid[1]):
                for iz in range(self.n_grid[2]):
                    k_squared = self.kx[ix]**2 + self.ky[iy]**2 + self.kz[iz]**2
                    pot_k_space[ix, iy, iz] = charge_k_space[ix, iy, iz]/k_squared
        
        ix = 0
        for iy in range(1, self.n_grid[1]):
            for iz in range(self.n_grid[2]):
                k_squared = self.ky[iy]**2 + self.kz[iz]**2
                pot_k_space[ix, iy, iz] = charge_k_space[ix, iy, iz]/k_squared
        
        ix = 0
        iy = 0
        for iz in range(1, self.n_grid[2]):
            k_squared = self.kz[iz]**2
            pot_k_space[ix, iy, iz] = charge_k_space[ix, iy, iz]/k_squared
        
        pot_k_space = pot_k_space/(4.0*np.pi*np.pi*eps_0)
        end = time.time()
        print ('| | k-space arithmetics took {} s'.format(end-start))
        
        start = time.time()
        self.pot_grid = np.fft.ifftn(pot_k_space).real
        end = time.time()
        print ('| | Inverse FFT took {} s'.format(end-start))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    filename = os.path.join(os.getcwd(), "crystals", 'pb_example.xyz')
    sim_cell = np.array([42.616, 73.816, 54.424])
    #filename = 'triangle_small_whole_step_charges.xyz'
    #sim_cell = np.array([42.616, 55.362, 57.424])
    #filename = '20x10x3_triangle_big_charges.xyz'
    #sim_cell = np.array([106.54, 92.27, 43.068])

    slice_z = 1.0

    # Load geometry and charges
    data = np.loadtxt(filename, skiprows=2, usecols=(1,2,3,4))
    atom_coordinates = data[:, 0:3]
    atom_charges = data[:, 3]
    surf_z = atom_coordinates[:, 2].max()

    # Solve electrostatic potential
    solver = ElectrostaticPotentialSolver(atom_coordinates, atom_charges, sim_cell)
    solver.set_parameter('k_cutoff', 3.0) # optimal: 3.0
    solver.set_parameter('gaussian_width', 0.5) # optimal: 0.5
    solver.set_parameter('gaussian_cutoff', 6.0) # optimal: 6.0
    solver.solve_potential()
    print(repr(solver))

    # If you want to get the complete data grid instead of just a slice, use get_data_grid method. For example:
    # xs, ys, zs, pot_grid = solver.get_data_grid('potential')

    # Plot charge distribution
    xs, ys, charge_slice = solver.get_data_slice('charge', surf_z+slice_z, normal_direction='z')
    plt.figure()
    plt.contourf(xs, ys, charge_slice.T, 50)
    plt.axes().set_aspect('equal')
    plt.title(u'Charge density, z = {:.2f} Å'.format(slice_z), size=16)
    plt.xlabel(u'x (Å)', size=14)
    plt.ylabel(u'y (Å)', size=14)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(u'$\\rho$ (e/Å$^{3}$)', size=14)
    plt.show()
