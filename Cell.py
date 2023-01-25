import numpy as np
from Probability import NormParamsSearch
from Probability import GammaParamsSearch
from scipy import stats

class Cell():
    def __init__(self, init_pos, init_vel, cell_ra_force, cell_ra_radius, init_mass, cell_propel, neighbor_radius):
        self.pos = init_pos
        self.vel = init_vel
        self.ra_force = cell_ra_force
        self.ra_radius = cell_ra_radius
        self.propel = cell_propel
        self.mass = init_mass
        self.neighbor_radius = neighbor_radius
        self.neighbors = []  # List of cell objects within a given radius

    def cellinfo(self):
        print('Position (x,y) = (%4.1f, %4.1f)' % (self.pos[0], self.pos[1]))
        print('Velocity <v_x, v_y> = <%4.1f, %4.1f>' % (self.vel[0], self.vel[1]))
        print('Cell Mass = %5.1f\n' % self.mass)

    def add_neighbor(self, other_cell):
        self.neighbors.append(other_cell)


class GrowCell(Cell):
    birth_rate_params = GammaParamsSearch(0.9, [50, 100])
    child_probability = 1  # If rand < 100% then grow, else, go
    init_vel_params = NormParamsSearch(0.9, [-0.02, 0.02])
    repulse_f_param = GammaParamsSearch(0.9, [9, 11])
    attract_f_param = GammaParamsSearch(0.9, [9, 11])
    repulse_radius_param = GammaParamsSearch(0.9, [0.05, 0.15])
    attract_radius_param = GammaParamsSearch(0.9, [9, 11])
    mass_params = GammaParamsSearch(0.9, [100, 200])
    mutate_params = GammaParamsSearch(0.9, [2, 5])
    neighbor_radius_params = GammaParamsSearch(0.9, [1, 2])

    def __init__(self, init_pos):
        self.celltype = "Grow"

        # Setting up the Physiological Parameters
        cell_propel = 0
        self.birthrate = stats.gamma.rvs(GrowCell.birth_rate_params[0], loc=0, scale=GrowCell.birth_rate_params[1])
        init_vel = stats.norm.rvs(loc=GrowCell.init_vel_params[0], scale=GrowCell.init_vel_params[1], size=2)
        repulse_force = stats.gamma.rvs(GrowCell.repulse_f_param[0], loc=0, scale=GrowCell.repulse_f_param[1])
        attract_force = stats.gamma.rvs(GrowCell.attract_f_param[0], loc=0, scale=GrowCell.attract_f_param[1])
        repulse_radius = stats.gamma.rvs(GrowCell.repulse_radius_param[0], loc=0, scale=GrowCell.repulse_radius_param[1])
        attract_radius = stats.gamma.rvs(GrowCell.attract_radius_param[0], loc=0, scale=GrowCell.attract_radius_param[1])
        cell_mass = stats.gamma.rvs(GrowCell.mass_params[0], loc=0, scale=GrowCell.mass_params[1])

        cell_ra_force = [repulse_force, attract_force]
        cell_ra_radius = [repulse_radius, attract_radius]

        neighbor_radius = stats.gamma.rvs(GrowCell.neighbor_radius_params[0], loc=0,
                                          scale=GrowCell.neighbor_radius_params[1])

        super().__init__(init_pos, init_vel, cell_ra_force, cell_ra_radius, cell_mass, cell_propel, neighbor_radius)

    def birth_check(self, vitro_obj):
        if np.random.rand() <= 1.0 / self.birthrate:
            eps = 1
            new_range = [self.pos[0] - eps, self.pos[1] - eps, self.pos[0] + eps, self.pos[1] + eps]
            if np.random.rand() <= self.child_probability:
                vitro_obj.place_cells(1, 'grow', new_range)
            else:
                vitro_obj.place_cells(1, 'go', new_range)

    def mutate_check(self, vitro_obj):
        x_moment = 0.0
        y_moment = 0.0
        total_mass = 0.0

        for c in vitro_obj.cells:
            x_moment += c.pos[0]*c.mass
            y_moment += c.pos[1]*c.mass
            total_mass += c.mass

        center_of_mass = np.array([x_moment / total_mass, y_moment / total_mass])
        dist = np.linalg.norm(self.pos - center_of_mass)

        if np.random.rand() <= stats.gamma.cdf(dist, GrowCell.mutate_params[0], loc=0, scale=GrowCell.mutate_params[1]):
            return True
        else:
            return False



class GoCell(Cell):
    propel_params = GammaParamsSearch(0.9, [0.01, 0.015])
    init_vel_params = NormParamsSearch(0.9, [-0.03, 0.03])
    repulse_f_param = GammaParamsSearch(0.9, [9, 11])
    attract_f_param = GammaParamsSearch(0.9, [9, 11])
    repulse_radius_param = GammaParamsSearch(0.9, [0.05, 0.1])
    attract_radius_param = GammaParamsSearch(0.9, [9, 11])
    mass_params = GammaParamsSearch(0.9, [10, 20])
    life_span_param = GammaParamsSearch(0.9, [100, 200])
    neighbor_radius_params = GammaParamsSearch(0.9, [1, 2])

    def __init__(self, init_pos):
        self.celltype = "Go"

        # Setting up the Physiological Parameters
        cell_propel = stats.gamma.rvs(GoCell.propel_params[0], loc=0, scale=GoCell.propel_params[1])
        init_vel = stats.norm.rvs(loc=GoCell.init_vel_params[0], scale=GoCell.init_vel_params[1], size=2)
        repulse_force = stats.gamma.rvs(GoCell.repulse_f_param[0], loc=0, scale=GoCell.repulse_f_param[1])
        attract_force = stats.gamma.rvs(GoCell.attract_f_param[0], loc=0, scale=GoCell.attract_f_param[1])
        repulse_radius = stats.gamma.rvs(GoCell.repulse_radius_param[0], loc=0, scale=GoCell.repulse_radius_param[1])
        attract_radius = stats.gamma.rvs(GoCell.attract_radius_param[0], loc=0, scale=GoCell.attract_radius_param[1])
        cell_mass = stats.gamma.rvs(GoCell.mass_params[0], loc=0, scale=GoCell.mass_params[1])
        self.life_span = stats.gamma.rvs(GoCell.life_span_param[0], loc=0, scale=GoCell.life_span_param[1])

        cell_ra_force = [repulse_force, attract_force]
        cell_ra_radius = [repulse_radius, attract_radius]

        neighbor_radius = stats.gamma.rvs(GrowCell.neighbor_radius_params[0], loc=0,
                                          scale=GrowCell.neighbor_radius_params[1])

        super().__init__(init_pos, init_vel, cell_ra_force, cell_ra_radius, cell_mass, cell_propel, neighbor_radius)

    def death_check(self):
        if np.random.rand() <= 1 / self.life_span:
            return True
        else:
            return False

    def inherit_grow_vel(self, vel):
        self.vel = vel
