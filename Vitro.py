from Cell import GrowCell
from Cell import GoCell
import numpy as np
import matplotlib.pyplot as plt


class Vitro:

    def __init__(self, bounds, drag, time_steps):
        self.bounds = bounds
        self.drag = drag
        self.cells = []
        self.cell_children = []
        self.cell_counts = []
        self.sim_length = time_steps

    def place_cells(self, count, cell_type, pop_range):

        rand_x = pop_range[0] + (pop_range[2] - pop_range[0]) * np.random.rand(count, 1)
        rand_y = pop_range[1] + (pop_range[3] - pop_range[1]) * np.random.rand(count, 1)
        cell_positions = np.array(np.concatenate((rand_x, rand_y), axis=1))

        if cell_type.lower() == "grow":
            for i in range(count):
                self.cells.insert(0, GrowCell(cell_positions[i]))

        elif cell_type.lower() == "go":
            for i in range(count):
                self.cells.insert(0, GoCell(cell_positions[i]))

    def display(self):
        for c in self.cells:
            if c.celltype.lower() == "go":
                plt.plot(c.pos[0], c.pos[1], linestyle="", marker=".", markersize=5, color="#FF5733")
            elif c.celltype.lower() == "grow":
                plt.plot(c.pos[0], c.pos[1], linestyle="", marker="o", markersize=10, color="#5D3FD3")

        plt.xlim([self.bounds[0], self.bounds[2]])
        plt.ylim([self.bounds[1], self.bounds[3]])
        plt.show()

    def create_image(self):

        plt.clf()
        plt.cla()
        plt.close('all')
        grow_count = 0
        go_count = 0

        fig, axs = plt.subplots(2, height_ratios=[4, 1])

        for c in self.cells:
            if c.celltype.lower() == "go":
                axs[0].plot(c.pos[0], c.pos[1], linestyle="", marker=".", markersize=5, color="#FF5733")
                go_count += 1
            elif c.celltype.lower() == "grow":
                axs[0].plot(c.pos[0], c.pos[1], linestyle="", marker="o", markersize=10, color="#5D3FD3")
                grow_count += 1

        axs[0].set_xlim([self.bounds[0], self.bounds[2]])
        axs[0].set_ylim([self.bounds[1], self.bounds[3]])

        # Not to plot the cell counts as a function of iteration time step
        self.cell_counts.append([grow_count, go_count])
        current_values = np.zeros((self.sim_length, 2))
        for obs in range(len(self.cell_counts)):
            current_values[obs, :] = self.cell_counts[obs]
        axs[1].plot(self.cell_counts)
        initial_count = self.cell_counts[0][0] + self.cell_counts[0][1]
        axs[1].set_xlim([0, self.sim_length])

        return plt

    def mature_cells(self):
        for c in reversed(self.cells):
            if c.celltype.lower() == 'grow':
                if c.birth_check(self):
                    eps = 1
                    new_range = [c.pos[0] - eps, c.pos[1] - eps, c.pos[0] + eps, c.pos[1] + eps]
                    self.place_cells(1, 'go', new_range)
                if c.mutate_check(self):
                    self.cells.append(GoCell(c.pos))
                    mutated_cell = self.cells[-1]
                    mutated_cell.inherit_grow_vel(c.vel)
                    self.cells.remove(c)
            elif c.celltype.lower() == 'go':
                if c.death_check():
                    self.cells.remove(c)

    # For the neighbor simulation constraint only
    def find_neighbors(self):
        for cell1 in self.cells:
            cell1.neighbors = []
            for cell2 in self.cells:
                x_diff = cell1.pos[0] - cell2.pos[0]
                y_diff = cell2.pos[1] - cell2.pos[1]
                dist = x_diff ** 2 + y_diff ** 2
                if 0 < dist < cell1.neighbor_radius:
                    cell1.add_neighbor(cell2)
