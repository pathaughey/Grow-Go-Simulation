import numpy as np
from scipy.integrate import solve_ivp


def morse_potent(cell1, cell2, dx, dy):
    delta = cell1.pos - cell2.pos
    dist = np.linalg.norm(delta)

    if dist < 1e-5:
        return 0

    RF = (cell1.ra_force[0] + cell2.ra_force[0]) / 2.0  # Repulsion Force
    RR = (cell1.ra_radius[0] + cell2.ra_radius[0]) / 2.0  # Repulsion Radius
    AF = (cell1.ra_force[1] + cell2.ra_force[1]) / 2.0  # Attraction Force
    AR = (cell1.ra_radius[1] + cell2.ra_radius[1]) / 2.0  # Attraction Radius

    cell1.pos = cell1.pos + np.array([dx, dy])
    pot_energy = RF * np.exp(-dist / RR) - AF * np.exp(-dist / AR)
    cell1.pos = cell1.pos - np.array([dx, dy])

    return pot_energy


def gradforce(cells):
    n = len(cells)
    combodict = {}

    # A way to organize interaction forces without doing it twice A -> B = B -> A
    def add2dict(index1, index2, combo):
        if index1 not in combodict:
            combodict[index1] = [combo]
        else:
            combodict[index1].append(combo)
        if index2 not in combodict:
            combodict[index2] = [combo]
        else:
            combodict[index2].append(combo)

    h = 1e-6

    # Create a vector nx1 where the jth element is the total current forces on j
    [add2dict(i, j, morse_potent(cells[i], cells[j], 0, 0))
     for i in range(n) for j in range(i + 1, n)]
    energy = np.array([sum(combodict[i]) for i in combodict])
    combodict.clear()

    # For the purposes of the gradient, we calculate the change in the x+h direction
    [add2dict(i, j, morse_potent(cells[i], cells[j], h, 0))
     for i in range(n) for j in range(i + 1, n)]
    energydx = np.array([sum(combodict[i]) for i in combodict])
    combodict.clear()

    # For the purposes of the gradient, we calculate the change in the y+h direction
    [add2dict(i, j, morse_potent(cells[i], cells[j], 0, h))
     for i in range(n) for j in range(i + 1, n)]
    energydy = np.array([sum(combodict[i]) for i in combodict])
    combodict.clear()

    gradx = np.divide((energydx - energy), h)
    grady = np.divide((energydy - energy), h)

    # The jth column represent the gradient of jth cell at current position
    return np.array((gradx, grady))


def move_cells(vitro_obj):
    cells = vitro_obj.cells

    n = len(cells)
    positions = np.array([c.pos for c in cells]).flatten(order='F')
    velocities = np.array([c.vel for c in cells]).flatten(order='F')
    masses = np.array([c.mass for c in cells]).T
    propels = np.array([c.propel for c in cells]).T
    drag = vitro_obj.drag

    pvvec = np.concatenate((positions, velocities))

    sol = solve_ivp(ode_system, t_span=(0, 1), y0=pvvec, method='RK45', args=(cells, masses, propels, drag))

    new_status = sol['y'][:, 2]

    for i in range(n):
        new_x = new_status[i]
        new_y = new_status[n + i]
        new_vel_x = new_status[2 * n + i]
        new_vel_y = new_status[3 * n + i]
        x_lower = vitro_obj.bounds[0]
        x_upper = vitro_obj.bounds[2]
        y_lower = vitro_obj.bounds[1]
        y_upper = vitro_obj.bounds[3]

        if new_x - x_lower < 0:
            new_x = x_lower
            new_vel_x = -new_vel_x
        elif x_upper - new_x < 0:
            new_x = x_upper
            new_vel_x = -new_vel_x
        elif new_y - y_lower < 0:
            new_y = y_lower
            new_vel_y = -new_vel_y
        elif y_upper - new_y < 0:
            new_y = y_upper
            new_vel_y = -new_vel_y
        vitro_obj.cells[i].pos = np.array([new_x, new_y])
        vitro_obj.cells[i].vel = np.array([new_vel_x, new_vel_y])

    return vitro_obj


def ode_system(t, pvvec, *args):
    cells, masses, propels, drag = args
    n = len(cells)

    xdot = pvvec[2 * n:3 * n]  # Velocities in X Direction
    ydot = pvvec[3 * n:4 * n]  # Velocities in Y Direction

    velmat = np.reshape(np.array(pvvec[2 * n:4 * n]), (2, n)).T  # [Vx, Vy]
    velmag2 = np.square(np.linalg.norm(velmat, axis=1))  # Row-wise squared 2-norm of velocity
    gradmat = np.transpose(gradforce(cells))  # nx2 Matrix Representing Gradient of Interactive Forces
    vdotx = np.divide(np.multiply((propels - drag * velmag2), velmat[:, 0]) - gradmat[:, 0], masses)
    vdoty = np.divide(np.multiply((propels - drag * velmag2), velmat[:, 1]) - gradmat[:, 1], masses)

    update_pvvec = np.concatenate((xdot, ydot, vdotx, vdoty)).flatten(order='C')

    return update_pvvec


### MOVING THE CELLS BY COMPARING TO THE CENTER OF MASS

def move_cells_COM(vitro_obj):
    x_moment = 0.0
    y_moment = 0.0
    total_mass = 0.0

    cells = vitro_obj.cells
    masses = []
    propels = []
    drag = vitro_obj.drag
    pv_mat = []

    for cell in vitro_obj.cells:
        total_mass += cell.mass
        x_moment += cell.pos[0] * cell.mass
        y_moment += cell.pos[1] * cell.mass
        masses.append(cell.mass)
        propels.append(cell.propel)
        pv_mat.append(np.concatenate((cell.pos, cell.vel)))

    COM = [x_moment / total_mass, y_moment / total_mass]

    pvvec = np.array(pv_mat).flatten(order='F')

    sol = solve_ivp(ode_system_COM, t_span=(0, 1), y0=pvvec, args=(cells, masses, propels, drag, COM))

    new_status = sol['y'][:, 2]

    n = len(cells)
    for i in range(n):
        new_x = new_status[i]
        new_y = new_status[n + i]
        new_vel_x = new_status[2 * n + i]
        new_vel_y = new_status[3 * n + i]
        x_lower = vitro_obj.bounds[0]
        x_upper = vitro_obj.bounds[2]
        y_lower = vitro_obj.bounds[1]
        y_upper = vitro_obj.bounds[3]

        if new_x - x_lower < 0:
            new_x = x_lower
            new_vel_x = -new_vel_x
        elif x_upper - new_x < 0:
            new_x = x_upper
            new_vel_x = -new_vel_x
        elif new_y - y_lower < 0:
            new_y = y_lower
            new_vel_y = -new_vel_y
        elif y_upper - new_y < 0:
            new_y = y_upper
            new_vel_y = -new_vel_y
        vitro_obj.cells[i].pos = np.array([new_x, new_y])
        vitro_obj.cells[i].vel = np.array([new_vel_x, new_vel_y])

    return vitro_obj


def ode_system_COM(t, pvvec, *args):
    cells, masses, propels, drag, COM = args
    n = len(cells)

    xdot = pvvec[2 * n:3 * n]  # Velocities in X Direction
    ydot = pvvec[3 * n:4 * n]  # Velocities in Y Direction

    velmat = np.reshape(np.array(pvvec[2 * n:4 * n]), (2, n)).T  # [Vx, Vy]
    velmag2 = np.square(np.linalg.norm(velmat, axis=1))  # Row-wise squared 2-norm of velocity
    gradmat = gradforce_COM(cells, COM)  # nx2 Matrix Representing Gradient of Interactive Forces
    vdotx = np.divide(np.multiply((propels - drag * velmag2), velmat[:, 0]) - gradmat[:, 0], masses)
    vdoty = np.divide(np.multiply((propels - drag * velmag2), velmat[:, 1]) - gradmat[:, 1], masses)

    update_pvvec = np.concatenate((xdot, ydot, vdotx, vdoty)).flatten(order='C')

    return update_pvvec


def morse_potent_COM(cells, COM, dx, dy):
    potential_vector = []

    for c in cells:
        dist = np.linalg.norm(np.array(COM) - [c.pos[0] + dx, c.pos[1] + dy])
        potential_vector.append(
            c.ra_force[0] * np.exp(-dist / c.ra_radius[0]) - c.ra_force[1] * np.exp(-dist / c.ra_radius[1]))

    return potential_vector


def gradforce_COM(cells, COM):
    h = 1e-6
    potent_vec = np.array(morse_potent_COM(cells, COM, 0, 0))
    potent_x_vec = np.array(morse_potent_COM(cells, COM, h, 0))
    potent_y_vec = np.array(morse_potent_COM(cells, COM, 0, h))

    grad_x_COM = (potent_x_vec - potent_vec) / h
    grad_y_COM = (potent_y_vec - potent_vec) / h

    return np.reshape(np.concatenate((grad_x_COM, grad_y_COM)), (2, len(cells))).T


# Moving the cells based on neighbors and not the total population

def move_cells_neighbors(vitro_obj):
    cells = vitro_obj.cells

    n = len(cells)
    positions = np.array([c.pos for c in cells]).flatten(order='F')
    velocities = np.array([c.vel for c in cells]).flatten(order='F')
    masses = np.array([c.mass for c in cells]).T
    propels = np.array([c.propel for c in cells]).T
    drag = vitro_obj.drag

    pvvec = np.concatenate((positions, velocities))

    sol = solve_ivp(ode_system_neighbors, t_span=(0, 1), y0=pvvec, method='RK45', args=(cells, masses, propels, drag))

    new_status = sol['y'][:, 2]

    for i in range(n):
        new_x = new_status[i]
        new_y = new_status[n + i]
        new_vel_x = new_status[2 * n + i]
        new_vel_y = new_status[3 * n + i]
        x_lower = vitro_obj.bounds[0]
        x_upper = vitro_obj.bounds[2]
        y_lower = vitro_obj.bounds[1]
        y_upper = vitro_obj.bounds[3]

        if new_x - x_lower < 0:
            new_x = x_lower
            new_vel_x = -new_vel_x
        elif x_upper - new_x < 0:
            new_x = x_upper
            new_vel_x = -new_vel_x
        elif new_y - y_lower < 0:
            new_y = y_lower
            new_vel_y = -new_vel_y
        elif y_upper - new_y < 0:
            new_y = y_upper
            new_vel_y = -new_vel_y
        vitro_obj.cells[i].pos = np.array([new_x, new_y])
        vitro_obj.cells[i].vel = np.array([new_vel_x, new_vel_y])

    return vitro_obj


def gradforce_neighbors(one_cell):
    h = 1e-6

    morse_0 = 0.0
    morse_dx = 0.0
    morse_dy = 0.0

    for cell in one_cell.neighbors:
        morse_0 += morse_potent(one_cell, cell, 0, 0)
        morse_dx += morse_potent(one_cell, cell, h, 0)
        morse_dy += morse_potent(one_cell, cell, 0, h)

    force_dx = (morse_dx - morse_0) / h
    force_dy = (morse_dy - morse_0) / h

    return [force_dx, force_dy]


def ode_system_neighbors(t, pvvec, *args):
    cells, masses, propels, drag = args
    n = len(cells)

    xdot = pvvec[2 * n:3 * n]  # Velocities in X Direction
    ydot = pvvec[3 * n:4 * n]  # Velocities in Y Direction

    velmat = np.reshape(np.array(pvvec[2 * n:4 * n]), (2, n)).T  # [Vx, Vy]
    velmag2 = np.square(np.linalg.norm(velmat, axis=1))  # Row-wise squared 2-norm of velocity

    gradmat = []  # nx2 Matrix Representing Gradient of Interactive Forces
    for cell in cells:
        gradmat.append(np.array(gradforce_neighbors(cell)))

    gradmat = np.array(gradmat)
    vdotx = np.divide(np.multiply((propels - drag * velmag2), velmat[:, 0]) - gradmat[:, 0], masses)
    vdoty = np.divide(np.multiply((propels - drag * velmag2), velmat[:, 1]) - gradmat[:, 1], masses)

    update_pvvec = np.concatenate((xdot, ydot, vdotx, vdoty)).flatten(order='C')

    return update_pvvec
