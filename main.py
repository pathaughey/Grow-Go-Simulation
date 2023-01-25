import os

import Dorsogna
import SimImages
from Vitro import Vitro
from os.path import exists
import time


def create_file_number(counter, str_length):
    str_count = str(counter)
    while len(str_count) < str_length:
        temp_list = list(str_count)
        temp_list.insert(0, '0')
        str_count = ''.join(temp_list)

    return str_count


if __name__ == "__main__":

    #INITIALIZING TIME STEPS AND CELL COUNTS
    ttl_steps = 100
    grow0 = 10
    go0 = 20

    # Simulation Heuristic
    neighbors = True

    # Parameter setting for the environment
    myVitro = Vitro([0, 0, 100, 100], 0.5, ttl_steps)

    # Set up the initial placement of cells
    myVitro.place_cells(grow0, 'grow', [40, 40, 60, 60])
    myVitro.place_cells(go0, 'go', [35, 35, 65, 65])

    # Create a Folder to Save Photos
    parent_path = # Your file path to the source code #
    folder_name = # The name of the folder #
    image_folder = parent_path + folder_name
    image_size = [6.5, 4.25]

    if not exists(parent_path+folder_name):
        os.mkdir(parent_path+folder_name)

    frame = myVitro.create_image()
    frame.savefig(image_folder + '/Frame0000')

    # Iterations
    count = 1
    while count < ttl_steps:
        t0 = time.time()

        frame = myVitro.create_image()
        file_num = create_file_number(count, 4)
        frame.savefig(image_folder + '/Frame' + file_num)

        if neighbors:
            myVitro.find_neighbors()
            myVitro = Dorsogna.move_cells_neighbors(myVitro)
        else:
            myVitro = Dorsogna.move_cells(myVitro)
            myVitro.mature_cells()

        myVitro.mature_cells()

        iteration_time = time.time() - t0
        n_count = len(myVitro.cells)
        print(F'Cell_Count: {n_count}\t\tRun_Time: {iteration_time}')
        count += 1

    SimImages.make_gif(image_folder + '/')

    SimImages.delete_frames(image_folder)
