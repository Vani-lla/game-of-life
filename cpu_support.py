from os import path, makedirs, getcwd
from time import time, localtime, asctime
from concurrent import futures
from threading import Thread
from glob import glob

import cv2 as cv
import numpy as np


def surroundings(ind_x, ind_y, size, grid):
    """
       Returns the sum of surroundings
    """
    s = -255 if grid[ind_y][ind_x] == 255 else 0

    for y in range(-1, 2):
        for x in range(-1, 2):
            xi, yi = ind_x+x, ind_y+y

            if xi < size[1] and xi >= 0 and yi < size[0] and yi >= 0:
                s += grid[yi][xi]
    return s


def future(cell, neighborhood):
    """
       Returns future of cell, according to the rules
       https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
    """
    if cell:
        if neighborhood < 510:
            return 0
        elif neighborhood < 1020:
            return 255
        else:
            return 0

    else:
        if neighborhood == 765:
            return 255
        else:
            return 0


def full_surroundings(ind_x, ind_y, size, ind_correction):
    """
       Returns indexes of all surrounding cells
    """
    cells = []
    for y in range(-1, 2):
        for x in range(-1, 2):
            xi, yi = ind_x + x, ind_y + y + ind_correction

            if xi < size[1] and xi >= 0 and yi < size[0] and yi >= 0:
                cells.append((xi, yi))

    return cells


# def opt_tick(mini_grid, grid, no, ind_correction, size):
#     to_check = set()

#     future_grid = np.copy(mini_grid)

#     if ind_correction > 0:
#         if ind_correction+len(future_grid) < len(grid):
#             check = np.row_stack(
#                 (grid[ind_correction-1], future_grid, grid[ind_correction+len(future_grid)]))
#         else:
#             check = np.row_stack((grid[ind_correction-1], future_grid))
#     elif ind_correction+len(future_grid) < len(grid):
#         check = np.row_stack((future_grid, grid[len(future_grid)]))
#     else:
#         check = future_grid

#     # Selecting cells to check
#     for ind_y, row in enumerate(check):
#         if 255 in row:
#             for ind_x, cell in enumerate(row):
#                 if cell == 255:
#                     full = full_surroundings(ind_x, ind_y, size)
#                     for xi, yi in full:
#                         if yi < len(future_grid):
#                             to_check.add((xi, yi))

#     # Generating future generation
#     for ind_x, ind_y in to_check:
#         cell = grid[ind_y+ind_correction][ind_x]
#         neighborhood = surroundings(ind_x, ind_y+ind_correction, size, grid)

#         future_grid[ind_y][ind_x] = future(cell, neighborhood)

#     return future_grid, no


def cells_to_detect(mini_grid, size, ind_correction):
    # Selecting cells to check
    to_check = set()

    for ind_y, row in enumerate(mini_grid):
        if 255 in row:
            for ind_x, cell in enumerate(row):
                if cell == 255:
                    for xi, yi in full_surroundings(ind_x, ind_y, size, ind_correction):
                        to_check.add((xi, yi))

    return to_check


def future_list(l, size, grid):
    odp = []
    for ind_x, ind_y in l:
        neighborhood = surroundings(ind_x, ind_y, size, grid)
        odp.append((ind_x, ind_y, future(grid[ind_y][ind_x], neighborhood)))
    return odp


def tick(n, grid, size):
    future_grid = np.copy(grid)

    with futures.ProcessPoolExecutor() as executor:
        threads = []

        # Selecting cells
        ind_correction = 0
        for array in np.array_split(future_grid, n):
            threads.append(executor.submit(
                cells_to_detect, array, size, ind_correction))
            ind_correction += len(array)

        # Filtering
        to_check = set()
        for thread in futures.as_completed(threads):
            to_check = to_check.union(thread.result())

        # Spliting to predict future
        threads.clear()
        for array in np.array_split(list(to_check), n):
            threads.append(executor.submit(future_list, array, size, grid))

        # Rendering output
        for thread in futures.as_completed(threads):
            for ind_x, ind_y, f in thread.result():
                try:
                    future_grid[ind_y][ind_x] = f
                except IndexError:
                    continue

    return future_grid


def interupt():
    global run
    print("Commands:")
    print("  'stop' - stops rendering frames")
    print("  'frames' - prints number of rendered frames")
    print("  'frames stop' - prints number of rendered frames and stops programme")
    while True:
        command = input()
        if command == 'stop':
            run = False
            break
        elif command == 'times':
            try:
                print(f'Average time per frame: {sum(times)/len(times):3f}s')
            except:
                print("No times to print yet")
        elif command == 'frames':
            print(n)
        elif command == 'frames stop':
            run = False
            print(n)
            break
    return


if __name__ == '__main__':
    # Creating an empty directory, if it doesn't already exist
    dir_path = path.join(getcwd(), 'frames')
    if not path.exists(dir_path):
        makedirs(dir_path)

    dir_path = path.join(getcwd(), 'logs')
    if not path.exists(dir_path):
        makedirs(dir_path)

    frames = [cv.imread(path) for path in sorted(
        glob('frames/frame*.png'), key=lambda path: int(path[12:].split('.')[0]))]
    resume = input(
        f"Do you want to resume from the last frame? ({len(frames)-1}) (y/n) ") if len(frames) else 0

    # First Frame
    if not resume == 'y':
        first_frame = cv.imread('start.png')
        *size, _ = first_frame.shape
        grid = np.array([[cell[0] for cell in row] for row in first_frame])
        n, start_num = 0, 0
    else:
        first_frame = frames[-1]
        *size, _ = first_frame.shape
        grid = np.array([[cell[0] for cell in row] for row in first_frame])
        n, start_num = len(frames), len(frames)

    # Creating frames
    Thread(target=interupt).start()

    start, run, times, dates, t = time(), True, [], [], int(input('Number of cores: '))
    while run:
        s_ = time()
        cv.imwrite(f'./frames/frame{n}.png', grid)
        grid = tick(t, grid, size)
        n += 1
        times.append(time()-s_)
        dates.append(localtime())

    print(f'{time()-start:3f}s')

    # Creating log
    logs = [int(path[8:-4]) for path in sorted(glob('logs/log*.txt'),
                                               key=lambda path: int(path[8:-4].split('.')[0]))]

    if len(logs) > 0:
        with open(f'logs/log{logs[-1]+1}.txt', 'w') as log:
            log.writelines(f'[{asctime(date)}] frame{x} - {time}s\n' for x,
                           time, date in zip(range(start_num, n+1), times, dates))
    else:
        with open(f'logs/log0.txt', 'w') as log:
            log.writelines(f'[{asctime(date)}] frame{x} - {time}s\n' for x,
                           time, date in zip(range(start_num, n+1), times, dates))
