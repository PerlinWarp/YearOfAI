import numpy as np


grid = np.zeros((15,15))
pattern = [[0,0,0,0,0],
           [0,0,0,0,0],
           [0,0,1,1,1],
           [0,1,1,1,0],
           [0,0,0,0,0]]
pattern = np.array(pattern)

grid[1:pattern.shape[0]+1,1:pattern.shape[1]+1] = pattern

def life_step(x,y,grid):
    n = np.copy(grid[x-1:x+2, y-1:y+2])
    c = n[1,1]
    n[1,1] = 0
    s = np.sum(n)

    if (s > 3):
        # Overpopulation
        v = 0
    elif (s < 2):
        # Underpopulation
        v = 0
    elif (c == 0 and s == 3):
        # Reproduction
        v = 1
    else:
        v = grid[x,y]
    return v

w,h = grid.shape

while (True):
    print(grid)
    new_grid = np.copy(grid)
    for i in range(1,w):
        for j in range(1,h):
            new_grid[i,j] = life_step(i,j, grid)
    print(new_grid)
    grid = np.copy(new_grid)