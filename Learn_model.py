from world import *
from worlditems import *
from visualisator import *
from model_base import *


VERSION = 3
EPISODE = 2
FILE_NAME = f"./images/Learning/Version{VERSION}/"
IS_LEARNING = True
SAVE_MODEL = False
LOADED_VERSION = 3
VISION_RANGE = 10
KNOW_EXIT = True

LOADED_MODEL = f"version_{LOADED_VERSION}-episode_{EPISODE}"

#domain definiton
REAL_W = 20 #in metres
REAL_H = 20 #in metres
cell_size = 0.4
w_width = int(REAL_W/cell_size+2) #add 2 cels for left and right boundary
w_height = int(REAL_H/cell_size+2) # add 2 cels for top and bottom boundary
im_width = 1000
hw_factor = 1


#initial condition
def init_condition(condition, rho = 2):
    #REMARK nastaveno seed
    data = []
    bound = rho*cell_size*cell_size
    for y in range(1, w_height-1):
        for x in range(1,w_width):
            random.seed = 1 #!!!!!!!
            if condition(x,y) and random.random() <= bound:
                data.append(SimulatedItem(x, y, ItemType.PED))
    print(len(data))
    return data


w =  PedestrianWorld(w_width, w_height, cell_x_size=cell_size, cell_y_size=cell_size)

#print(world_item.world)
WorldItem.world=w

print(WorldItem.world)

outflow_width = 9
wall_len = int((0.33*REAL_W)/cell_size)+3
wall_len_second = int((0.33*REAL_W)/cell_size)+3
wall_second = int((0.67*REAL_W)/cell_size)


#Boundaries
w.add_boundary([BoundaryItem(i,0) for i in range(wall_len)]) #top
w.add_boundary([BoundaryItem(i+wall_second,0) for i in range(wall_len_second)]) #top
w.add_boundary([BoundaryItem(i,w_height-1) for i in range(wall_len)])#bottom
w.add_boundary([BoundaryItem(i+wall_second,w_height-1) for i in range(wall_len_second)])#bottom
w.add_boundary([BoundaryItem(0, i) for i in range(wall_len)])#left
w.add_boundary([BoundaryItem(0, i+wall_second) for i in range(wall_len_second)])#left
w.add_boundary([BoundaryItem(w_width-1, i) for i in range(wall_len)])#right
w.add_boundary([BoundaryItem(w_width-1, i+wall_second) for i in range(wall_len_second)])#right

#Outflow
outflow_start = int((0.33*REAL_W)/cell_size)+3
outflow_end = int((0.67*REAL_W)/cell_size)
#Pravy vychod
w.add_boundary([BoundaryItem(int(REAL_W/cell_size)+1, i, BoundaryType.OUT) for i in range(outflow_start, outflow_end)])#right outflow
#Horni vychod
w.add_boundary([BoundaryItem(i, 0, BoundaryType.OUT) for i in range(outflow_start, outflow_end)])#right outflow
#Levy vychod
w.add_boundary([BoundaryItem(0, i, BoundaryType.OUT) for i in range(outflow_start, outflow_end)])#right outflow
#Dolni vychod
w.add_boundary([BoundaryItem(i, int(REAL_H/cell_size)+1, BoundaryType.OUT) for i in range(outflow_start, outflow_end)])#right outflow


#down right outflow
#w.add_boundary([BoundaryItem(w.world_width-1, i, BoundaryType.OUT) for i in range(6*w_height//8, w_height-1)])#right outflow
#w.add_boundary([BoundaryItem(w.world_width-1, i) for i in range(6*w_height//8)])#right outflow

w.create_circle_hole(REAL_W/2, 5, 2) # Top Hole
w.create_rectangle_hole(0, 0, REAL_W/3, REAL_H/3) # Top left Wall

w.create_circle_hole(REAL_H-5, REAL_W/2, 2) # Right Hole
w.create_rectangle_hole(REAL_W*(2/3), 0, REAL_W, REAL_H/3) # Top right Wall

w.create_circle_hole(REAL_W/2, REAL_H-5, 2) # Bottom Hole
w.create_rectangle_hole(REAL_W*(2/3), REAL_H*(2/3), REAL_W, REAL_H) # Bottom right Wall

w.create_circle_hole(5, REAL_W/2, 2) # Left Hole
w.create_rectangle_hole(0, REAL_H*(2/3), REAL_W/3, REAL_H) # Bottom left Wall


#map_second_half = (REAL_W * (3/4))/0.4
#map_first_half = (REAL_W * (1/4))/0.4


w.ped_init_data = init_condition(lambda x, y: w.is_cell_free(x, y))
w.add_pedestrians(w.ped_init_data)



w.prepare_world()

model_simulate(FILE_NAME, VERSION, EPISODE, IS_LEARNING, SAVE_MODEL, VISION_RANGE, KNOW_EXIT, w_height, w_width,  w, im_width, hw_factor, LOADED_MODEL)