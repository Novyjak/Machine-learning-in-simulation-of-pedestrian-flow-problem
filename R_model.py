from world import *
from worlditems import *
from visualisator import *
from model_base import *


VERSION = 4
EPISODE = 5
FILE_NAME = f"./images/R/Version{VERSION}/"
IS_LEARNING = True
SAVE_MODEL = False
LOADED_VERSION = 2
LOADED_MODEL = f"version_{LOADED_VERSION}-episode_{EPISODE}"

#domain definiton
REAL_W = 40 #in metres
REAL_H = 10 #in metres
cell_size = 0.4
w_width = int(REAL_W/cell_size+2) #add 2 cels for left and right boundary
w_height = int(REAL_H/cell_size+2) # add 2 cels for top and bottom boundary
im_width = 1200
hw_factor = 4


#initial condition

# ------ Tvoření mapy

w =  PedestrianWorld(w_width, w_height, cell_x_size=cell_size, cell_y_size=cell_size)

#print(world_item.world)
WorldItem.world=w

print(WorldItem.world)

w.add_boundary([BoundaryItem(i,0) for i in range(w_width)]) #top
w.add_boundary([BoundaryItem(i,w_height-1) for i in range(w_width)])#bottom
w.add_boundary([BoundaryItem(0, i) for i in range(w_height)])#left
#right outflow
w.add_boundary([BoundaryItem(w.world_width-1, i, BoundaryType.OUT) for i in range(1, w_height-1)])#right outflow


w.create_circle_hole(32, 5, 2) #32

w.ped_init_data = init_condition( (lambda x,y: x<=20/0.4), w_height, w_width, cell_size, rho = 2)
w.add_pedestrians(w.ped_init_data)

w.prepare_world()

# ------ Konec tvoření mapy

model_simulate(FILE_NAME, VERSION, EPISODE, IS_LEARNING, SAVE_MODEL, w_height, w_width,  w, im_width, hw_factor, LOADED_MODEL)