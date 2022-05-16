from world import *
from worlditems import *
from visualisator import *
from model_base import *


VERSION = 4
EPISODE = 5
FILE_NAME = f"./images/T/Version{VERSION}/"
IS_LEARNING = True
SAVE_MODEL = False
LOADED_VERSION = 2
LOADED_MODEL = f"version_{LOADED_VERSION}-episode_{EPISODE}"
VISION_RANGE = 10
KNOW_EXIT = True

#domain definiton
REAL_W = 28 #in metres
REAL_H = 22 #in metres
cell_size = 0.4
w_width = int(REAL_W/cell_size+2) #add 2 cels for left and right boundary
w_height = int(REAL_H/cell_size+2) # add 2 cels for top and bottom boundary
im_width = 800
hw_factor = 1


# ------ Tvoření mapy

w =  PedestrianWorld(w_width, w_height, cell_x_size=cell_size, cell_y_size=cell_size)

#print(world_item.world)
WorldItem.world=w

print(WorldItem.world)

w.add_boundary([BoundaryItem(i,0) for i in range(w_width)]) #top
w.add_boundary([BoundaryItem(i,w_height-1) for i in range(int((10-0.3)/0.4),int((18+0.5)/0.4) )])#bottom

w.add_boundary([BoundaryItem(int((10-0.3)/0.4), i) for i in range(int(6.8/0.4), int(23/0.4))])#left
w.add_boundary([BoundaryItem(int((18+0.5)/0.4), i) for i in range(int(6.8/0.4), int(23/0.4))])#right

w.create_rectangle_hole(-0.4, 6.4, 9.7, 23)#left
w.create_rectangle_hole(18.5, 6.4, 29, 23)#left

w.add_boundary([BoundaryItem(i, int(6.4/0.4)) for i in range(int((18+0.5)/0.4), w_width)])#right2
w.add_boundary([BoundaryItem(i, int(6.4/0.4)) for i in range(0, int((10)/0.4))])#left2
# vykousnuti prava
w.add_boundary([BoundaryItem(w.world_width - 1, i, BoundaryType.OUT) for i in range(int(4.4/0.4), int(6.4/0.4))])#oright 1
w.add_boundary([BoundaryItem(w.world_width - 1, i, BoundaryType.OUT) for i in range(1, int(2.5/0.4))])#oright 2

w.add_boundary([BoundaryItem( i, int(2.5/0.4)) for i in range(int(24/0.4),w_width)])#rline1
w.add_boundary([BoundaryItem( i, int(4/0.4)) for i in range(int(24/0.4),w_width)])#rline2
w.add_boundary([BoundaryItem( int(24/0.4), i) for i in range(int(2.5/0.4), int(4/0.4))])#rline3

w.create_rectangle_hole(24, 2, 28.4, 4)#right

#vykousnuti leva
w.add_boundary([BoundaryItem(0, i, BoundaryType.OUT) for i in range(int(4.4/0.4), int(6.4/0.4))])#oleft 1
w.add_boundary([BoundaryItem(0, i, BoundaryType.OUT) for i in range(1, int(2.5/0.4))])#oleft 2

w.add_boundary([BoundaryItem( i, int(2.5/0.4)) for i in range(int(4.4/0.4))])#lline1
w.add_boundary([BoundaryItem( i, int(4/0.4)) for i in range(int(4.4/0.4))])#lline2
w.add_boundary([BoundaryItem( int(4.6/0.4), i) for i in range(int(2.5/0.4), int(4.4/0.4))])#lline3
w.create_rectangle_hole(-0.4, 2, 4, 4)#left


w.ped_init_data = init_condition((lambda x, y: x >= 10/0.4 and x <= 18/0.4 and y >=10/0.4), w_height, w_width, cell_size, rho = 2)
w.add_pedestrians(w.ped_init_data)

w.prepare_world()

# ------ Konec tvoření mapy

model_simulate(FILE_NAME, VERSION, EPISODE, IS_LEARNING, SAVE_MODEL, VISION_RANGE, KNOW_EXIT, w_height, w_width,  w, im_width, hw_factor, LOADED_MODEL = None)