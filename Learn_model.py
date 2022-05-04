from world import *
from worlditems import *
from visualisator import *
import random
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import learning as lern
import os


VERSION = 2
FILE_NAME = f"./images/Learning/Version{VERSION}/"
IS_LEARNING = True
LOADED_MODEL = "version_3-episode_2"

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


ped_init_data = init_condition(lambda x, y: w.is_cell_free(x, y))
w.add_pedestrians(ped_init_data)



w.prepare_world()

# Přidaný kód #####################################
# Mapa ze které se budou vytahovat data pro učení.
# Vytvoří bázi s boundaries, které se nemění.
learning_map = lern.LearningMap(w_width, w_height)
for boundary in w._boundary:
    learning_map.add_boundary(boundary.x, boundary.y, boundary.btype)
for hole in w._holes:
    learning_map.add_boundary(hole.x, hole.y)
learning_map.reset_map()

learning_map.save_version_info(FILE_NAME, VERSION, IS_LEARNING, w.pedestrian_count)

###################################################

# Použit k vytvoření obrazu 
wvisu = WorldVisualisator(w)
images = []
im = wvisu.DrawWorld(size = (im_width,im_width//hw_factor))
im.save(FILE_NAME + '0.png')

images.append(im) #dej tam obrazek c 0

if not IS_LEARNING:
    learning_map.model.set_weights(learning_map.load_model(LOADED_MODEL))


for step in range(2001):
    print("step: " + str(step))

    minfo = w.move(learning_map, IS_LEARNING)  #  proměnná pedestrians je použita pro strojové učení
    #print(minfo)
    

    if IS_LEARNING and (learning_map.episode_step_count >= learning_map.max_steps_per_episode):
        # update the the target network with new weights
        learning_map.model_target.set_weights(learning_map.model.get_weights())
        # Log details
        template = "running reward: {:.2f} at episode {}, frame count {}"
        print(template.format(learning_map.running_reward, learning_map.episode_count, learning_map.frame_count))

        learning_map.episode_step_count = 0
        learning_map.episode_reward_history.append(learning_map.episode_reward)
        learning_map.running_reward = np.mean(learning_map.episode_reward)
        learning_map.episode_reward = []
        
        learning_map.save_model(f"version_{VERSION}-episode_{learning_map.episode_count}")
        learning_map.episode_count += 1

    learning_map.reset_map()

    imped = wvisu.DrawWorld(size=(im_width, im_width // hw_factor))
    images.append(imped)
    imped.save(FILE_NAME + f'steps/img-version-{VERSION}-step-{step}.png')

    if step % 100 == 0:
        #for item in sorted(minfo, key = minfo.get):
        #    print(f'{item}:-> {minfo[item]}')
        images[0].save(FILE_NAME + f'img-version-{VERSION}-step-{step}.gif',
                       save_all=True,
                       append_images=images[1:],
                       duration=200,
                       loop=0)
        images = []
        if step % 2000 == 0 and False: # Ponecháno, v případě, že by jsem v budoucnu chtěl upravit
            w.remove_pedestrians()
            w.add_pedestrians(init_condition(lambda x, y: x <= 20 / 0.4, rho=1))
    #print(w.pedestrian_count)
    if w.pedestrian_count == 0:
        if not IS_LEARNING:
            print("end")
            break
        w.remove_pedestrians()
        w.add_pedestrians(ped_init_data)
        print("added pedestrians")
        w.prepare_world()
        learning_map.reset_map()






#learning_map.save_data("learningdata")
#learning_map.save_model(f"version-{VERSION}")
print('done')
