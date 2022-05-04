from world import *
from worlditems import *
from visualisator import *
import random
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import learning as lern
import os


VERSION = 4
EPISODE = 5
FILE_NAME = f"./images/T/Version{VERSION}/"
IS_LEARNING = True
LOADED_VERSION = 2
LOADED_MODEL = f"version_{LOADED_VERSION}-episode_{EPISODE}"

#domain definiton
REAL_W = 28 #in metres
REAL_H = 22 #in metres
cell_size = 0.4
w_width = int(REAL_W/cell_size+2) #add 2 cels for left and right boundary
w_height = int(REAL_H/cell_size+2) # add 2 cels for top and bottom boundary
im_width = 800
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


ped_init_data = init_condition(lambda x, y: x >= 10/0.4 and x <= 18/0.4 and y >=10/0.4, rho = 2)
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
    learning_map.model_target.set_weights(learning_map.load_model(LOADED_MODEL))


for step in range(1001):
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
    imped.save(FILE_NAME + f'steps/img-version-{VERSION}-episode-{EPISODE}-step-{step}.png')

    if step % 100 == 0:
        #for item in sorted(minfo, key = minfo.get):
        #    print(f'{item}:-> {minfo[item]}')
        images[0].save(FILE_NAME + f'img-version-{VERSION}-episode-{EPISODE}-step-{step}.gif',
                       save_all=True,
                       append_images=images[1:],
                       duration=200,
                       loop=0)
        images = []
        if step % 2000 == 0 and False: # Ponecháno, v případě, že by jsem v budoucnu chtěl upravit
            w.remove_pedestrians()
            w.add_pedestrians(init_condition(lambda x, y: x <= 20 / 0.4, rho=1))
    if w.pedestrian_count == 0:
        if not IS_LEARNING:
            print("end")
            break
        w.remove_pedestrians()
        w.add_pedestrians(ped_init_data)
        learning_map.reset_map()
        print("added pedestrians")






#learning_map.save_data("learningdata")
#learning_map.save_model(f"version-{VERSION}")
print('done')
