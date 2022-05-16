from world import *
from worlditems import *
from visualisator import *
import random
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import learning as lern
import os

def init_condition(condition, w_height, w_width, cell_size, rho = 2, random_seed = 1):
    #REMARK nastaveno seed
    data = []
    bound = rho*cell_size*cell_size
    for y in range(1, w_height-1):
        for x in range(1,w_width):
            random.seed = random_seed #!!!!!!!
            if condition(x,y) and random.random() <= bound:
                data.append(SimulatedItem(x, y, ItemType.PED))
    print(len(data))
    return data

def model_simulate(FILE_NAME, VERSION, EPISODE, IS_LEARNING, SAVE_MODEL, w_height, w_width,  w, im_width, hw_factor, LOADED_MODEL = None):
    isExist = os.path.exists(FILE_NAME)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(FILE_NAME + "steps/")

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
            
            if SAVE_MODEL:
                learning_map.save_model(f"version_{VERSION}-episode_{learning_map.episode_count}")

            learning_map.episode_count += 1

        learning_map.reset_map()

        imped = wvisu.DrawWorld(size=(im_width, im_width // hw_factor))
        images.append(imped)
        imped.save(FILE_NAME + f'steps/img-version-{VERSION}-episode-{EPISODE}-step-{step}.png')

        if step % 100 == 0:
            #for item in sorted(minfo, key = minfo.get):
            #    print(f'{item}:-> {minfo[item]}')
            images[0].save(FILE_NAME + f'gif-version-{VERSION}-episode-{EPISODE}-step-{step}.gif',
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
