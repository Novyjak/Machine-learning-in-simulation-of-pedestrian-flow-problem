import math
from enum import IntEnum
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
from datetime import date


class CellType(IntEnum):

    """
    Enum for type of the cell. Only three types of the cell are available
    """
    UTYPE = 0  # unseen cell
    PTYPE = 1  # pedestrian cell
    BTYPE = 2  # boundary cell
    ETYPE = 3  # empty cell
    OTYPE = 4  # out cell
    

VISION_RANGE = 10
OBSERVATION_SPACE = (2*VISION_RANGE +1, 2*VISION_RANGE +1, 1)
KNOW_EXIT = True


class LearningMap:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._mapItemsBase = np.full((width,height), self.add_ct(CellType.ETYPE))  # Pro uložení stěn a východů, které se němění
        self._mapItems = self._mapItemsBase.copy()  # Aktuální stav mapy (zdi + chodci)
        self._futureMap = self._mapItemsBase.copy()
        self._outBoundaries = []  # Je nutné vědět, kde jsou východy.

        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Epsilon greedy parameter
        self.epsilon_min = 0.05  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        self.epsilon_interval = (
                self.epsilon_max - self.epsilon_min
        )  # Rate at which to reduce chance of random action being taken
        self.batch_size = 10000  # Size of batch taken from replay buffer
        self.max_steps_per_episode = 10000

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.episode_reward = []
        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0
        self.episode_step_count = 0

        self.max_memory_length = 100000
        self.num_actions = 9
        self.loss_function = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        # Number of frames to take random action and observe output
        self.epsilon_random_frames = 0
        # Number of frames for exploration
        self.epsilon_greedy_frames = 1000.0

        self.model = self.create_q_model()
        self.model_target = self.create_q_model()
        self.model_target.set_weights(self.model.get_weights())

    def create_q_model(self):

        num_actions = 9
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=OBSERVATION_SPACE)

        # Convolutions on the frames on the screen
        convlayer1 = layers.Conv2D(32, 2, activation="relu")(inputs)
        poollayer1 = layers.MaxPool2D(pool_size=(2,2))(convlayer1)
        convlayer2 = layers.Conv2D(64, 2, activation="relu")(poollayer1)
        poollayer2 = layers.MaxPool2D(pool_size=(2,2))(convlayer2)
        #layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)


        layer4 = layers.Flatten()(poollayer2)

        #layer4_1 = layers.Dense(512, activation="relu")(layer4)

        layer5 = layers.Dense(64, activation="linear")(layer4)
        action = layers.Dense(num_actions, activation="linear")(layer5)

        return keras.Model(inputs=inputs, outputs=action)


    def ped_state(self, ped_x, ped_y):
        return_state = np.ones((2*VISION_RANGE+1, 2*VISION_RANGE+1))
        offsets = list(range(-VISION_RANGE, VISION_RANGE+1))
        out_closest = self._outBoundaries[0]
        out_len = self.compute_pythagoras(ped_x, ped_y, out_closest[0], out_closest[1])

        for i in offsets:
            for j in offsets:
                if i == 0 and j == 0:
                    return_state[i, j] = 0.9
                    # Pokus o ukázání, jakým směrem je nejbližší východ
                    """ for item in self._outBoundaries:
                        len = self.compute_pythagoras(ped_x, ped_y, item[0], item[1])
                        if len < out_len:
                            out_closest = item
                            out_len = len

                        out_vector_x = math.sqrt((out_closest[0] - ped_x) ** 2) / out_len
                        out_vector_y = math.sqrt((out_closest[1] - ped_y) ** 2) / out_len
                        center_number = 0.5 + (np.mod(np.degrees( np.arctan2(out_vector_x, out_vector_y) ), 360) / 720)
                        return_state[i, j] = center_number """
                # Přidej kontrolu, jestli aplikace nešahá mimo mapu
                if ped_x + i < 0 or ped_x + i >= self._width:
                    return_state[i, j] = self.add_ct(CellType.BTYPE)
                elif ped_y + j < 0 or ped_y + j >= self._height:
                    return_state[i, j] = self.add_ct(CellType.BTYPE)
                else:
                    return_state[i, j] = self._mapItems[i,j]
        return return_state

    def ped_future_state(self, ped_x, ped_y):
        return_state = np.ones((2*VISION_RANGE+1, 2*VISION_RANGE+1))
        offsets = list(range(-VISION_RANGE, VISION_RANGE+1))
        out_closest = self._outBoundaries[0]
        out_len = self.compute_pythagoras(ped_x, ped_y, out_closest[0], out_closest[1])

        for i in offsets:
            for j in offsets:
                if i == 0 and j == 0:
                    return_state[i, j] = 0.9
                    # Pokus o ukázání, jakým směrem je nejbližší východ
                    """ for item in self._outBoundaries:
                        len = self.compute_pythagoras(ped_x, ped_y, item[0], item[1])
                        if len < out_len:
                            out_closest = item
                            out_len = len

                        if out_len == 0:
                            out_len = 1

                        out_vector_x = math.sqrt((out_closest[0] - ped_x) ** 2) / out_len
                        out_vector_y = math.sqrt((out_closest[1] - ped_y) ** 2) / out_len
                        center_number = 0.5 + (np.mod(np.degrees( np.arctan2(out_vector_x, out_vector_y) ), 360) / 720)
                        return_state[i, j] = center_number """
                # Přidej kontrolu, jestli aplikace nešahá mimo mapu
                if ped_x + i < 0 or ped_x + i >= self._width:
                    return_state[i, j] = self.add_ct(CellType.BTYPE)
                elif ped_y + j < 0 or ped_y + j >= self._height:
                    return_state[i, j] = self.add_ct(CellType.BTYPE)
                else:
                    return_state[i, j] = self._futureMap[i,j]
        return return_state

    def get_reward(self, ped_x, ped_y, choice_x, choice_y):
        if self._mapItems[choice_x, choice_y] == self.add_ct(CellType.OTYPE):
            return 100000  # Pokud je vychod
        elif self._mapItems[choice_x, choice_y] == self.add_ct(CellType.BTYPE):
            return -200  # Pokud narazí do stěny
        elif self._mapItems[choice_x, choice_y] == self.add_ct(CellType.PTYPE):
            return -100  # Pokud narazí do chodce
        elif self._mapItems[choice_x, choice_y] == self.add_ct(CellType.ETYPE):
            if KNOW_EXIT: # Jestli chodci ví, jakým směrem je východ
                out_closest = self._outBoundaries[0]
                out_len = self.compute_pythagoras(ped_x, ped_y, out_closest[0], out_closest[1])
                for item in self._outBoundaries:
                    len = self.compute_pythagoras(ped_x, ped_y, item[0], item[1])
                    if len < out_len:
                        out_closest = item
                        out_len = len
                com_pos_x = abs(out_closest[0] - ped_x)
                com_pos_y = abs(out_closest[1] - ped_y)
                com_choice_x = abs(out_closest[0] - choice_x)
                com_choice_y = abs(out_closest[1] - choice_y)
                if com_choice_x < com_pos_x and com_choice_y < com_pos_y:
                    return 10  # jde směrem přímo k východu
                elif com_choice_x < com_pos_x or com_choice_y < com_pos_y:
                    return 0  # jde nepřímo k východu
                else:
                    return -10  # jde mimo východ
            else:
                return -10 
        else:
            print(self._mapItems[choice_x, choice_y])
            return 0  # pokud je něco jinak, může být problém a je lepší oznámit


    def add_boundary(self, x: int, y:int, btype = 1):
        if btype == 2:
            self._mapItemsBase[x, y] = self.add_ct(CellType.OTYPE)
            self._outBoundaries.append((x, y))
        else:
            self._mapItemsBase[x, y] = self.add_ct(CellType.BTYPE)

    def add_pedestrian(self, x, y):
        self._mapItems[x, y] = self.add_ct(CellType.PTYPE)

    def add_future_pedestrian(self, x, y):
        self._futureMap[x, y] = self.add_ct(CellType.PTYPE)

    def reset_map(self):
        self._mapItems = self._mapItemsBase.copy()
        self._futureMap = self._mapItemsBase.copy()

    def save_map(self, file):
        self._mapItems.to_pickle("map.pkl")

    def save_model(self, file):
        weights = self.model.get_weights()
        try:
            np.save(f"model/{file}" ,weights)
        except Exception as ex:
            print("Error during saving np data", ex)

    def load_model(self, file):
        try:
            weights = np.load(f"model/{file}.npy", allow_pickle=True)
        except Exception as ex:
            print("Error during saving np data", ex)

        return weights
        

    def save_version_info(self, file, version, is_learning, ped_number):
        tmpDict = {
            'version': version,
            'is_learning': is_learning,
            'vision_range': VISION_RANGE,
            'know_exit': KNOW_EXIT,
            'ped_number': ped_number,
            'gamma': self.gamma,
            'min_epsilon': self.epsilon_min,
            'batch_size': self.batch_size,
            'max_steps_episode': self.max_steps_per_episode,
            'max_memory_length': self.max_memory_length,
            'random_frames': self.epsilon_random_frames,
            'epsilon_greedy_frames': self.epsilon_greedy_frames,
            'date': str(date.today())
        }
        with open(file + "info.txt", 'w') as file:
            file.write(json.dumps(tmpDict))
        

        

    @property
    def mapItems(self):
        return self._mapItems.copy()

    # Při učení je lepší mít hodnoty v rozmezí 0 až 1, nebo -1 až 1.
    # Já používám 0 až 1.
    # Body na mapě jsou uložené v hodnotách v rozmezí <0,0.5)
    # Hodnoty <0.5,1> jsou ponechány pro směr východu
    # Při změně postupu stačí upravit funkci
    @staticmethod
    def add_ct(ctype: CellType):
        tmp = ctype/(len(CellType)*2)
        return tmp



    @staticmethod
    def compute_pythagoras(x_from, y_from, x_to=None, y_to=None):
        if x_to is None and y_to is None:
            x = x_from
            y = y_from
        else:
            x = x_to - x_from
            y = y_to - y_from
        return math.sqrt(x * x + y * y)

