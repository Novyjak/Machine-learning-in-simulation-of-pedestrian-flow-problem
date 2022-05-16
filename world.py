from typing import Iterable
from enum import IntEnum
import numpy as np
import worlditems as wi
import math
import random
#from worlditems import WorldItem,BoundaryItem, BoundaryType
import learning as lern
import tensorflow as tf

class CType(IntEnum):
    """
    Enum for type of the cell. Only three types of the cell are available
    """
    STYPE = 1  # simulated cell
    BTYPE = 2  # boundary cell
    ETYPE = 3  # empty cell
    RTYPE = 4  #removed cell, invisible


class PedestrianWorld:

    def __init__(self, width: int, height: int, cell_x_size = 0.4, cell_y_size = 0.4):
        assert width > 0, f"World width must be greater than zero. Got {width}."
        assert height > 0, f"World height must be greater than zero. Got {height}."
        self._width = width
        self._height = height
        self._holes = set()  # set of nonsimulated entries
        self._boundary = set()  # set of all boundaries (outflow and wall)
        self._outflow_boundary_indices = set()  # set of  outflow cell indices
        self._pedestrians = set()  # set of pedestrians cells
        self._wdata = np.zeros((height, width), dtype=int) # world data area
        self._rhodata = np.ndarray((height,width))
        self._phi = np.ndarray((height, width)) #matrix for the potential
        self._future = -1 *np.ones((height, width)) # matice obsahujici budouci obsazeni chodci
        self._cell_x_size = cell_x_size #real size of cell in x direction
        self._cell_y_size = cell_y_size #real size of cell in y direction
        self._p_average = np.zeros((3,3))
        self._MRHO = 0 #testovaci max rho
        self._trajectories = {}
        print(f"World created: {width}x{height}")

    def _add_boundary_item(self, item):
        """
        Adds one boundary item into the world.
        :param item:
        """
        self._boundary.add(item) #all boundary to set of boundaries
        if item.boundary_type == wi.BoundaryType.OUT: #special treatment of  outflow
            self._outflow_boundary_indices.add((item.y_position, item.x_position))
        item.register_to_world(CType.BTYPE) #register boundary to the world


    def _add_hole_item(self, item):
        """Adds one non simulated item (hole) into the world.
        This method only encapsulates adding to the set with non simulated items.
        :param item:
        """
        self._holes.add(item)
        item.register_to_world(CType.ETYPE)  # register hole into the world

    def _add_simulated_item(self, item):
        """
        Adds simulated item into the world.
        :param item:
        """
        self._pedestrians.add(item)
        item.register_to_world(CType.STYPE)  # register sim. cell into the world
        if item.tracked:
            #print("TRACKED")
            self._trajectories[item.id] = []
            self._trajectories[item.id].append((item.x_position, item.y_position))

    def add_boundary(self, boundary: Iterable[wi.BoundaryItem]):
        """Adds collection of the boundary items into the world."""
        for item in boundary:
            assert type(item) is wi.BoundaryItem, f"BoundaryItems expected."
            self._add_boundary_item(item)

    def add_hole(self, holes : Iterable[wi.NonSimulatedItem]):
        """Adds collection of the holes items to the world"""
        for item in holes:
            assert type(item) is wi.NonSimulatedItem, f"NonSimulatedItems expected."
            self._add_hole_item(item)

    def add_pedestrians(self, pedestrians: Iterable[wi.SimulatedItem]):
        """
        Adds collection of the pedestrian into the world
        :param pedestrians:
        :return: None
        """
        for item in pedestrians:
            assert type(item) is wi.SimulatedItem, f"SimulatedItems expected."
            self._add_simulated_item(item)
    
    def is_cell_free(self, x, y):
        """
        Checks if cell in world is not boundary or empty cell type.
        :param x: x coordinate in the world
        :param y: y coordinate in the world
        :return: boolean
        """
        cell_value = self._wdata[x, y]
        if cell_value == CType.ETYPE or cell_value == CType.BTYPE:
            return False
        else:
            return True

    def remove_pedestrians(self):
        self._pedestrians = set()
        self._future.fill(-1)
        for bound in self._boundary:
            if (bound.y_position, bound.x_position) not in self._outflow_boundary_indices:
                self._future[bound.y, bound.x] = -2
            else:
                pass
                # print("boundary")
        for hole in self._holes:
            self._future[hole.y, hole.x] = -2

    def prepare_world(self):
        """
        Initialize world for simulation:
        1 ] Set all empty cells as simulated cell with empty flag.
        :return: None
        """
        for irow,icol in np.ndindex(self._wdata.shape):
            if self._wdata[irow, icol] == 0:
                self._wdata[irow, icol] = CType.STYPE
        self._future.fill(-1) #set future position to default
        for bound in self._boundary:
            if (bound.y_position, bound.x_position) not in self._outflow_boundary_indices:
                self._future[bound.y, bound.x] = -2
            else:
                pass
                #print("boundary")
        for hole in self._holes:
            self._future[hole.y, hole.x] = -2

    def create_circle_hole(self, xcoord, ycoord, radius):
        """
        Create circular hole into the world
        :param xcoord: x-coordinate of the hole center
        :param ycoord: y-coordinate of the hole center
        :param radius:
        :return:
        """
        r2 = radius**2
        for irow,icol in np.ndindex(self._wdata.shape):
            centerx = (icol-1)*0.4 + 0.2 #recompute index to real coordinates
            centery = (irow-1)*0.4 + 0.2 #recompute index to real coordinates
            if (centerx-xcoord)**2+(centery-ycoord)**2 <= r2:
                self._add_hole_item(wi.NonSimulatedItem(icol, irow))

    def create_rectangle_hole(self, x1coord, y1coord, x2cord, y2cord):
        """
        Create square hole into the world
        :param x1coord: x-coordinate of the left top corner
        :param y1coord: y-coordinate of the left top corner
        :param x2coord: x-coordinate of the right bottom corner
        :param y2coord: y-coordinate of the right bottom corner
        :return:
        """
        for irow,icol in np.ndindex(self._wdata.shape):
            realx = (icol-1)*0.4 + 0.2 #recompute index to real coordinates
            realy = (irow-1)*0.4 + 0.2 #recompute index to real coordinates
            if x1coord <= realx <= x2cord and y1coord <= realy <= y2cord:
                self._add_hole_item(wi.NonSimulatedItem(icol, irow))

    def create_hole(self, xcoord, ycoord):
        """
        Set item on given coords as non simulated hole item
        :param xcoord:
        :param ycoord:
        :return:
        """
        px = int(xcoord/0.4)
        py = int(ycoord/0.4)
        self._add_hole_item(wi.NonSimulatedItem(px,py))

    def create_hole_on_index(self, xpos, ypos, colored = True):
        """
        Indices are input
        :param xpos:
        :param ypos:
        :return:
        """
        self._add_hole_item(wi.NonSimulatedItem(xpos, ypos, colored))

    def get_pedestrians(self):
        return self._pedestrians.copy()


    def move(self, learning_map: lern.LearningMap, is_learning):
        """
        Compute possible future position of each pedestrian.
        :return:
        """
        def transform(choice, xp, yp ):
            if choice == 0:
                return xp-1, yp - 1
            if choice == 1:
                return xp, yp - 1
            if choice == 2:
                return xp+1, yp - 1
            if choice == 3:
                return xp - 1, yp
            if choice == 4:
                return xp, yp
            if choice == 5:
                return xp + 1, yp
            if choice == 6:
                return xp - 1, yp + 1
            if choice == 7:
                return xp, yp + 1
            if choice == 8:
                return xp + 1, yp + 1


        out = {}
        probabilities = {}

        for i in range(len(self._future)):
            self._future[i, self._future[i] > -2] = 0

        for ped in self._pedestrians:
            learning_map.add_pedestrian(ped.x_position, ped.y_position)

        for item in self._pedestrians: #inicializace pohybu
            item.moved = False #nastav ze se jeste nehnul
            x = item.x_position
            y = item.y_position
            p = item.P.flatten() #rozbal pole


            if(is_learning):
                #######################
                learning_map.frame_count += 1
                learning_map.episode_step_count += 1

            item.state = learning_map.ped_state(x, y)
            #######################



            self._future[y, x] = -1 #aktualni pozice jsou obsazeny


            if is_learning and (learning_map.frame_count < learning_map.epsilon_random_frames or learning_map.epsilon > np.random.rand(1)[0]):
                #print("frame count: " + str(learning_map.frame_count) + ", random frames: " + str(learning_map.epsilon_random_frames))

                # Take random action
                #item.choice = np.random.choice(9, p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.9,0.0125,0.0125,0.0125])
                item.choice = np.random.choice(9)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(item.state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = learning_map.model(state_tensor, training=False)
                #print("action_probs: " + str(action_probs))
                # Take best action
                tmp_weights = np.ones(9, dtype=int)
                tmp_choice = tf.argmax(action_probs[0]).numpy()
                tmp_weights[tmp_choice] = 80
                item.choice = random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8], weights=tmp_weights, k=1)[0]  # hod kostkou

#####################
            if is_learning:
                learning_map.epsilon -= learning_map.epsilon_interval / learning_map.epsilon_greedy_frames
                learning_map.epsilon = max(learning_map.epsilon, learning_map.epsilon_min)
            choice = item.choice
#####################

            item.moved = False #nuluj si pohyb
            item.move_position = transform(choice, x, y)  # spocti novou polohu
            item.move_probability = p[choice] #vem si pouzitou pravdepodobnost
            probabilities[item.id] = p[choice] #zapamatuj si ji
            out[choice] = out.get(choice, 0) + 1

        while True:
            nonmoved = [item for item in self._pedestrians if not item.moved]
            #print(f"Size set {len(nonmoved)}")
            for item in  nonmoved: #pres vsechny nehnute
                (xn, yn) = item.move_position #vem si predpokladanou novou polohu
                if self._future[yn, xn] > -1: #je aktualne volno, ale muze se tam nekdo presunout
                    if self._future[yn,xn] == 0: #nikdo tam zatim neni
                        self._future[yn, xn]  = item.id #uloz tam pozici
                    #elif probabilities[self._future[yn, xn]] < item.move_probability:
                        #self._future[yn, xn] = item. id #uloz si tam nasledujici

            moved_count = 0

            for item in nonmoved:  # pres vsechny nehnute
                (xn,yn) = item.move_position
                if self._future[yn,xn] == item.id: #kdyz tam ma byt on
                    self._future[yn,xn] = -1 # zaber si ji
                    self._future[item.y, item.x] = 0 # uvolni starou
                    item.moved=True
                    item.old_x = item.x_position
                    item.old_x = item.y_position
                    item.x_position = xn
                    item.y_position = yn
                    moved_count+=1 #zvys pocet pohnutych
            #print(f"Moved {moved_count}")
            if moved_count ==0 :
                break
        
        if is_learning:
            for ped in self._pedestrians:
                if (ped.y_position,ped.x_position) in self._outflow_boundary_indices:
                    ped.done = 1
                else:
                    learning_map.add_future_pedestrian(ped.x_position, ped.y_position)
            for item in self._pedestrians:
                x = item.old_x
                y = item.old_x
                reward = learning_map.get_reward(x, y, item.move_position[0], item.move_position[1])
                learning_map.action_history.append(item.choice)
                learning_map.state_history.append(item.state)
                learning_map.rewards_history.append(reward)
                learning_map.state_next_history.append(learning_map.ped_future_state(item.x_position, item.y_position))
                learning_map.done_history.append(item.done)

                learning_map.episode_reward.append(reward)
            if len(learning_map.done_history) > learning_map.batch_size:
                #print("relearning")
                indices = np.random.choice(range(len(learning_map.done_history)), size=learning_map.batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([learning_map.state_history[i] for i in indices])
                state_next_sample = np.array([learning_map.state_next_history[i] for i in indices])
                rewards_sample = [learning_map.rewards_history[i] for i in indices]
                action_sample = [learning_map.action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(learning_map.done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = learning_map.model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + learning_map.gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                #print("updated q_values: " + str(updated_q_values))

                masks = tf.one_hot(action_sample, learning_map.num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = learning_map.model(state_sample)
                    #print("q_values: " + str(q_values))

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    #print("q_action: " + str(q_action))

                    # Calculate loss between new Q-value and old Q-value
                    loss = learning_map.loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, learning_map.model.trainable_variables)
                    learning_map.optimizer.apply_gradients(zip(grads, learning_map.model.trainable_variables))

                if len(learning_map.rewards_history) > learning_map.max_memory_length:
                    del learning_map.rewards_history[:1]
                    del learning_map.state_history[:1]
                    del learning_map.state_next_history[:1]
                    del learning_map.action_history[:1]
                    del learning_map.done_history[:1]


        moved = [item for item in self._pedestrians if item.moved]
        for ped in moved:
            if ped.tracked:
                self._trajectories[ped.id].append((ped.x_position, ped.y_position))
            if (ped.y_position,ped.x_position) in self._outflow_boundary_indices:
                print(f"Removed {ped}")
                self._pedestrians.remove(ped)
        return out

    @property
    def world_width(self)->int:
        return self._width

    @property
    def world_height(self) -> int:
        return self._height

    @property
    def world_holes(self) -> set:
        return self._holes

    @property
    def world_boundary(self) -> set:
        return self._boundary

    @property
    def world_pedestrians(self) -> set:
        return self._pedestrians

    @property
    def average_P(self) -> np.array :
       return self._p_average/len(self._pedestrians)


    @property
    def pedestrian_count(self):
        return len(self._pedestrians)

    @property
    def trajectories(self):
        return self._trajectories
