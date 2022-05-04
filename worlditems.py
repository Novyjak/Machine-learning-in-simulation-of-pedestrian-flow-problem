import inspect
from typing import Tuple
import world
import math
from enum import IntEnum
import numpy as np

class MetaWorldItem(type):
    def __new__(mc1, name, bases, nmspc):
        nmspc.update({'world': MetaWorldItem.world})
        return super(MetaWorldItem, mc1).__new__(mc1, name, bases, nmspc)

    @property
    def world(cls):
        if not inspect.isclass(cls):
            cls = type(cls)
        return cls._world

    @world.setter
    def world(cls, value):
        assert isinstance(value, world.PedestrianWorld), f"Instance of PedestrianWorld expected. Got {type(value)}"
        if not inspect.isclass(cls):
            cls = type(cls)
        cls._world = value
        print(f"World item setter calling - world object {value}")


class BoundaryType(IntEnum):
    """
    Enum for type of the boundary
    """
    WALL = 1 #fixed wall
    OUT = 2 #outflow

class ItemType(IntEnum):
    """
    Enum for type of the simulated cell
    """
    PED = 1
    EMPTY = 2


class WorldItem(metaclass=MetaWorldItem):
    """Base class for world items"""

    _world = None
    _index = 0

    def __init__(self, x: int, y: int):
        """
        Initialize and check world item, i.e. cell in matrix
        """
        assert x >= 0, f"World x-coord must be greater or equal to zero. Got {x}."
        assert y >= 0, f"World y-coord must be greater or equal to zero. Got {y}."
        assert type(self).world is not None, f"Unitialized world association."
        WorldItem._index+=1 #inkrementuj index
        self.x = x
        self.y = y
        self._index = WorldItem._index #pouzij rostouci index objektu

    def register_to_world(self, cell_value : int):
        """
        Register item to the world
        :param cell_value:
        :return:
        """
        type(self)._world._wdata[self.y, self.x] = cell_value



    @property
    def x_position(self) -> int:
        """
        World coordinate, ie. coordinate in matrix column
        :return:
        """
        return self.x

    @x_position.setter
    def x_position(self, x):
        self.x = x

    @property
    def y_position(self) -> int:
        """
        World coordinate, ie. coordinate in matrix row
        :return:
        """
        return self.y

    @y_position.setter
    def y_position(self, y):
        self.y = y

    @property
    def center(self)->Tuple[float,float]:
        """
        Return tuple of coordinates of the centroid of the cell
       :return:
        """
        dx =  type(self)._world._cell_x_size
        dy =  type(self)._world._cell_y_size
        return ((self.x+0.5)*dx, (self.y+0.5)*dy)

    @property
    def id(self):
        """
        Return identification number of instance
        :return:
        """
        return self._index

    def __str__(self):
        return f"World item on position ({self.x},{self.y})"


class NonSimulatedItem(WorldItem):
    """Items of this type are removed from simulation."""

    def __init__(self, x: int, y: int, colored = True):
        super().__init__(x, y)
        self._colored = colored

    def __str__(self):
        return f"Non simulated item on position ({self.x},{self.y})"

    @property
    def colored(self):
        return self._colored


class BoundaryItem(WorldItem):
    """Items of this type create world boundary. """

    def __init__(self, x: int, y: int, btype=BoundaryType.WALL):
        assert isinstance(btype, BoundaryType), f"Boundary type must by specified. Got{type(btype)}"
        super().__init__(x, y)
        self.btype = btype  # set type of boundary: outlet, wall,

    @property
    def boundary_type(self) -> BoundaryType:
        """Property getter with type of boundary"""
        return self.btype

    def __str__(self):
        return f"Boundary item on position ({self.x},{self.y}) type of {self.btype}"


class SimulatedItem(WorldItem):
    """Simulated items of the world"""
    def __init__(self, x : int, y : int, ctype = ItemType.EMPTY, tracked=False):
        assert isinstance(ctype, ItemType), f"Type of the item must be specified. Got{type(ctype)}"
        super().__init__(x, y)
        self.ctype = ctype  # set type of the item: PED, EMPTY
        self._EP_matrix = np.ndarray((3, 3)) #pole pro pravdepodobnosti
        self._move_probability = 0 # pravdepodobnost daneho prechodu
        self._move_position = (x,y) #souradnice kam se bude pohybovat
        self._moved = False
        self._tracked = tracked #zda je trackovana

        self.state = []
        self.choice = 4
        self.next_next = []
        self.old_x = x
        self.old_y = y
        self.done = 0



    @property
    def tracked(self):
        return self._tracked

    @property
    def item_type(self) -> BoundaryType:
        """Property getter with type of cell"""
        return self.ctype

    @property
    def P(self):
        """
        Probability matrix, no setter - all is referenced
        :return:
        """
        return self._EP_matrix

    @property
    def move_probability(self):
        """
        Get probability of move
        :return:
        """
        return self._move_probability

    @move_probability.setter
    def move_probability(self, p):
        """
        Set probability of move
        :param p:
        :return:
        """
        self._move_probability = p

    @property
    def move_position(self):
        """
        Tuple with selected position(x,y)
        :return:
        """
        return self._move_position

    @move_position.setter
    def move_position(self, position):
        """Setter for possible position"""
        self._move_position = position

    @property
    def moved(self):
        return self._moved

    @moved.setter
    def moved(self, decision):
        self._moved = decision



    def __str__(self):
        return f"Simulation item on position ({self.x},{self.y}) type of {self.ctype}"



