import worlditems as wi
import world as ww
import numpy as np
from PIL import  Image, ImageDraw #for graphics
from PIL import PSDraw

class WorldVisualisator:
    def __init__(self, world):
        self._world = world

    def DrawWorld(self, size = (640,480), bcolor = 'blue', obcolor = 'green',hcolor = 'blue', lcolor = 'black', pedcolor = 'red', pedcolors = ['aqua', 'violet', 'black', 'green']):
        dx = size[0] / self._world.world_width
        dy = size[1] / self._world.world_height
        wim = Image.new('RGB', size,(255,255,255))
        imw = ImageDraw.Draw(wim)

        for boundary in self._world.world_boundary:
             xp = boundary.x_position
             yp = boundary.y_position
             if boundary.boundary_type != wi.BoundaryType.OUT:
                imw.rectangle([(xp*dx,yp*dy),((xp+1)*dx,(yp+1)*dy)], bcolor)
             else:
                imw.rectangle([(xp * dx, yp * dy), ((xp + 1) * dx, (yp + 1) * dy)], obcolor)


        for hole in self._world.world_holes:
            if hole.colored:
                color = hcolor
            else:
                color = 'pink'
            xp = hole.x_position
            yp = hole.y_position
            imw.rectangle([(xp * dx, yp * dy), ((xp + 1) * dx, (yp + 1) * dy)], color)

        cindex = 0
        for ped in self._world.world_pedestrians:
            if not ped.tracked:
                xp = ped.x_position
                yp = ped.y_position
                imw.rectangle([(xp * dx, yp * dy), ((xp + 1) * dx, (yp + 1) * dy)], pedcolor)
            else:
                 xp = ped.x_position
                 yp = ped.y_position
                 imw.rectangle([(xp * dx, yp * dy), ((xp + 1) * dx, (yp + 1) * dy)], pedcolors[cindex])
                 cindex = cindex+1
            if(ped.choice == 0):
                xl = -1
                yl = -1
            elif(ped.choice == 1):
                xl = 0
                yl = -1
            elif(ped.choice == 2):
                xl = 1
                yl = -1
            elif(ped.choice == 3):
                xl = -1
                yl = 0
            elif(ped.choice == 4):
                xl = 0
                yl = 0
            elif(ped.choice == 5):
                xl = 1
                yl = 0
            elif(ped.choice == 6):
                xl = -1
                yl = 1
            elif(ped.choice == 7):
                xl = 0
                yl = 1
            elif(ped.choice == 8):
                xl = 1
                yl = 1
            shape = [((xp * dx)+0.5*dx, (yp * dy)+0.5*dy), ((xp * dx)+0.5*dx+(0.5*dx*xl), (yp * dy)+0.5*dy+(0.5*dy*yl))]
            imw.line(shape, fill ="black", width = 0)
            



        for y in range(self._world.world_height):
                imw.line([(0,y*dy),(size[0],y*dy)], lcolor)
        for x in range(self._world.world_width):
                imw.line([(x*dx,0),(x*dx,size[1])], lcolor)
        return wim

    def DrawWorldTrajectory(self, size=(640, 480), bcolor='blue', obcolor='green', hcolor='blue', lcolor='black', pedcolors = ['aqua', 'violet', 'black', 'green']):
        dx = size[0] / self._world.world_width
        dy = size[1] / self._world.world_height
        wim = Image.new('RGB', size, (255, 255, 255))
        imw = ImageDraw.Draw(wim)

        for boundary in self._world.world_boundary:
            xp = boundary.x_position
            yp = boundary.y_position
            if boundary.boundary_type != wi.BoundaryType.OUT:
                imw.rectangle([(xp * dx, yp * dy), ((xp + 1) * dx, (yp + 1) * dy)], bcolor)
            else:
                imw.rectangle([(xp * dx, yp * dy), ((xp + 1) * dx, (yp + 1) * dy)], obcolor)

        for hole in self._world.world_holes:
            if hole.colored:
                color = hcolor
            else:
                color = 'pink'
            xp = hole.x_position
            yp = hole.y_position
            imw.rectangle([(xp * dx, yp * dy), ((xp + 1) * dx, (yp + 1) * dy)], color)

        icolor = 0
        for id, trajectory in  self._world.trajectories.items():
            print(type(trajectory))
            for xp,yp in trajectory:
                imw.rectangle([(xp * dx, yp * dy), ((xp + 1) * dx, (yp + 1) * dy)], pedcolors[icolor])
            icolor += 1
        for y in range(self._world.world_height):
                imw.line([(0,y*dy),(size[0],y*dy)], lcolor)
        for x in range(self._world.world_width):
                imw.line([(x*dx,0),(x*dx,size[1])], lcolor)
        return wim