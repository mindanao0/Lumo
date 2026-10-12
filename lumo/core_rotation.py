"""Quaternion rotation for the central cloud and an unrestricted drag trackball."""
import math
import numpy as np


def multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw))


class CoreRotation:
    def __init__(self):
        self.reset()

    def reset(self):
        self.quaternion = np.array((1., 0., 0., 0.))
        self.auto = True
        self.drag_start = None

    def step(self, dt, clock):
        if dt <= 0 or not self.auto or self.drag_start is not None:
            return
        velocity = np.array((.075+.025*math.sin(clock*.11),
                             .11+.03*math.cos(clock*.09), .045))
        speed = np.linalg.norm(velocity)
        half = speed*dt*.5
        delta = np.concatenate(([math.cos(half)], velocity/speed*math.sin(half)))
        self.quaternion = multiply(delta, self.quaternion)
        self.quaternion /= np.linalg.norm(self.quaternion)

    @property
    def matrix(self):
        w, x, y, z = self.quaternion
        return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                         (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                         (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))), dtype=np.float32)

    @staticmethod
    def project(pos, width, height, scale=1.):
        x = (pos[0]-(width*.5-height*.0125*scale))/(height*.29*scale)
        y = (height*.5-pos[1])/(height*.29*scale)
        length = x*x+y*y
        if length > 1:
            return np.array((x, y, 0.))/math.sqrt(length)
        return np.array((x, y, math.sqrt(1-length)))

    def begin(self, pos, width, height, scale=1.):
        self.drag_start = self.project(pos, width, height, scale)
        self.drag_orientation = self.quaternion.copy()

    def drag(self, pos, width, height, scale=1.):
        if self.drag_start is None:
            return
        target = self.project(pos, width, height, scale)
        dot = float(np.clip(np.dot(self.drag_start, target), -1, 1))
        delta = np.concatenate(([1+dot], np.cross(self.drag_start, target)))
        if np.linalg.norm(delta) < 1.e-8:
            axis = np.cross(self.drag_start, (1., 0., 0.))
            if np.linalg.norm(axis) < 1.e-8:
                axis = np.cross(self.drag_start, (0., 1., 0.))
            delta = np.concatenate(([0.], axis/np.linalg.norm(axis)))
        else:
            delta /= np.linalg.norm(delta)
        self.quaternion = multiply(delta, self.drag_orientation)
        self.quaternion /= np.linalg.norm(self.quaternion)
