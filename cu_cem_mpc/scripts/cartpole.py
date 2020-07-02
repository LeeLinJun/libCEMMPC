from math import pi
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches

class Cartpole:
    '''
    Cart Pole implemented by pytorch
    '''
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    #  Height of the cart
    H = 0.5

    STATE_X = 0
    STATE_V = 1
    STATE_THETA = 2
    STATE_W = 3
    CONTROL_A = 0

    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2


    def enforce_bounds(self, temp_state):
        if temp_state[0] < self.MIN_X:
               temp_state[0] = self.MIN_X
        elif temp_state[0] > self.MAX_X:
               temp_state[0]=self.MAX_X
        if temp_state[1] < self.MIN_V:
                temp_state[1] = self.MIN_V
        elif temp_state[1] > self.MAX_V:
                temp_state[1] = self.MAX_V
        if temp_state[2] < -np.pi:
                temp_state[2] += 2 * np.pi
        elif temp_state[2] > np.pi:
                temp_state[2] -= 2 * np.pi
        if temp_state[3] < self.MIN_W:
                temp_state[3] = self.MIN_W
        elif temp_state[3] > self.MAX_W:
                temp_state[3] = self.MAX_W
        return temp_state


    def propagate(self, start_state, control, num_steps, integration_step):
        temp_state = start_state.copy()
        for _ in range(num_steps):
            deriv = self.update_derivative(temp_state, control)
            temp_state[0] += integration_step * deriv[0]
            temp_state[1] += integration_step * deriv[1]
            temp_state[2] += integration_step * deriv[2]
            temp_state[3] += integration_step * deriv[3]
            temp_state = self.enforce_bounds(temp_state).copy()
        return temp_state

    def visualize_point(self, state):
        x2 = state[self.STATE_X] + (self.L) * np.sin(state[self.STATE_THETA])
        y2 = -(self.L) * np.cos(state[self.STATE_THETA])
        return state[self.STATE_X], self.H, x2, y2

    def update_derivative(self, state, control):
        '''
        Port of the cpp implementation for computing state space derivatives
        '''
        I = self.I
        L = self.L
        M = self.M
        m = self.m
        g = self.g
        #  Height of the cart
        deriv = state.copy()
        temp_state = state.copy()
        _v = temp_state[self.STATE_V]
        _w = temp_state[self.STATE_W]
        _theta = temp_state[self.STATE_THETA]
        _a = control[self.CONTROL_A]
        mass_term = (self.M + self.m)*(self.I + self.m * self.L * self.L) - self.m * self.m * self.L * self.L * np.cos(_theta) * np.cos(_theta)

        deriv[self.STATE_X] = _v
        deriv[self.STATE_THETA] = _w
        mass_term = (1.0 / mass_term)
        deriv[self.STATE_V] = ((I + m * L * L)*(_a + m * L * _w * _w * np.sin(_theta)) + m * m * L * L * np.cos(_theta) * np.sin(_theta) * g) * mass_term
        deriv[self.STATE_W] = ((-m * L * np.cos(_theta))*(_a + m * L * _w * _w * np.sin(_theta))+(M + m)*(-m * g * L * np.sin(_theta))) * mass_term
        return deriv

    def get_state_bounds(self):
        return [(self.MIN_X, self.MAX_X),
                (self.MIN_V, self.MAX_V),
                (self.MIN_THETA, self.MAX_THETA)
                (self.MIN_W, self.MAX_W)]

    def get_control_bounds(self):
        return None


class Cartpole_Visualizer:
    def __init__(self, model):
        self.model = model
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 2, 1)
        self.bx = self.fig.add_subplot(1, 2, 2)
        plt.show(block=False)
        self.count = 0

    def current_state(self, x, goal_state, curr_color='g'):
        self.ax.cla()
        _, _, xg, yg = self.model.visualize_point(goal_state)
        x1, y1, x2, y2 = self.model.visualize_point(x)
        self.ax.plot([x1]+[x2], [0]+[y2], color='gray')
        rect = patches.Rectangle((x1-2, y1), 5, -1,linewidth=1,edgecolor='skyblue',facecolor='skyblue')
        self.ax.add_patch(rect)

        # self.ax.scatter([x1], [y1], color='gray', s=30)
        self.ax.scatter([x2], [y2], color=curr_color, s=5)
        self.ax.scatter([xg], [yg], color='r', s=10)
        # self.ax.set_xlim(self.model.MIN_X, self.model.MAX_X)
        self.ax.set_xlim(-30, 30)
        self.ax.set_ylim(-20, 20)


        # self.ax.set_ylim(self.model.MIN_X, self.model.MAX_X)
        # self.ax.axis('equal')

        self.bx.scatter(x[0], x[2], c=curr_color)
        self.bx.scatter(goal_state[0], goal_state[2], c='r')
        self.bx.set_xlim(self.model.MIN_X, self.model.MAX_X)
        self.bx.set_ylim(-np.pi, np.pi)
        plt.savefig('tmp/'+str(self.count)+'.png')

        plt.pause(1e-6)
        self.count += 1



if __name__ == '__main__':
    model = Cartpole()
    dt = 2e-3
    # state = [-2.55734286,  -1.6512908 ,  -2.85039622,  -0.36047371]
    # goal = [0, 0, 0, 0]
    state = [-20, 0, 0, 0]
    goal = [20, 0, np.pi, 0]
    control = [1528.527933,1071.257315,1690.888528,892.980754,913.062267,706.518236,432.269678,-759.486078,206.395824,111.712993,-559.096808,-417.062751,-666.455433,-291.179806,-719.114127,-360.195079,169.571201,45.629374,-203.466431,-3.529200,-5.745073,274.037242,-236.512889,-168.371267,1.125498,-352.782362,-254.367856,217.973237,128.599666,-529.267795,-420.118403,-51.832802,90.799409,-105.503347,46.543540,-844.609895,-191.704195,-306.227796,-156.980133,-972.506407,65.903373,-310.277340,90.257000,-786.126782,-887.240688,-610.352290,-715.966515,-765.088383,-1305.327432,-459.636611,-736.527846,-512.530792,227.972314,-65.709463,890.294796,1147.911152,47.029616,674.945945,63.142092,203.088144,50.137423,-344.246384,-304.211486,-760.075975,]
    time = [0.061380,0.067850,0.073789,0.071681,0.053965,0.037677,0.032908,0.035534,0.034671,0.033108,0.053915,0.027878,0.050382,0.066182,0.036703,0.033827,0.027406,0.024166,0.024028,0.017269,0.018329,0.046596,0.067693,0.039558,0.039372,0.046968,0.032981,0.029011,0.036220,0.037045,0.050335,0.039705,0.075353,0.052319,0.025809,0.008793,0.041411,0.062102,0.025299,0.048867,0.048592,0.046602,0.036482,0.052758,0.072661,0.051659,0.044531,0.053408,0.026864,0.028405,0.083563,0.098236,0.080577,0.064565,0.057334,0.086128,0.064082,0.063243,0.048690,0.113408,0.076016,0.054812,0.044764,0.053998,]
    viz = Cartpole_Visualizer(model)
    for i in range(control.__len__()):
        state = model.propagate(state, [control[i]], int(time[i]/dt), dt)
        viz.current_state(state, goal)
 