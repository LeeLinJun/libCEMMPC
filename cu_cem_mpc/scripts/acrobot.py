
import numpy as np
from matplotlib import pyplot as plt

class Acrobot:
    '''
    Two joints pendulum that is activated in the second joint (Acrobot)
    '''
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 1.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    l = 1.0
    g = 9.81

    def propagate(self, start_state, control, num_steps, integration_step, viz=None):
        state = start_state
        for i in range(num_steps):
            state += integration_step * self._compute_derivatives(state, control)

            if state[0] < -np.pi:
                state[0] += 2*np.pi
            elif state[0] > np.pi:
                state[0] -= 2 * np.pi
            if state[1] < -np.pi:
                state[1] += 2*np.pi
            elif state[1] > np.pi:
                state[1] -= 2 * np.pi

            state[2:] = np.clip(
                state[2:],
                [self.MIN_V_1, self.MIN_V_2],
                [self.MAX_V_1, self.MAX_V_2])
            if viz is not None:
                viz.current_state(state)
        return state

    def visualize_point(self, state):
        x1 = self.LENGTH * np.cos(state[self.STATE_THETA_1] - np.pi / 2)
        x2 = x1 + self.LENGTH * np.cos(state[self.STATE_THETA_1] + state[self.STATE_THETA_2] - np.pi/2)
        y1 = self.LENGTH * np.sin(state[self.STATE_THETA_1] - np.pi / 2)
        y2 = y1 + self.LENGTH * np.sin(state[self.STATE_THETA_1] + state[self.STATE_THETA_2] - np.pi / 2)
        return x1, y1, x2, y2

    def _compute_derivatives(self, state, control):
        '''
        Port of the cpp implementation for computing state space derivatives
        '''
        theta2 = state[self.STATE_THETA_2]
        theta1 = state[self.STATE_THETA_1] - np.pi/2
        theta1dot = state[self.STATE_V_1]
        theta2dot = state[self.STATE_V_2]
        _tau = control[0]
        m = self.m
        l2 = self.l2
        lc2 = self.lc2
        l = self.l
        lc = self.lc
        I1 = self.I1
        I2 = self.I2

        d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * np.cos(theta2)) + I1 + I2
        d22 = m * lc2 + I2
        d12 = m * (lc2 + l * lc * np.cos(theta2)) + I2
        d21 = d12

        c1 = -m * l * lc * theta2dot * theta2dot * np.sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * np.sin(theta2))
        c2 = m * l * lc * theta1dot * theta1dot * np.sin(theta2)
        g1 = (m * lc + m * l) * self.g * np.cos(theta1) + (m * lc * self.g * np.cos(theta1 + theta2))
        g2 = m * lc * self.g * np.cos(theta1 + theta2)

        deriv = state.copy()
        deriv[self.STATE_THETA_1] = theta1dot
        deriv[self.STATE_THETA_2] = theta2dot

        u2 = _tau - 1 * .1 * theta2dot
        u1 = -1 * .1 * theta1dot
        theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
        theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
        deriv[self.STATE_V_1] = theta1dot_dot
        deriv[self.STATE_V_2] = theta2dot_dot
        return deriv

    def get_state_bounds(self):
        return [(self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_V_1, self.MAX_V_1),
                (self.MIN_V_2, self.MAX_V_2)]

    def get_control_bounds(self):
        return [(self.MIN_TORQUE, self.MAX_TORQUE)]

    # def distance_computer(self):
    #     return _sst_module.TwoLinkAcrobotDistance()
class Acrobot_Visualizer:
    def __init__(self, model, goal=None):
        self.model = model
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 2, 1)
        self.bx = self.fig.add_subplot(1, 2, 2)
        plt.show(block=False)
        self.goal = goal
        self.count = 0
       

    def current_state(self, x, goal_state=None, curr_color='g'):
        self.ax.cla()
       
        if goal_state is None:
            goal_state = self.goal
        _, _, xg, yg = self.model.visualize_point(goal_state)
        x1, y1, x2, y2 = self.model.visualize_point(x)
        self.ax.plot([0]+[x1], [0]+[y1], color='gray')
        self.ax.plot([x1]+[x2], [y1]+[y2], color='gray')
        self.ax.scatter([x1], [y1], color='gray', s=30)
        self.ax.scatter([x2], [y2], color=curr_color, s=5)
        self.ax.scatter([xg], [yg], color='r', s=10)
        # self.ax.axis('equal')
        self.ax.set(xlim=(-2*self.model.LENGTH, 2*self.model.LENGTH), ylim=(-2*self.model.LENGTH, 2*self.model.LENGTH))

        self.bx.scatter(x[0], x[1], c=curr_color)
        self.bx.scatter(goal_state[0], goal_state[1], c='r')

        self.bx.set_xlim(self.model.MIN_ANGLE, self.model.MAX_ANGLE)
        self.bx.set_ylim(self.model.MIN_ANGLE, self.model.MAX_ANGLE)
        # self.bx.axis('equal')

        plt.savefig('tmp/'+str(self.count)+'.png')
        self.count += 1
        plt.pause(1e-6)

if __name__ == '__main__':   
    model = Acrobot()
    dt = 2e-2
    state = np.array([0., 0., 0., 0.])
    goal = np.array([np.pi,  np.pi,  0, 0])


    # control = [-0.512885,3.895765,-0.808992,-3.679621,-2.857158,3.684517,3.914919,-0.920894,-3.879264,2.229122,]
    # time = [0.356690,0.886550,0.284282,0.632338,0.516671,0.582229,0.945147,0.335861,0.557898,0.389469,]
    control = [0.760619,2.496962,0.329872,-3.263391,-2.114961,3.134194,1.775590,-3.092880,-3.931654,2.468221,]
    time = [0.580043,0.561246,0.219913,0.618602,0.264407,0.462065,0.557775,0.497166,0.677087,0.707207,]


    viz = Acrobot_Visualizer(model, goal)
    for i in range(len(control)):
        # print(int(time[i]/dt))
        state = model.propagate(state.copy(), [control[i]], int(time[i]/dt), dt, viz)
        viz.current_state(state, goal)
    # print(state-goal)
    # print((state[:2] - goal[:2])**2)
