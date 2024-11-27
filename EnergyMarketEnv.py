import numpy as np
from gym import Env
from gym.spaces import Box  # , Discrete, Dict
import scipy.special as sp
import math
import random

""" ** Set the environment for the agent ** """


class EnergyMarketEnv(Env):
    def __init__(self):
        # tau: time lag between commitment and delivery: in present case, the producer commits 4 periods in advance
        self.tau = 4
        # Action is one-dimensional array, with value between 0 and x_max = 6.25 GW
        self.action_space = Box(low=np.array([0.0]), high=np.array([6.25]), dtype=np.float32)

        # State: Time are discrete values between 0 and 20, storage level,
        #        the last tree commitments and the selling price on the energy market
        self.observation_space = Box(low=np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                     high=np.array([20, 2.5, 6.25, 6.25, 6.25, 6.25, float('inf')]), dtype=np.float32)
        # Initialize the state variables (current state)
        self.time = 0
        self.storage_level = 0.0
        self.past_commitments = np.zeros(self.tau)
        # price value at time 0 ist the mean value mu = 40.712
        self.selling_price = 40.712

        self.state = np.array([self.time, self.storage_level, *self.past_commitments, self.selling_price])

        # Define further parameters for the environment
        self.m = 2  # Ratio of market price and penalty
        self.max_steps = 20  # Maximum number of time steps

        # Maximum storage level in {0, 1.25, 2.5, 3.75, 5, 6.25, 7.5}
        self.storage_level_max = 2.5

        # efficiency_charge/efficiency_discharge with efficiency_charge*efficiency_discharge in {0.5, 0.6, 0.7, 0.8, 0.9, 1}
        self.efficiency_charge = math.sqrt(0.9)
        self.efficiency_discharge = math.sqrt(0.9)

        # s_ci: cut-in speed, s_co: cut-out speed, s_r: rated speed, r: rated power, delta_t: length of period
        self.s_ci, self.s_r, self.s_co, self.r, self.delta_t = 3, 12, 25, 20, 0.25

        # Excess: production at time t + storage level at time t is insufficient for commitment from time t - tau
        self.excess = False

        # Solve lineare equation system A(a , b) = _b to get a and b values
        A = np.array([[1, self.s_ci ** 3], [1, self.s_r ** 3]])
        _b = np.array([0, self.r])
        # Solve system of equations
        solve = np.linalg.solve(A, _b)
        # Auxiliary parameters for wind turbine (a, b)
        self.a = solve[0]
        self.b = solve[1]

        # Weibull distribution parameters for energy production
        self.weibull_scale = 1/0.127  # 1/lamda value
        self.weibull_shape = 1.43  # k value
        self.num_samples = 10000
        self.energy_productions = self.energy_production_total(self.num_samples, self.weibull_shape, self.weibull_scale)
        self.energy_production = self.energy_productions[self.time]

        # Ornstein-Uhlenbeck process parameters for selling price
        self.ou_kappa = 1.035
        self.ou_mu = 40.712
        self.epsilon = np.random.normal(0, 12.693)
        self.selling_price = 40.712

        # Discount factor beta, time is discrete
        self.beta = 1

    def reset(self):
        # Reset the environment to initial state
        self.time = 0
        self.storage_level = 0.0
        self.past_commitments = np.zeros(self.tau)
        # The price value at time 0 ist the mean value  mu = 100
        self.selling_price = 40.712

        self.state = np.array([self.time, self.storage_level, *self.past_commitments, self.selling_price])
        self.energy_productions = self.energy_production_total(self.num_samples, self.weibull_shape, self.weibull_scale)
        self.energy_production = self.energy_productions[self.time]
        return self.state

    def step(self, action):
        # Execute the selected action
        commitment = action[0]

        # Update the state variables based on the action and new information at time t+1
        self.time += 1

        # Storage level at time t for computing the expected penalty at time t
        self.storage_level = self.storage_level_t(self.storage_level, self.energy_production)
        # Calculate the reward based on the current state and action
        reward = self.selling_price * commitment - self.beta * self.expected_penalty_value()
        # Check if time horizon is reached
        if self.time >= self.max_steps - 1:
            done = True
        else:
            done = False

        # update state variable: next state
        self.past_commitments = np.roll(self.past_commitments, -1)
        self.past_commitments[-1] = commitment
        self.energy_production = self.energy_productions[self.time]
        self.epsilon = np.random.normal(0, 25)
        self.selling_price = self.ornstein_uhlenbeck_process(self.selling_price, self.ou_mu, self.ou_kappa, self.epsilon)

        # Set placeholder for info
        info = {}

        # Return step information
        self.state = np.array([self.time, self.storage_level, *self.past_commitments, self.selling_price])
        return self.state, reward, done, info

    # Generate energy production over the planning horizon
    def energy_production_total(self, num_samples, shape, scale):
        wind_speeds_periods = []
        for _ in range(self.max_steps):
            wind_speeds = self.WS(num_samples, shape, scale)
            wind_speeds_periods.append(wind_speeds)
        # Choose a single wind speed for each period randomly
        selected_wind_speeds = [random.choice(wind_speeds) for wind_speeds in wind_speeds_periods]  # eigentlich unnötig
        energy_productions = [self.energy_production_t(ws) for ws in selected_wind_speeds]
        return energy_productions

    # Ornstein-Uhlenbeck process
    def ornstein_uhlenbeck_process(self, p_current, mean, kappa, epsilon):
        p_next = kappa * mean + (1 - kappa) * p_current + epsilon
        return p_next

    # Wind speed generated by Weibull distribution
    def WS(self, num_samples, shape, scale):
        return np.random.weibull(shape, num_samples) * scale

    # Energy production: Wind speed
    def energy_production_t(self, WS):
        if WS < self.s_ci or WS >= self.s_co:
            return 0
        elif self.s_ci <= WS < self.s_r:
            return (self.a + self.b * (WS ** 3)) * self.delta_t  # self.r *((WS- self.s_ci) / (self.s_r- self.s_ci))
        elif self.s_r <= WS < self.s_co:
            return self.r * self.delta_t

    # Storage Level evolution
    def storage_level_t(self, prior_storage_level, production_level):
        if self.past_commitments[0] < production_level:
            return min(self.storage_level_max,
                       prior_storage_level + self.efficiency_charge * (production_level - self.past_commitments[0]))
        else:
            self.excess = True
            return max(0, prior_storage_level - (1 / self.efficiency_discharge) * (
                        self.past_commitments[0] - production_level))

    # Fwb(lamda, kappa)
    def cumulative_WB(self, WS):
        return 1 - np.exp(-(1/self.weibull_scale * WS) ** self.weibull_shape)

    # fwb(lamda, kappa)
    def density_WB(self, WS):
        return 1/self.weibull_scale * self.weibull_shape * (
                    (1/self.weibull_scale * WS) ** (1/self.weibull_shape - 1)) * np.exp(
            -(1/self.weibull_scale * WS) ** self.weibull_shape)

    # cdf of Y_t
    def cumulative_Dis_Y_t(self, y):
        if y < 0:
            return 0
        elif 0 <= y < self.delta_t * self.r:
            return 1 - np.exp(-(1/self.weibull_scale * self.s_ci) ** self.weibull_shape) + np.exp(
                -(1/self.weibull_scale * self.s_co) ** self.weibull_shape) + self.cumulative_WB(
                self.sp_(y)) - self.cumulative_WB(self.s_ci)
        else:
            return 1

    def sp_(self, y):
        return ((y/self.delta_t - self.a) / self.b) ** (1/3)

    def one_delta_t(self, d):
        if d < self.r * self.delta_t:
            return 0
        else:
            return 1

    def integral_y_dF(self, c, d):
        x_1 = self.delta_t * (self.a * (-np.exp(-1/self.weibull_scale * self.sp_(d)) ** self.weibull_shape +
                                        np.exp(-1/self.weibull_scale * self.sp_(c)) ** self.weibull_shape))
        x_2 = (self.delta_t * self.b * self.weibull_scale ** 3) * \
                  (sp.gamma(1+3/self.weibull_shape) * sp.gammainc(1+3/self.weibull_shape,
                  (1/self.weibull_scale * self.sp_(d)) ** self.weibull_shape) - sp.gamma(1+3/self.weibull_shape)
                  * sp.gammainc(1+3/self.weibull_shape, (1/self.weibull_scale * self.sp_(c)) ** self.weibull_shape))
        x_3 = self.one_delta_t(d) * (self.delta_t * self.r * (np.exp(-(1/self.weibull_scale * self.s_r) ** self.weibull_shape))
                  - np.exp(-(1/self.weibull_scale * self.s_co) ** self.weibull_shape))
        return x_1 + x_2 + x_3

    def expected_penalty_value(self):
        # The reward is evaluated after the commitment of x_t
        if self.past_commitments[0] - (self.efficiency_discharge * self.storage_level + self.energy_production) < 0:
            return 0
        F_y = self.cumulative_Dis_Y_t(self.past_commitments[0] - self.efficiency_discharge * self.storage_level)
        E_P = self.conditional_expected_P_t()
        integral_y = self.integral_y_dF(0, self.past_commitments[0] - self.efficiency_discharge * self.storage_level)
        return F_y * (self.m * E_P * (self.past_commitments[0] - self.efficiency_discharge * self.storage_level)) - self.m * E_P * integral_y

    # E(Q_t+1 ()| S_t)
    def conditional_expected_P_t(self):
        return self.ou_kappa * self.ou_mu + (1 - self.ou_kappa) * self.selling_price