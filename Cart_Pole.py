from typing import List, Tuple

import gym
import numpy as np

EPISODE_MULTI = 10
EPISODES = EPISODE_MULTI * 1000
EPISODE_SHOW = EPISODE_MULTI * 100
EPISODE_PRINT = EPISODE_MULTI * 10

TIME_STEPS = 100
EPISODE_LEARN = 20

EPSILON = 0.9
EPSILON_CAP = 0.2
EPSILON_DISCOUNT = 0.95
EPSILON_DECAY = EPSILON_DISCOUNT ** 0.001

MULTIPLIER_EPSILON = 10
MULTIPLIER_IMPROVE = 0.2
MULTIPLIER_RAND = 0.1

PRESET_PID_CART = (0, 0, 0)
# PRESET_PID_CART = (-0.924861051852458, 0.03002569629632385, -0.2994526736003437)
# PRESET_PID_CART = (-0.924861051852458, 0.03002569629632385, -0.3848792315820444)

PRESET_PID_POLE = (0, 0, 0)
PRESET_PID_POLE = (-1.8394291134013734, 0.20138629211940734, -5.200392703676353)
# PRESET_PID_POLE = (-4.512285277021222, 3.7468386916186502, 7.743010122805896)
# PRESET_PID_POLE = (-14.118865930245665, -0.8059379276598793, -1.3162381258382854)

class PIDController:
    def __init__(self, preset=(0, 0, 0)):
        self.action_space = []

        self.current_rewards = [0]
        self.current_control = preset
        self.previous_rewards = [0]
        self.previous_control = preset

        self.derivative = 0
        self.integral = 0

        self.p_control, self.i_control, self.d_control = preset

    def set_control(self, preset):
        self.p_control, self.i_control, self.d_control = preset

    def get_control(self):
        return self.p_control, self.i_control, self.d_control

    def get_output(self, value, setpoint):
        error = setpoint - value
        self.integral += error

        p = self.p_control * error
        i = self.i_control * (error - self.integral)
        d = self.d_control * (error - self.derivative)

        self.derivative = error # Used as previous error
        return p + i + d

    def explore(self):
        previous_control = self.previous_control
        new_control = list(self.current_control)

        for i in range(len(new_control)):
            improve = (np.random.rand() - 0.5) * MULTIPLIER_IMPROVE * MULTIPLIER_EPSILON
            if previous_control[i] == new_control[i]:
                if np.random.rand() > EPSILON:
                    new_control[i] += (new_control[i] - previous_control[i]) * improve
                    break
        else:
            i = np.random.randint(len(new_control))
            new_control[i] += (np.random.rand() - 0.5) * MULTIPLIER_RAND * MULTIPLIER_EPSILON

        self.set_control(new_control)
        self.current_control = tuple(new_control)

    def reflect(self):
        previous_rewards = self.previous_rewards
        current_rewards = self.current_rewards

        if sum(current_rewards) >= sum(previous_rewards):
            # When the newer control has scored an equal or better score
            # Overwrite the previous reward and control
            self.previous_rewards = current_rewards.copy()
            self.previous_control = tuple(self.get_control())
        else:
            # Revert the changes
            # Reset current reward and control
            self.current_control = self.previous_control

        self.set_control(self.current_control)
        self.current_rewards.clear()

    def reset(self):
        self.derivative = 0
        self.integral = 0

    def reward(self, reward):
        self.derivative = 0
        self.integral = 0
        self.current_rewards.append(reward)


env = gym.make('CartPole-v1')

pole_agent = PIDController(PRESET_PID_POLE)
cart_agent = PIDController(PRESET_PID_CART)
# agent = cart_agent
# agent.name = 'CART'
agent = pole_agent
agent.name = 'POLE'
sum_reward = 0


def action_space(output):
    if output > 0:
        action = 1
    else:
        action = 0
    return action


for i_episode in range(EPISODES):
    if (i_episode + 1) % EPISODE_SHOW == 0:
        print()
        print(f'Episode {i_episode + 1}')
        print(f'Last {EPISODE_LEARN} rewards: {sum(agent.previous_rewards)}')
        print(f'Highest reward: {max(agent.previous_rewards)}')
        print(f'Lowest reward: {min(agent.previous_rewards)}')
        print(f'Epsilon: {EPSILON}')
        print(f'Multiplier: {MULTIPLIER_EPSILON}')
        print(f'PRESET_PID_{agent.name} = {agent.current_control}')
        print()
    observation = env.reset()
    for time_steps in range(TIME_STEPS):
        if (i_episode + 1) % EPISODE_SHOW == 0:
            env.render()
        output = pole_agent.get_output(observation[2], 0.0)
        # output += cart_agent.get_output(observation[0], -0.50)
        action = action_space(output)
        observation, reward, done, info = env.step(action)
        # reward -= (observation[0] - 0.50) ** 2 * 10
        sum_reward += reward
        if done:
            if (i_episode + 1) % EPISODE_PRINT == 0:
                print("Episode {} finished after {} timesteps".format(i_episode + 1, time_steps + 1))
            elif (i_episode + 1) % EPISODE_SHOW == 0:
                print("Episode {} finished after {} timesteps".format(i_episode + 1, time_steps + 1))
            break
    else:
        if (i_episode + 1) % EPISODE_PRINT == 0:
            print("Episode {} finished after {} timesteps".format(i_episode + 1, TIME_STEPS))
        elif (i_episode + 1) % EPISODE_SHOW == 0:
            print("Episode {} finished after {} timesteps".format(i_episode + 1, TIME_STEPS))

    agent.reward(sum_reward)
    sum_reward = 0
    if (i_episode + 1) % EPISODE_LEARN == 0:
        agent.reflect()
        agent.explore()

    if MULTIPLIER_EPSILON > 1.0:
        MULTIPLIER_EPSILON *= EPSILON_DECAY
    if EPSILON > EPSILON_CAP:
        EPSILON *= EPSILON_DECAY

env.close()
