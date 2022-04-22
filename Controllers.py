import gym
import numpy as np

import Settings


class PID_Controller:
    def __init__(self, preset):
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

        self.derivative = error  # Used as previous error
        return p + i + d

    def reset(self):
        self.derivative = 0
        self.integral = 0

    def get_string(self):
        return f'{tuple(float("{:.4f}".format(x)) for x in self.get_control())}'


class PID_Learning_Controller(PID_Controller):
    def __init__(self, preset=(0, 0, 0)):
        super().__init__(preset)
        self.current_rewards = []
        self.current_control = self.get_control()
        self.previous_rewards = []
        self.previous_control = self.get_control()

    def explore(self):
        previous_control = self.previous_control
        new_control = list(self.current_control)

        # Todo make explore progress comparison
        for i in range(len(new_control)):
            improve = (np.random.rand() - 0.5) * Settings.MULTIPLIER_IMPROVE * Settings.MULTIPLIER_EPSILON
            if previous_control[i] == new_control[i]:
                if np.random.rand() > Settings.EPSILON:
                    new_control[i] += (new_control[i] - previous_control[i]) * improve
                    break
        else:
            # Random explore settings
            i = np.random.randint(len(new_control))
            new_control[i] += (np.random.rand() - 0.5) * Settings.MULTIPLIER_RAND * Settings.MULTIPLIER_EPSILON

        self.set_control(new_control)
        self.current_control = tuple(new_control)

    def reflect(self):
        previous_rewards = self.previous_rewards
        current_rewards = self.current_rewards

        if len(previous_rewards) == 0:
            # When it is the first run
            self.previous_rewards = current_rewards.copy()
            self.previous_control = tuple(self.get_control())
        elif sum(current_rewards) >= sum(previous_rewards):
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

    def reward(self, reward):
        self.reset()
        self.current_rewards.append(reward)


class Logger:
    def __init__(self):
        self.rewards = []

        self.episodes = []
        self.averages = []
        self.maximums = []
        self.minimums = []
        self.epsilons = []
        self.multipliers = []

    def get_log(self, n=-1):
        episode = self.episodes[n]
        average = self.averages[n]
        maximum = self.maximums[n]
        minimum = self.minimums[n]
        epsilon = self.epsilons[n]
        multiplier = self.multipliers[n]
        return f'''
        Episode {episode}
        Last {Settings.EPISODE_LEARN} average rewards: {average}
        Highest reward: {maximum}
        Lowest reward: {minimum}
        Epsilon: {epsilon}
        Multiplier: {multiplier}
        '''


    def monitor(self, reward):
        self.rewards.append(reward)

    def process(self, episode, epsilon, multiplier):
        self.epsilons. append(epsilon)
        self.multipliers.append(multiplier)
        self.episodes.append(episode)
        rewards = self.rewards
        self.averages.append(sum(rewards)/len(rewards))
        self.maximums.append(max(rewards))
        self.minimums.append(min(rewards))
        self.rewards.clear()

class Environment:
    def __init__(self, environment: str, agent: PID_Learning_Controller, name: str):
        self.env = gym.make(environment)
        self.logger = Logger()

        self.agent = agent
        self.agent.name = name

        self.episode = 1
        self.running = False
        self.rewards = 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        self.env.close()

    def step_single(self):
        episode = self.episode
        observation = self.env.reset()
        for time_steps in range(Settings.TIME_STEPS):
            if (self.episode + 1) % Settings.EPISODE_SHOW == 0:
                self.env.render()
            output = 0
            output += pole_agent.get_output(observation[2], 0.0)
            # output += cart_agent.get_output(observation[0], -0.50)
            action = action_space(output)
            observation, reward, done, info = self.env.step(action)
            # reward -= (observation[0] - 0.50) ** 2 * 10
            self.rewards += reward
            if done:
                if (self.episode + 1) % Settings.EPISODE_PRINT == 0 or (self.episode + 1) % Settings.EPISODE_SHOW == 0:
                    print("Episode {} finished after {} timesteps".format(self.episode + 1, time_steps + 1))
                break
        else:
            if (self.episode + 1) % Settings.EPISODE_PRINT == 0 or (self.episode + 1) % Settings.EPISODE_SHOW == 0:
                print("Episode {} finished after {} timesteps".format(self.episode + 1, Settings.TIME_STEPS))

    def step_episode(self):
        logger = self.logger
        agent = self.agent
        episode = self.episode

        agent.reward(self.rewards)
        logger.monitor(self.rewards)
        self.rewards = 0

        if episode % Settings.EPISODE_LEARN == 0:
            logger.process(episode, Settings.EPSILON, Settings.MULTIPLIER_EPSILON)
            agent.reflect()
            agent.explore()

        if Settings.MULTIPLIER_EPSILON > 1.0:
            Settings.MULTIPLIER_EPSILON *= Settings.EPSILON_DECAY
        if Settings.EPSILON > Settings.EPSILON_CAP:
            Settings.EPSILON *= Settings.EPSILON_DECAY

        if episode % Settings.EPISODE_SHOW == 0:
            log = logger.get_log()
            log += f'PRESET_PID_{agent.name} = {agent.get_string()}' + '\n'
            print(log)

        if episode > Settings.EPISODES:
            self.stop()