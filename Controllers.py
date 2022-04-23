import abc
import textwrap
from typing import List

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


class Learning_Controller:
    name = ''

    @abc.abstractmethod
    def explore(self) -> None:
        pass

    @abc.abstractmethod
    def reflect(self) -> None:
        pass

    @abc.abstractmethod
    def reward(self, reward):
        pass

    @abc.abstractmethod
    def get_string(self) -> str:
        return ''

    @abc.abstractmethod
    def reset(self):
        pass


class Learning_PID_Controller(PID_Controller, Learning_Controller):
    def __init__(self, name='', preset=(0, 0, 0)):
        super().__init__(preset)
        self.name = name
        self.current_rewards = []
        self.current_control = self.get_control()
        self.previous_rewards = []
        self.previous_control = self.get_control()

    def get_string(self):
        return f'{tuple(float("{:.4f}".format(x)) for x in self.get_control())}'

    def explore(self):
        previous_control = self.previous_control
        new_control = list(self.current_control)

        # Todo make explore progress comparison than to do it randomly
        for i in range(len(new_control)):
            improve = Settings.MULTIPLIER_IMPROVE * Settings.MULTIPLIER_EPSILON
            if previous_control[i] != new_control[i]:
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
            has_improved = False
        else:
            current_min = min(current_rewards)
            previous_min = min(previous_rewards)
            improvement = 0
            if current_min > 0 and previous_min > 0:
                improvement = current_min / previous_min
            elif current_min > 0 and previous_min < 0:
                improvement = 1
            elif current_min < 0 and previous_min > 0:
                difference = previous_min - current_min
                if difference > 1:
                    improvement = 1 / difference
                else:
                    improvement = 1
            elif current_min < 0 and previous_min < 0:
                difference = current_min - previous_min
                if difference > -1:
                    improvement = 1
                else:
                    improvement = -1 / difference
            has_improved = improvement > np.random.rand() * Settings.EPSILON_DISCOUNT

        if has_improved and sum(current_rewards) >= sum(previous_rewards):
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


class Multi_Learning_Controller(Learning_Controller):
    def __init__(self):
        self.controllers: List[Learning_Controller] = []
        self.selected: Learning_Controller = Learning_Controller()
        self.is_rotating = True
        self.index = 0
        self.count = 0
        self.previous_rewards = []
        self.current_rewards = []

    def add_controller(self, controller: Learning_Controller):
        self.controllers.append(controller)
        if not len(self.controllers) > 1:
            self.select_controller(0)

    def select_controller(self, index):
        if len(self.controllers) <= index:
            index = 0
        self.index = index
        self.count = 0
        self.selected = self.controllers[index]
        self.name = self.selected.name

    def get_string(self) -> str:
        return self.selected.get_string()

    def next_controller(self):
        self.select_controller(self.index + 1)

    def explore(self) -> None:
        self.selected.explore()

    def reflect(self) -> None:
        for reward in self.current_rewards:
            self.selected.reward(reward)
        self.selected.reflect()
        if len(self.previous_rewards) == 0:
            self.previous_rewards = self.current_rewards
            self.count += 1
        elif sum(self.current_rewards) >= sum(self.previous_rewards):
            self.previous_rewards = self.current_rewards
            self.count += 1
        elif self.is_rotating:
            self.next_controller()
            self.previous_rewards.clear()
        else:
            pass
        if self.is_rotating and self.count > 10:
            self.next_controller()
            self.previous_rewards.clear()
        self.current_rewards = []

    def reward(self, reward):
        for controller in self.controllers:
            controller.reset()
        self.current_rewards.append(reward)


class Environment_Controller:
    def __init__(self, env: gym.Env):
        self.action_space = env.action_space

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        return self.action_space.sample()

    @abc.abstractmethod
    def get_reward(self, observation: gym.core.ObsType) -> float:
        return 0


class Environment_Monitor:
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
        text = f'''
        Episode {episode}
        Last {Settings.EPISODE_LEARN} average rewards: {average:.3f}
        Highest reward: {maximum:.3f}
        Lowest reward: {minimum:.3f}
        Epsilon: {epsilon:.3f}
        Multiplier: {multiplier:.3f}
        '''
        return textwrap.dedent(text)

    def monitor(self, reward):
        self.rewards.append(reward)

    def process(self, episode):
        self.epsilons.append(Settings.EPSILON)
        self.multipliers.append(Settings.MULTIPLIER_EPSILON)
        self.episodes.append(episode)
        rewards = self.rewards
        self.averages.append(sum(rewards) / len(rewards))
        self.maximums.append(max(rewards))
        self.minimums.append(min(rewards))
        self.rewards.clear()


class Environment:
    def __init__(self, environment: gym.Env, learner: Learning_Controller, controller: Environment_Controller):
        self.env = environment
        self.controller = controller
        self.logger = Environment_Monitor()

        self.learner = learner

        self.episode = 1
        self.running = False
        self.rewards = 0

    def start(self, once=False):
        self.running = True
        if once:
            self.run_once()

    def stop(self):
        self.running = False
        self.env.close()

    def step_episode(self):
        episode = self.episode
        if episode % Settings.EPISODE_RENDER == 0:
            self.env.render()
        observation = self.env.reset()
        self.controller.reset()
        for time_steps in range(Settings.TIME_STEPS):
            if episode % Settings.EPISODE_SHOW == 0:
                self.env.render()

            action = self.controller.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            reward += self.controller.get_reward(observation)

            self.rewards += reward
            if done or time_steps + 1 == Settings.TIME_STEPS:
                if not Settings.EPISODE_PRINT_TOGGLE:
                    pass
                elif episode % Settings.EPISODE_PRINT == 0 or episode % Settings.EPISODE_SHOW == 0:
                    print("Episode {} finished after {} timesteps".format(episode, time_steps + 1))
                break

    def step_end(self):
        logger = self.logger
        learner = self.learner
        episode = self.episode

        learner.reward(self.rewards)
        logger.monitor(self.rewards)
        self.rewards = 0

        if episode % Settings.EPISODE_LEARN == 0:
            logger.process(episode)
            learner.reflect()
            learner.explore()

        if Settings.MULTIPLIER_EPSILON > Settings.EPSILON_CAP:
            Settings.MULTIPLIER_EPSILON *= Settings.EPSILON_DECAY_RATE
        if Settings.EPSILON > Settings.EPSILON_CAP:
            Settings.EPSILON *= Settings.EPSILON_DECAY_RATE

        if episode % Settings.EPISODE_SHOW == 0:
            log = logger.get_log()
            log += f'{self.learner.name} = {learner.get_string()}' + '\n'
            print(log)

        if episode > Settings.EPISODE_CAP:
            self.stop()
        self.episode += 1

    def run_once(self):
        observation = self.env.reset()
        rewards = 0
        for time_steps in range(Settings.TIME_STEPS * 2):
            self.env.render()
            action = self.controller.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            rewards += reward
            if done:
                print("Episode {} finished after {} timesteps".format(1, time_steps + 1))
                print("Collected rewards:", rewards)
                break
        self.stop()
