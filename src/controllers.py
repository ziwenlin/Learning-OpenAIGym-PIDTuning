import abc
import textwrap
from abc import ABC
from typing import List

import gym
import numpy as np

import settings


class InOutController:
    @abc.abstractmethod
    def __init__(self):
        self.output = 0

    @abc.abstractmethod
    def set_control(self, preset: tuple):
        pass

    @abc.abstractmethod
    def get_control(self) -> tuple:
        pass

    @abc.abstractmethod
    def get_output(self, values: tuple, target: float) -> float:
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class LearningController:
    @abc.abstractmethod
    def __init__(self, name=''):
        self.previous_rewards = []
        self.current_rewards = []
        self.name = name

    @abc.abstractmethod
    def explore(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def reflect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self, reward):
        raise NotImplementedError

    @abc.abstractmethod
    def get_string(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class EnvironmentController:
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


class PIDController(InOutController):
    def __init__(self, preset):
        super().__init__()
        self.d_value = 0
        self.i_value = 0

        self.p_control, self.i_control, self.d_control = preset

    def set_control(self, preset):
        self.p_control, self.i_control, self.d_control = preset

    def get_control(self):
        return self.p_control, self.i_control, self.d_control

    def get_output(self, values, target):
        error = target - sum(values)

        p = self.p_control * error
        i = self.i_control * (error + self.i_value)
        d = self.d_control * (error - self.d_value)

        self.i_value += error
        self.d_value = error  # Used as previous error
        self.output = p + i + d
        return self.output

    def reset(self):
        self.output = 0
        self.d_value = 0
        self.i_value = 0


class NodeController(InOutController):
    def __init__(self, preset):
        super().__init__()
        self.control = preset

    def set_control(self, preset):
        self.control = tuple(preset)

    def get_control(self) -> tuple:
        return tuple(self.control)

    def get_output(self, observation, offset) -> float:
        output = offset
        for weight, value in zip(self.control, observation):
            output += weight * value
        self.output = output
        return output

    def reset(self):
        self.output = 0


class LearningInOutController(InOutController, LearningController, ABC):
    def __init__(self, name='', preset=(0, 0, 0)):
        InOutController.__init__(self)
        LearningController.__init__(self, name)
        self.current_control = preset
        self.previous_control = preset

    def get_string(self):
        return f'{tuple(float("{:.4f}".format(x)) for x in self.get_control())}'

    def explore(self):
        previous_control = self.previous_control
        new_control = list(self.current_control)

        # Todo make explore progress comparison than to do it randomly
        for i in range(len(new_control)):
            improve = settings.MULTIPLIER_IMPROVE * settings.MULTIPLIER_EPSILON
            if previous_control[i] != new_control[i]:
                if np.random.rand() > settings.EPSILON:
                    new_control[i] += (new_control[i] - previous_control[i]) * improve
                    break
        else:
            # Random explore settings
            i = np.random.randint(len(new_control))
            new_control[i] += (
                    (np.random.rand() - 0.5) *
                    settings.MULTIPLIER_RAND * settings.MULTIPLIER_EPSILON
            )

        self.current_control = tuple(new_control)
        self.set_control(self.current_control)

    def reflect(self):
        previous_rewards = self.previous_rewards
        current_rewards = self.current_rewards

        if len(previous_rewards) == 0:
            # When it is the first run
            self.previous_rewards = current_rewards
            self.previous_control = self.current_control
            has_improved = False
        else:
            current_min = min(current_rewards)
            previous_min = min(previous_rewards)
            improvement = get_improvement(current_min, previous_min)
            has_improved = improvement > np.random.rand() * settings.EPSILON_DISCOUNT

        if has_improved and sum(current_rewards) >= sum(previous_rewards):
            # When the newer control has scored an equal or better score
            # Overwrite the previous reward and control
            self.previous_rewards = current_rewards
            self.previous_control = self.current_control
        else:
            # Revert the changes
            # Reset current reward and control
            self.current_control = self.previous_control

        self.set_control(self.current_control)
        self.current_rewards = []

    def reward(self, reward):
        self.reset()
        self.current_rewards.append(reward)


class LearningPIDController(LearningInOutController, PIDController):
    def __init__(self, name='', preset=(0, 0, 0)):
        LearningInOutController.__init__(self, name, preset)
        PIDController.__init__(self, preset)


class LearningNodeController(LearningInOutController, NodeController):
    def __init__(self, name='', preset=None):
        LearningInOutController.__init__(self, name, preset)
        NodeController.__init__(self, preset)
        if preset is None:
            raise ValueError('Please provide a preset')


class RotatingController():
    def __init__(self):
        self.controllers: List[LearningInOutController] = []
        self.selected = LearningInOutController()
        self.index = 0

    def add_controller(self, controller: LearningInOutController):
        self.controllers.append(controller)
        if not len(self.controllers) > 1:
            self.select_controller(0)

    def select_controller(self, index):
        if len(self.controllers) <= index:
            index = 0
        self.index = index
        self.selected = self.controllers[index]

    def get_string(self) -> str:
        return self.selected.get_string()

    def next_controller(self):
        self.select_controller(self.index + 1)


class LearningMultiController(LearningController, RotatingController):
    def __init__(self):
        LearningController.__init__(self)
        RotatingController.__init__(self)
        self.is_rotating = True
        self.is_next = False
        self.count = 0

    def explore(self) -> None:
        if self.is_rotating and self.is_next:
            self.next_controller()
            self.name = self.selected.name
            self.previous_rewards = []
            self.is_next = False
            self.count = 0
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
        else:
            self.is_next = True
        if self.count >= 10:
            self.is_next = True
        self.current_rewards = []

    def get_string(self) -> str:
        return RotatingController.get_string(self)

    def reset(self):
        for controller in self.controllers:
            controller.reset()

    def reward(self, reward):
        self.current_rewards.append(reward)
        self.reset()


# class GeneticMultiController(LearningController):
#     def __init__(self):
#         self.controllers: List[LearningController] = []
#         self.selected: LearningController = LearningController()
#         self.is_rotating = True
#         self.index = 0
#         self.count = 0
#         self.previous_rewards = []
#         self.current_rewards = []
#
# class GeneticInOutController(InOutController, LearningController):
#     def __init__(self, name='', preset=(0, 0, 0)):
#         super().__init__(preset)
#         self.name = name


class EnvironmentMonitor:
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
        Last {settings.EPISODE_LEARN} average rewards: {average:.3f}
        Highest reward: {maximum:.3f}
        Lowest reward: {minimum:.3f}
        Epsilon: {epsilon:.3f}
        Multiplier: {multiplier:.3f}
        '''
        return textwrap.dedent(text)

    def monitor(self, reward):
        self.rewards.append(reward)

    def process(self, episode):
        self.epsilons.append(settings.EPSILON)
        self.multipliers.append(settings.MULTIPLIER_EPSILON)
        self.episodes.append(episode)
        rewards = self.rewards
        self.averages.append(sum(rewards) / len(rewards))
        self.maximums.append(max(rewards))
        self.minimums.append(min(rewards))
        self.rewards.clear()


class Environment:
    def __init__(self,
                 environment: gym.Env,
                 learner: LearningController,
                 controller: EnvironmentController):
        self.env = environment
        self.controller = controller
        self.logger = EnvironmentMonitor()

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
        if episode % settings.EPISODE_RENDER == 0:
            self.env.render()
        observation = self.env.reset()
        self.controller.reset()
        for time_steps in range(settings.TIME_STEPS):
            if episode % settings.EPISODE_SHOW == 0:
                self.env.render()

            action = self.controller.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            reward += self.controller.get_reward(observation)

            self.rewards += reward
            if done or time_steps + 1 == settings.TIME_STEPS:
                if not settings.EPISODE_PRINT_TOGGLE:
                    pass
                elif episode % settings.EPISODE_PRINT == 0 or \
                        episode % settings.EPISODE_SHOW == 0:
                    print("Episode {} finished after {} time steps"
                          .format(episode, time_steps + 1))
                break

    def step_end(self):
        episode = self.episode

        self.learner.reward(self.rewards)
        self.logger.monitor(self.rewards)
        self.rewards = 0

        if episode % settings.EPISODE_LEARN == 0:
            self.logger.process(episode)
            self.learner.reflect()
            self.learner.explore()

        if settings.MULTIPLIER_EPSILON > settings.EPSILON_CAP:
            settings.MULTIPLIER_EPSILON *= settings.EPSILON_DECAY_RATE
        if settings.EPSILON > settings.EPSILON_CAP:
            settings.EPSILON *= settings.EPSILON_DECAY_RATE

        if episode % settings.EPISODE_SHOW == 0:
            log = self.logger.get_log()
            log += f'{self.learner.name} = {self.learner.get_string()}' + '\n'
            print(log)

        if episode > settings.EPISODE_CAP:
            self.stop()
        self.episode += 1

    def run_once(self):
        observation = self.env.reset()
        rewards = 0
        for time_steps in range(settings.TIME_STEPS * 2):
            self.env.render()
            action = self.controller.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            rewards += reward
            if done:
                print("Episode {} finished after {} time steps"
                      .format(1, time_steps + 1))
                print("Collected rewards:", rewards)
                break
        self.stop()


def get_improvement(current, previous):
    if current > 0 and previous > 0:
        pass
    elif current > previous:
        current += abs(previous) + 2
        previous += abs(previous) + 1
    elif current < previous:
        previous += abs(current) + 2
        current += abs(current) + 1
    elif current == previous:
        current = 9
        previous = 10
    improvement = current / previous
    return improvement
