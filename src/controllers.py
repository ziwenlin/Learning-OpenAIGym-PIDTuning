import statistics
import textwrap
import time
from abc import ABC, abstractmethod

import gym
import numpy as np
from tabulate import tabulate

import settings

THRESHOLD_MIN = 0.8
THRESHOLD_RND = 1.0 - THRESHOLD_MIN


class InOutController:
    @abstractmethod
    def __init__(self):
        self.output = 0

    @abstractmethod
    def set_control(self, preset: tuple):
        pass

    @abstractmethod
    def get_control(self) -> tuple:
        return 0, 0, 0

    @abstractmethod
    def get_output(self, values: tuple, target: float) -> float:
        return 0

    @abstractmethod
    def reset(self):
        self.output = 0


class LearningController:
    @abstractmethod
    def __init__(self, name=''):
        self.name = name

    @abstractmethod
    def explore(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reflect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reward(self, reward) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_string(self) -> str:
        raise NotImplementedError


class EnvironmentWorker:
    def __init__(self, env: gym.Env):
        self.action_space = env.action_space
        self.difficulty = 0

    @abstractmethod
    def reset(self):
        self.difficulty = 0

    @abstractmethod
    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        return self.action_space.sample()

    @abstractmethod
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


class ImprovingController(LearningController, ABC):
    def __init__(self, name=''):
        super().__init__(name)
        self.previous_rewards = []
        self.current_rewards = []
        self.is_improving = [False, False, False]

    def reward(self, reward):
        self.current_rewards.append(reward)

    def reflect(self) -> None:
        self.is_improving = get_is_improving(
            self.current_rewards, self.previous_rewards
        )
        self.previous_rewards = self.current_rewards
        self.current_rewards = []


class ImprovingInOutController(InOutController, ImprovingController, ABC):
    def __init__(self, name='', preset=(0, 0, 0)):
        InOutController.__init__(self)
        ImprovingController.__init__(self, name)
        self.current_control = preset
        self.previous_control = preset

    def get_string(self):
        return get_tuple_string(self.get_control())

    def explore(self):
        self.current_control = get_control_mutated(
            self.current_control, self.previous_control)
        self.set_control(self.current_control)

    def reflect(self):
        previous_rewards = self.previous_rewards
        current_rewards = self.current_rewards
        ImprovingController.reflect(self)

        is_improving = get_is_improving_random(
            self.is_improving, current_rewards, previous_rewards)
        avg, low, high = is_improving

        if low and avg or low and high or avg and high:
            # When the newer control has scored an equal or better score
            # Overwrite the previous reward and control
            self.previous_rewards = current_rewards
            self.previous_control = self.current_control
        else:
            # Revert the changes
            # Restore previous reward and current control
            self.previous_rewards = previous_rewards
            self.current_control = self.previous_control

        self.set_control(self.current_control)
        self.current_rewards = []


class ImprovingPIDController(ImprovingInOutController, PIDController):
    def __init__(self, name='', preset=(0, 0, 0)):
        ImprovingInOutController.__init__(self, name, preset)
        PIDController.__init__(self, preset)

    def explore(self):
        self.current_control = get_control_mutated(
            self.current_control, self.previous_control, True)
        self.set_control(self.current_control)


class ImprovingNodeController(ImprovingInOutController, NodeController):
    def __init__(self, name='', preset=None):
        ImprovingInOutController.__init__(self, name, preset)
        NodeController.__init__(self, preset)
        if preset is None:
            raise ValueError('Please provide a preset')


class RotatingController:
    def __init__(self):
        self.controllers: list[LearningController] = []
        self.selected = LearningController()
        self.index = 0

    def add_controller(self, controller: LearningController):
        self.controllers.append(controller)
        if len(self.controllers) > 1:
            return
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


class RotatingImprovingController(ImprovingController, RotatingController):
    def __init__(self):
        ImprovingController.__init__(self, 'Rotating')
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
        previous_rewards = self.previous_rewards
        current_rewards = self.current_rewards
        ImprovingController.reflect(self)
        avg, low, high = self.is_improving

        if avg or low or high:
            self.previous_rewards = current_rewards
            self.count += 1
        else:
            self.previous_rewards = previous_rewards
            self.is_next = True
        if self.count >= 10:
            self.is_next = True

    def get_string(self) -> str:
        return RotatingController.get_string(self)

    def reset(self):
        for controller in self.controllers:
            controller.reset()


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
        self.buffer: list[dict] = []
        self.results: list[dict[str, any]] = []

    def get_log(self, n=-1):
        result = self.results[n]
        text = '\n'

        simple_info = {a: [b] for a, b in result.items()
                       if type(b) is not dict}
        text += tabulate(simple_info, headers='keys',
                         tablefmt='github') + '\n\n'

        header_info = ['category'] + list(result['episode'].keys())
        other_info = [[c] + list(item.values())
                      for c, item in result.items()
                      if type(item) is dict]
        text += tabulate(other_info, headers=header_info,
                         tablefmt='github') + '\n'

        return textwrap.dedent(text)

    def monitor(self, reward):
        self.buffer.append(reward)

    def process(self, episode):
        if len(self.buffer) == 0:
            raise IndexError
        result: dict[str, float | dict[str, float | int]] = {}
        self.results.append(result)

        result['division'] = episode
        result['highest'] = 0
        result['average'] = 0
        result['lowest'] = 0
        result['median'] = 0
        result['middle'] = 0
        result['epsilon'] = settings.EPSILON
        result['multiplier'] = settings.MULTIPLIER_EPSILON

        division = sorted(self.buffer, key=lambda ep: ep['reward'])
        self.buffer.clear()
        div_rewards = [ep['reward'] for ep in division]

        def get_index(value):
            item = min(division, key=lambda x: abs(x['reward'] - value))
            return division.index(item)

        median_value = statistics.median(div_rewards)
        median_i = get_index(median_value)

        middle_value = (div_rewards[0] + div_rewards[-1]) / 2
        middle_i = get_index(middle_value)

        mean_value = statistics.mean(div_rewards)
        mean_i = get_index(mean_value)

        def get_result(division, category):
            return {
                'highest': division[-1][category],
                'average': division[mean_i][category],
                'lowest': division[0][category],
                'median': division[median_i][category],
                'middle': division[middle_i][category],
            }

        result['reward'] = get_result(division, 'reward')
        for k, i in result['reward'].items():
            result['reward'][k] = round(i, 2)
        result['difficulty'] = get_result(division, 'difficulty')
        for k, i in result['difficulty'].items():
            result['difficulty'][k] = round(i, 2)
        result['episode'] = get_result(division, 'episode')
        for k, i in result['episode'].items():
            result['episode'][k] = (i - 1) % len(division) + 1

        result['average'] = round(mean_value, 2)
        result['median'] = round(median_value, 2)
        result['middle'] = round(middle_value, 2)
        result['lowest'] = round(division[0]['reward'], 2)
        result['highest'] = round(division[-1]['reward'], 2)


class EnvironmentManager:
    def __init__(self,
                 environment: gym.Env,
                 agent: LearningController,
                 worker: EnvironmentWorker):
        self.env = environment
        self.worker = worker
        self.logger = EnvironmentMonitor()

        self.agent = agent

        self.fps_time = time.time()
        self.episode = 1
        self.running = False
        self.rewards = 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        self.env.close()

    def step_episode(self):
        episode = self.episode
        frame_time = time.time()
        if frame_time > self.fps_time:
            self.fps_time = frame_time + 0.2
            self.env.render()
        observation = self.env.reset()
        self.worker.reset()
        self.agent.reset()
        for time_steps in range(settings.TIME_STEPS):
            if episode % settings.EPISODE_SHOW == 1:
                self.env.render()

            action = self.worker.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            reward += self.worker.get_reward(observation)

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

        self.agent.reward(self.rewards)
        self.logger.monitor({
            'episode': episode,
            'reward': self.rewards,
            'difficulty': self.worker.difficulty,
        })
        self.rewards = 0

        if episode % settings.EPISODE_LEARN == 0:
            self.logger.process(episode)
            self.agent.reflect()
            self.agent.explore()

        if settings.MULTIPLIER_EPSILON > settings.EPSILON_CAP:
            settings.MULTIPLIER_EPSILON *= settings.EPSILON_DECAY_RATE
        if settings.EPSILON > settings.EPSILON_CAP:
            settings.EPSILON *= settings.EPSILON_DECAY_RATE

        if episode % settings.EPISODE_SHOW == 0:
            log = self.logger.get_log()
            log += f'{self.agent.name} = {self.agent.get_string()}' + '\n'
            print(log)

        if episode > settings.EPISODE_CAP:
            self.stop()
        self.episode += 1

    def run(self):
        self.start()
        while self.running:
            try:
                self.step_episode()
                self.step_end()
            except KeyboardInterrupt:
                break
        self.stop()

    def run_once(self):
        observation = self.env.reset()
        done = False
        time_steps = 0
        rewards = 0
        while not done:
            self.env.render()
            action = self.worker.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            rewards += reward
            time_steps += 1
        print("Episode {} finished after {} time steps"
              .format(1, time_steps + 1))
        print("Collected rewards:", rewards)
        self.stop()


def get_tuple_string(array: tuple):
    return f'{tuple(float(f"{value:.4f}") for value in array)}'


def get_improvement_gain(current, previous):
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


def get_is_improvement(bypass, new_values, old_values, func, threshold):
    if bypass:
        return bypass
    value_new = func(new_values)
    value_old = func(old_values)
    improvement = get_improvement_gain(value_new, value_old)
    return improvement > threshold


def get_is_improving_random(improvements, new_values, old_values):
    avg, low, high = improvements
    threshold = THRESHOLD_MIN + THRESHOLD_RND * np.random.rand()
    low = get_is_improvement(low, new_values, old_values, min, threshold)
    high = get_is_improvement(high, new_values, old_values, max, threshold)
    avg = get_is_improvement(avg, new_values, old_values, sum, threshold)
    return avg, low, high


def get_is_improving(new_values, old_values):
    a, b, c = (True, True, True)
    if len(old_values) == 0:
        return a, b, c
    else:
        a = sum(new_values) > sum(old_values)
        b = min(new_values) > min(old_values)
        c = max(new_values) > max(old_values)
    return a, b, c


def get_control_mutated(new_control, previous_control, is_pid=False):
    if type(new_control) is not list:
        new_control = list(new_control)
    current_index = get_index_changed(new_control, previous_control)
    if np.random.rand() > settings.EPSILON:
        # Improve the changed setting
        get_control_improved_mutation(
            new_control, previous_control, current_index)
        return tuple(new_control)

    # Random explore settings which has not been changed
    random_index = get_index_random(len(new_control), current_index)
    if is_pid:
        multiplier = (5, 0.1, 2)[random_index]
    else:
        multiplier = 1
    new_control[random_index] += get_mutation_random() * multiplier
    return tuple(new_control)


def get_control_improved_mutation(new_control, old_control, index):
    # Mutate parameter that has been changed before
    if type(new_control) is not list:
        new_control = list(new_control)
    improve = get_mutation_improved(new_control, old_control, index)
    new_control[index] += improve
    return tuple(new_control)


def get_control_random_mutation(control, index=-1):
    # Random mutate parameter at index
    if type(control) is not list:
        control = list(control)
    if index == -1:
        index = np.random.randint(len(control))
    control[index] += get_mutation_random()
    return tuple(control)


def get_mutation_improved(new_control, old_control, index):
    improve = settings.MULTIPLIER_IMPROVE * settings.MULTIPLIER_EPSILON
    difference = (new_control[index] - old_control[index])
    return difference * improve


def get_mutation_random():
    improve = settings.MULTIPLIER_RAND * settings.MULTIPLIER_EPSILON
    return (np.random.rand() - 0.5) * improve


def get_index_changed(tuple_a, tuple_b):
    for index in range(len(tuple_a)):
        if tuple_b[index] != tuple_a[index]:
            return index
    return -1


def get_index_random(size, not_index=-1):
    random_index = np.random.randint(size)
    while random_index == not_index:
        random_index = np.random.randint(size)
    return random_index
