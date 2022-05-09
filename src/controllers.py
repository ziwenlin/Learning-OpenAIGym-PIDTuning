import statistics
import textwrap
import time
import abc

import gym
import numpy as np
from tabulate import tabulate

import settings
from mutations import mutate_io_controller


class InOutModel:
    """
    Base class that provides the interface for any models.

    Available methods:
    :method:`set_model()`,
    :method:`get_model()`,
    :method:`get_output()`,
    :method:`reset()`
    """

    @abc.abstractmethod
    def __init__(self):
        self.output = 0

    @abc.abstractmethod
    def set_model(self, model: tuple) -> None:
        """
        Overwrite the current model with a new
        model configuration.

        :param model: Desired model configuration
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_model(self) -> tuple:
        """
        Returns the current model configuration.

        :return: Current model configuration
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_output(self, values: tuple, target: float) -> float:
        """
        Returns an output based on the input values and target.
        The model tries to match the input values to the target.

        :param values: Environment observation values
        :param target: Desired output value
        :return: Output of the model
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Restores all attributes to the starting state.

        :return: None
        """
        self.output = 0


class LearningController:
    @abc.abstractmethod
    def __init__(self, name=''):
        self.name = name

    @abc.abstractmethod
    def explore(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def reflect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self, reward) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_string(self) -> str:
        raise NotImplementedError


class BaseManager:
    """
    Base class that stores and manages
    models or controllers.
    """

    @abc.abstractmethod
    def add_controller(self, controller) -> None:
        """
        Adds a controller to the manager configuration.

        :param controller: Any instance of a controller
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_controller(self, index) -> None:
        """
        Selects a controller at the given index in the
        manager configuration.

        :param int index: Index of a controller.
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def next_controller(self) -> None:
        """
        Selects the next controller in the manager configuration.

        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self) -> int:
        """
        Returns the amount of stored controllers configuration
        inside the controller manager.

        :return: Amount stored controllers
        """
        raise NotImplementedError


class EnvironmentWorker:
    def __init__(self, env: gym.Env):
        self.action_space = env.action_space
        self.difficulty = 0

    @abc.abstractmethod
    def reset(self):
        self.difficulty = 0

    @abc.abstractmethod
    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        return self.action_space.sample()

    @abc.abstractmethod
    def get_reward(self, observation: gym.core.ObsType) -> float:
        return 0


class PIDModel(InOutModel):
    def __init__(self, preset):
        super().__init__()
        self.d_value = 0
        self.i_value = 0

        self.p_control, self.i_control, self.d_control = preset

    def set_model(self, model):
        self.p_control, self.i_control, self.d_control = model

    def get_model(self):
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


class NodeModel(InOutModel):
    def __init__(self, preset):
        super().__init__()
        self.control = preset

    def set_model(self, model):
        self.control = tuple(model)

    def get_model(self) -> tuple:
        return tuple(self.control)

    def get_output(self, observation, offset) -> float:
        output = offset
        for weight, value in zip(self.control, observation):
            output += weight * value
        self.output = output
        return output

    def reset(self):
        self.output = 0


class ImprovingController(LearningController, abc.ABC):
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


class ImprovingInOutModel(
    InOutModel, ImprovingController, abc.ABC):
    def __init__(self, name='', preset=(0, 0, 0)):
        InOutModel.__init__(self)
        ImprovingController.__init__(self, name)
        self.current_control = preset
        self.previous_control = preset

    def get_string(self):
        return get_tuple_string(self.get_model())

    def explore(self):
        self.current_control = mutate_io_controller(
            self.current_control, self.previous_control)
        self.set_model(self.current_control)

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

        self.set_model(self.current_control)
        self.current_rewards = []


class ImprovingPIDModel(ImprovingInOutModel, PIDModel):
    def __init__(self, name='', preset=(0, 0, 0)):
        ImprovingInOutModel.__init__(self, name, preset)
        PIDModel.__init__(self, preset)

    def explore(self):
        self.current_control = mutate_io_controller(
            self.current_control, self.previous_control, 'pid')
        self.set_model(self.current_control)


class ImprovingNodeModel(ImprovingInOutModel, NodeModel):
    def __init__(self, name='', preset=None):
        ImprovingInOutModel.__init__(self, name, preset)
        NodeModel.__init__(self, preset)
        if preset is None:
            raise ValueError('Please provide a preset')


class LearningControllerManager(BaseManager):
    """
    Implements :class:`BaseManager` and stores and manages
    multiple instances of :class:`LearningController`.
    """

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
        if not index < len(self.controllers):
            index = 0
        self.index = index
        self.selected = self.controllers[index]

    def get_string(self) -> str:
        return self.selected.get_string()

    def next_controller(self):
        self.select_controller(self.index + 1)

    def get_size(self):
        return len(self.controllers)


class ImprovingControllerManager(ImprovingController, LearningControllerManager):
    def __init__(self):
        ImprovingController.__init__(self, 'Manager')
        LearningControllerManager.__init__(self)
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
        return LearningControllerManager.get_string(self)

    def reset(self):
        for controller in self.controllers:
            controller.reset()


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

        division = sorted(self.buffer, key=lambda ep: ep['reward'])
        self.buffer.clear()
        division_rewards = [ep['reward'] for ep in division]

        lowest_i, highest_i = 0, -1
        lowest_value = division_rewards[lowest_i]
        highest_value = division_rewards[highest_i]

        median_value = statistics.median(division_rewards)
        median_i = get_index_closest(median_value, division_rewards)

        middle_value = (lowest_value + highest_value) / 2
        middle_i = get_index_closest(middle_value, division_rewards)

        mean_value = statistics.mean(division_rewards)
        mean_i = get_index_closest(mean_value, division_rewards)

        def get_result(category):
            result[category] = {
                'highest': division[highest_i][category],
                'average': division[mean_i][category],
                'lowest': division[lowest_i][category],
                'median': division[median_i][category],
                'middle': division[middle_i][category],
            }
            if category == 'episode':
                for k, value in result['episode'].items():
                    result['episode'][k] = (value - 1) % len(division) + 1
                return
            for k, value in result[category].items():
                result[category][k] = round(value, 2)

        get_result('reward')
        get_result('difficulty')
        get_result('episode')

        result['division'] = episode
        result['highest'] = round(highest_value, 2)
        result['average'] = round(mean_value, 2)
        result['lowest'] = round(lowest_value, 2)
        result['median'] = round(median_value, 2)
        result['middle'] = round(middle_value, 2)
        result['epsilon'] = settings.EPSILON
        result['multiplier'] = settings.MULTIPLIER_EPSILON


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
    threshold = settings.THRESHOLD_MIN
    threshold += settings.THRESHOLD_RND * np.random.rand()
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


def get_index_difference(tuple_a, tuple_b) -> int:
    """
    Compares two data sets and returns the index
    when a difference has been found.

    :param tuple or list tuple_a: Container to compare
    :param tuple or list tuple_b: Container to compare
    :return: Index of first difference, defaults to -1
     if nothing has been found
    """
    for index in range(len(tuple_a)):
        if tuple_b[index] != tuple_a[index]:
            return index
    return -1


def get_index_random(index_range, index_skip=-1) -> int:
    """
    Gets random index within a maximum range.
    When index_skip is specified it will not return that number.

    :param int index_range: Maximum value of the random index
    :param int index_skip: Index that is not allowed to be picked
    :return: Random index number
    """
    random_index = np.random.randint(index_range)
    while random_index == index_skip:
        random_index = np.random.randint(index_range)
    return random_index


def get_index_closest(value, data) -> int:
    """
    Returns the index of the closest value at the given value.

    :param list or tuple data: Container with values
    :param float or int value: Any value that has some
    relation to the given container
    :return: Index at closest value
    """
    item = min(data, key=lambda x: abs(x - value))
    return data.index(item)
