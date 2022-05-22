import abc
import statistics
import textwrap
import time

import gym
import numpy as np
from tabulate import tabulate

from src import settings
from src.mutations import mutate_io_model


class IOModel:
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
    """
    Base class that provides the interface for any learning controllers.
    Used in machine learning environments to train agents.

    Available methods:
    :method:`explore()`,
    :method:`reflect()`,
    :method:`reward()`,
    :method:`reset()`,
    :method:`get_string()`
    """

    @abc.abstractmethod
    def __init__(self, name=''):
        self.name = name

    @abc.abstractmethod
    def explore(self) -> None:
        """
        After calling the reflect method the explore method
        gets called. This method manages the configuration.

        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reflect(self) -> None:
        """
        After every nth episode reflect method gets called
        to look for any results and processes them.

        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self, reward: float) -> None:
        """
        This method gets called after every episode and
        stores the received rewards.

        :param reward: Reward of a episode
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """
        This method gets called after receiving the rewards
        and resets configuration attributes to the starting state.

        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_string(self) -> str:
        """
        Returns a representation of the current
        learning controller configuration.
        """
        raise NotImplementedError


class BaseManager:
    """
    Base class that stores and manages controllers.

    Available methods:
    :method:`add_controller()`,
    :method:`select_controller()`,
    :method:`next_controller()`,
    :method:`get_size()`
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
    """
    Interface class which delivers the input values to the models,
    converts the output of the model to the action space of
    the environment, and calculates additional rewards.

    Available methods:
    :method:`reset()`,
    :method:`get_action()`,
    :method:`get_reward()`,
    """

    def __init__(self, env: gym.Env):
        self.action_space = env.action_space
        self.difficulty = 0

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets all attributes before proceeding to the next
        episode in the gym environment.

        :return: None
        """
        self.difficulty = 0

    @abc.abstractmethod
    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        """
        Returns action which is processed by the gym environment.

        :param observation: Gym observation object
        :return: Gym action object
        """
        return self.action_space.sample()

    @abc.abstractmethod
    def get_reward(self, observation: gym.core.ObsType) -> float:
        """
        Returns reward by processing observation values into
        rewards.

        :param observation: Gym observation object
        :return: Reward value
        """
        return 0


class PIDModel(IOModel):
    """
    Implements from :class:`IOModel`. Uses PID based model
    to calculate a output based on the input values.


    """

    def __init__(self, preset):
        super().__init__()
        self.value_d = 0
        self.value_i = 0

        self.model_p, self.model_i, self.model_d = preset

    def set_model(self, model: tuple) -> None:
        self.model_p, self.model_i, self.model_d = model

    def get_model(self) -> tuple:
        return self.model_p, self.model_i, self.model_d

    def get_output(self, values: tuple, target: float) -> float:
        error = target - sum(values)

        p = self.model_p * error
        i = self.model_i * (error + self.value_i)
        d = self.model_d * (error - self.value_d)

        self.value_i += error
        self.value_d = error
        self.output = p + i + d
        return self.output

    def reset(self) -> None:
        self.output = 0
        self.value_d = 0
        self.value_i = 0


class NodeModel(IOModel):
    """
    Implements from :class:`IOModel`. Uses weight based model
    to calculate a output based on the input values.
    """

    def __init__(self, preset):
        super().__init__()
        self.control = preset

    def set_model(self, model: tuple) -> None:
        self.control = tuple(model)

    def get_model(self) -> tuple:
        return tuple(self.control)

    def get_output(self, values: tuple, target: float) -> float:
        output = target
        for weight, value in zip(self.control, values):
            output += weight * value
        self.output = output
        return output

    def reset(self) -> None:
        self.output = 0


class ImprovingController(LearningController):
    """
    Implements from :class:`LearningController` and is base class
    that provides an interface for improvement based learning.
    Uses reward based calculations to decide whether improvements
    have been made.
    """

    def __init__(self, name=''):
        super().__init__(name)
        self.previous_rewards = []
        self.current_rewards = []
        self.is_improving = [False, False, False]

    def reward(self, reward: float) -> None:
        self.current_rewards.append(reward)

    def reflect(self) -> None:
        previous_rewards = self.previous_rewards
        current_rewards = self.current_rewards
        self.is_improving = get_is_improving(
            current_rewards, previous_rewards
        )
        self.previous_rewards = self.current_rewards
        self.current_rewards = []

    @abc.abstractmethod
    def explore(self) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def get_string(self) -> str:
        return self.name


class ImprovingModelController(ImprovingController):
    """
    Implements from :class:`IOModel` and :class:`ImprovingController`
    and is base class that provides improvement based learning to any
    instances of :class:`IOModel`. Uses reward based calculations to
    decide whether improvements have been made.
    """

    def __init__(self, name='', preset=(0, 0, 0)):
        ImprovingController.__init__(self, name)
        self.model = IOModel()
        self.current_model = preset
        self.previous_model = preset

    def get_string(self) -> str:
        return get_tuple_string(self.model.get_model())

    def explore(self) -> None:
        self.current_model = mutate_io_model(
            self.current_model, self.previous_model)
        self.model.set_model(self.current_model)

    def reflect(self) -> None:
        previous_rewards = self.previous_rewards
        current_rewards = self.current_rewards
        ImprovingController.reflect(self)

        is_improving = get_is_improving_random(
            self.is_improving, current_rewards, previous_rewards)
        avg, low, high = is_improving

        if low and avg or low and high or avg and high:
            # When the newer model has scored an equal or better score
            # Overwrite the previous reward and model
            self.previous_rewards = current_rewards
            self.previous_model = self.current_model
        else:
            # Revert the changes
            # Restore previous reward and current model
            self.previous_rewards = previous_rewards
            self.current_model = self.previous_model

        self.model.set_model(self.current_model)
        self.current_rewards = []

    def reward(self, reward: float) -> None:
        super().reward(reward)

    def reset(self) -> None:
        self.model.reset()


class ImprovingPIDModel(ImprovingModelController):
    """
    Implements from :class:`ImprovingModelController` and :class:`PIDModel`
    and uses reward based calculations to
    decide whether improvements have been made.
    """

    def __init__(self, name='', preset=(0, 0, 0)):
        ImprovingModelController.__init__(self, name, preset)
        self.model = PIDModel(preset)

    def explore(self) -> None:
        self.current_model = mutate_io_model(
            self.current_model, self.previous_model, 'pid')
        self.model.set_model(self.current_model)


class ImprovingNodeModel(ImprovingModelController):
    """
    Implements from :class:`ImprovingModelController` and :class:`NodeModel`
    and uses reward based calculations to
    decide whether improvements have been made.
    """

    def __init__(self, name='', preset=None):
        ImprovingModelController.__init__(self, name, preset)
        self.model = NodeModel(preset)
        if preset is None:
            raise ValueError('Please provide a preset')


class LearningControllerManager(BaseManager):
    """
    Implements :class:`BaseManager` which stores and manages
    multiple instances of :class:`LearningController`. The
    class keeps track of selected controller and provides
    rotating select mechanism to the manager.

    Available methods:
    :method:`add_controller()`,
    :method:`select_controller()`,
    :method:`next_controller()`,
    :method:`get_size()`
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


class ImprovingControllerManager(ImprovingController,
                                 LearningControllerManager):
    """
    Implements from :class:`ImprovingController`
    and :class:`LearningControllerManager`
    and uses reward based calculations to
    decide whether a controller should be learning
    or the next controller should be selected for learning.
    """

    def __init__(self):
        ImprovingController.__init__(self, 'Manager')
        LearningControllerManager.__init__(self)
        self.is_rotating = True
        self.is_next = False
        self.count = 0

    def select_controller(self, index: int) -> None:
        super().select_controller(index)
        self.name = self.selected.name
        self.previous_rewards = []
        self.is_next = False
        self.count = 0

    def explore(self) -> None:
        if self.is_rotating and self.is_next:
            self.next_controller()
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

    def reset(self) -> None:
        for controller in self.controllers:
            controller.reset()


class EnvironmentMonitor:
    """
    Class monitors the episode values, processes the
    values into high, low, average, etc..., and generates
    logs with representation of the values in a table.
    """

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
            if category not in division[0]:
                # Prevent key error of categories
                # which do not exist in division
                return
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
        get_result('steps')
        get_result('difficulty')
        get_result('episode')

        result['episodes'] = episode
        result['highest'] = round(highest_value, 2)
        result['average'] = round(mean_value, 2)
        result['lowest'] = round(lowest_value, 2)
        result['median'] = round(median_value, 2)
        result['middle'] = round(middle_value, 2)
        result['epsilon'] = round(settings.EPSILON.VALUE, 3)
        result['multiplier'] = round(settings.MULTIPLIER_EPSILON, 3)


class EnvironmentManager:
    """
    Class manages the environment, episodes, agent, worker
    and all components' method calls.
    """

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
        self.steps = 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        self.env.close()

    def step_epsilon(self):
        if settings.MULTIPLIER_EPSILON > settings.EPSILON.CAP:
            settings.MULTIPLIER_EPSILON *= settings.EPSILON.DECAY_RATE
        if settings.EPSILON.VALUE > settings.EPSILON.CAP:
            settings.EPSILON.VALUE *= settings.EPSILON.DECAY_RATE

    def step_monitor(self):
        self.logger.monitor({
            'episode': self.episode,
            'reward': self.rewards,
            'difficulty': self.worker.difficulty,
            'steps': self.time_steps,
        })

    def step_print(self):
        episode, time_steps = self.episode, self.time_steps
        if not settings.EPISODE.PRINT_TOGGLE:
            pass
        elif (episode % settings.EPISODE.PRINT == 0
              or episode % settings.EPISODE.SHOW == 0):
            print("Episode {} finished after {} time steps"
                  .format(episode, time_steps))
        if episode % settings.EPISODE.SHOW == 0:
            log = self.logger.get_log()
            log += f'{self.agent.name} = {self.agent.get_string()}'
            print(log, '\n')

    def step_frame(self):
        frame_time = time.time()
        if frame_time < self.fps_time:
            return
        self.fps_time = frame_time + 0.1
        self.env.render()

    def step_agent(self):
        self.agent.reward(self.rewards)
        if self.episode % settings.EPISODE_LEARN == 0:
            self.logger.process(self.episode)
            self.agent.reflect()
            self.agent.explore()

    def step_render(self):
        if self.episode % settings.EPISODE.SHOW != 1:
            return
        # Get best episode to show
        time_steps, rewards, done = 1, 0, False
        observation = self.env.reset()
        self.worker.reset()
        self.agent.reset()
        while not done:
            self.env.render()
            action = self.worker.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            reward += self.worker.get_reward(observation)
            rewards += reward
            time_steps += 1
            if time_steps == settings.TIME_STEPS:
                break
        self.time_steps = time_steps
        self.rewards = rewards

    def step_episode(self):
        time_steps, rewards, done = 1, 0, False
        observation = self.env.reset()
        self.worker.reset()
        self.agent.reset()
        while not done:
            action = self.worker.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            reward += self.worker.get_reward(observation)
            rewards += reward
            time_steps += 1
            if time_steps == settings.TIME_STEPS:
                break
        self.time_steps = time_steps
        self.rewards = rewards

    def step_end(self):
        if self.episode > settings.EPISODE.CAP:
            self.stop()
        self.episode += 1

    def run(self):
        self.start()
        self.stop_event()
        while self.running:
            self.run_sequence()
        self.stop()

    def run_sequence(self):
        self.step_render()
        self.step_episode()
        self.step_frame()
        self.step_monitor()
        self.step_agent()
        self.step_print()
        self.step_end()
        self.step_epsilon()

    def run_once(self):
        self.episode = 1
        self.step_render()
        print("Episode {} finished after {} time steps"
              .format(1, self.time_steps))
        print("Collected rewards:", self.rewards)
        self.stop()

    def stop_event(self):
        import tkinter, threading
        def thread():
            root = tkinter.Tk()
            text = tkinter.Label(root, text='Close to stop training')
            text.pack()
            root.mainloop()
            self.running = False

        th = threading.Thread(target=thread, daemon=True)
        th.start()



def get_tuple_string(array: tuple) -> str:
    """
    Returns a reformatted tuple in text form.

    :param array: Collection of float values
    :return: Tuple in text
    """
    return f'{tuple(round(value, 4) for value in array)}'


def get_improvement_gain(current: float, previous: float) -> float:
    """
    Returns the improvement factor between the
    current value and previous value.

    :param current: Current values
    :param previous: Previous values
    :return: Improvement factor
    """
    if current > 0 and previous > 0:
        pass
    elif current > previous:
        current += abs(previous) + 2
        previous += abs(previous) + 1
    elif current < previous:
        previous += abs(current) + 2
        current += abs(current) + 1
    elif current == previous:
        current = 10
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
    threshold = settings.IMPROVEMENT_THRESHOLD
    threshold += settings.IMPROVEMENT_THRESHOLD_RNG * np.random.rand()
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
