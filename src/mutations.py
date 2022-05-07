import numpy

import settings

import controllers


def mutate_io_controller(current, previous, io_type=''):
    """
    Compares, modifies, and returns new tuple where a single float of the tuple has been changed.
    Decides randomly wheather the controller should improve or explore the settings.
    At the start the exploring option is picked frequently than the improving option.

    :param tuple or list current: Current controller configuration
    :param tuple or list previous: Previous controller configuration
    :param str io_type: Type of the in out controller, defaults to empty string
    :return: Tuple with a single modified value
    """
    if type(current) is not list:
        current = list(current)
    current_index = controllers.get_index_changed(current, previous)

    # Explore or improve, at the start it will explore more
    if numpy.random.rand() > settings.EPSILON:
        # Improve the previously changed controller setting
        improve = get_improved_float(current, previous, current_index)
        current[current_index] += improve
        return tuple(current)

    # Random explore controller setting at index that has not been changed
    random_index = controllers.get_index_random(len(current), current_index)
    return mutate_io_controller_random(current, random_index, io_type)


def mutate_io_controller_random(controller, index=-1, io_type=''):
    """
    Compares, modifies, and returns new tuple where a single float of the tuple has been changed.
    When index is specified it changes the value of the given position
    otherwise, it will take a random index.

    :param tuple or list controller: Current controller configuration
    :param int index: Place of the value which needs to be changed
    :param str io_type: Type of the in out controller
    :return: Tuple with a changed value
    """
    if type(controller) is not list:
        controller = list(controller)
    if index == -1:
        index = numpy.random.randint(len(controller))
    multiplier = get_io_multiplier(index, io_type)
    controller[index] += get_random_float() * multiplier
    return tuple(controller)


def get_io_multiplier(index, io_type):
    """
    Returns a multiplier value at given index if the controller is a PID controller.
    Otherwise, it returns 1.

    :param int index: Index of PID controller
    :param io_type: Type of the in out controller
    :return: Multiplier based on controller and index
    """
    if io_type == 'pid':
        multiplier = (5, 0.1, 2)[index]
    else:
        multiplier = 1
    return multiplier


def get_improved_float(current, previous, index):
    """
    Returns a slightly improved float at index. Takes float at the
    index of new and old controller to calculate a difference.
    Improvement is the difference times the multipliers of improve and epsilon.

    :param tuple or list current: Current controller
    :param tuple or list previous: Previous controller
    :param int index: Position index of both data sets
    :return: Improved float at given index
    :rtype: float
    """
    improve = settings.MULTIPLIER_IMPROVE
    difference = (current[index] - previous[index])
    return difference * improve


def get_random_float():
    """
    Returns a random float between -0.5 and 0.5 times
    the multipliers of improve and epsilon.

    :return: Random float
    :rtype: float
    """
    improve = settings.MULTIPLIER_RANDOM * settings.MULTIPLIER_EPSILON
    return (numpy.random.rand() - 0.5) * improve
