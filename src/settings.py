"""
This module holds the variables for the whole project.

Changing one variable in this file affects all the scripts.
Better modify the variable inside the script by importing
this file as a module.
"""

EPISODE_MULTIPLIER = 10
EPISODE_CAP = EPISODE_MULTIPLIER * 1000
EPISODE_SHOW = EPISODE_MULTIPLIER * 100
EPISODE_PRINT = EPISODE_MULTIPLIER * 10
EPISODE_PRINT_TOGGLE = False
EPISODE_RENDER = 50

TIME_STEPS = 200
EPISODE_LEARN = 20

EPSILON = 0.9
EPSILON_CAP = 0.05
EPSILON_DISCOUNT = 0.95
EPSILON_DECAY_RATE = EPSILON_DISCOUNT ** (10 / EPISODE_CAP)

MULTIPLIER_EPSILON = 10
MULTIPLIER_IMPROVE = 0.8
MULTIPLIER_RANDOM = 0.1

IMPROVEMENT_THRESHOLD = 0.95
IMPROVEMENT_THRESHOLD_RNG = 1.0 - IMPROVEMENT_THRESHOLD

MODULE_DICT_SKIP = ('EPISODE_MULTIPLIER', 'EPSILON_DECAY_RATE',)


def recalculate_improvement_threshold(minimum):
    """
    Recalculates the threshold for random improvement.

    :param minimum: Threshold minimum value
    """
    global IMPROVEMENT_THRESHOLD, IMPROVEMENT_THRESHOLD_RNG
    IMPROVEMENT_THRESHOLD = minimum
    IMPROVEMENT_THRESHOLD_RNG = 1.0 - IMPROVEMENT_THRESHOLD


def recalculate_episode(multiplier):
    """
    Recalculates the threshold for EPISODE_CAP,
    EPISODE_SHOW, EPISODE_PRINT, and EPSILON_DECAY_RATE
    with the given multiplier.

    :param multiplier: Multiplier of how many episodes need to run
    :return: None
    """
    global EPISODE_CAP, EPISODE_SHOW, EPISODE_PRINT, EPSILON_DECAY_RATE
    global EPISODE_MULTIPLIER
    EPISODE_MULTIPLIER = multiplier
    EPISODE_CAP = EPISODE_MULTIPLIER * 1000
    EPISODE_SHOW = EPISODE_MULTIPLIER * 100
    EPISODE_PRINT = EPISODE_MULTIPLIER * 10
    EPSILON_DECAY_RATE = EPSILON_DISCOUNT ** (10 / EPISODE_CAP)


def get_dict():
    """
    Processes variables to return copy of the module as dictionary.

    :return: A copy of the modules variables
    """
    return {
        'EPISODE_CAP': EPISODE_CAP,
        'EPISODE_SHOW': EPISODE_SHOW,
        'EPISODE_PRINT': EPISODE_PRINT,
        'EPISODE_PRINT_TOGGLE': EPISODE_PRINT_TOGGLE,
        'EPISODE_RENDER': EPISODE_RENDER,

        'TIME_STEPS': TIME_STEPS,
        'EPISODE_LEARN': EPISODE_LEARN,

        'EPSILON': EPSILON,
        'EPSILON_CAP': EPSILON_CAP,
        'EPSILON_DISCOUNT': EPSILON_DISCOUNT,
        # 'EPSILON_DECAY_RATE': EPSILON_DECAY_RATE,

        'MULTIPLIER_EPSILON': MULTIPLIER_EPSILON,
        'MULTIPLIER_IMPROVE': MULTIPLIER_IMPROVE,
        'MULTIPLIER_RAND': MULTIPLIER_RANDOM,
    }


def set_dict(info):
    """
    Processes dictionary values and update the modules variable
    values to match the passed dictionary values.

    :param dict[str, int | float] info: Modified copy of this module
    :return: None
    """
    for id, value in info:
        if id in MODULE_DICT_SKIP:
            continue
        globals()[id] = value
