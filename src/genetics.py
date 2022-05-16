import numpy.random

import controllers
from mutations import mutate_io_controller_random

ALPHABET = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split(' ')
ALPHABET_EXTENDED = [a for a in ALPHABET]
ALPHABET_EXTENDED += [
    a + b for a in ALPHABET for b in ALPHABET
]
ALPHABET_EXTENDED += [
    c + d + e for c in ALPHABET
    for d in ALPHABET for e in ALPHABET
]
ALPHABET_EXTENDED += [
    c + d + e + f for c in ALPHABET
    for d in ALPHABET for e in ALPHABET
    for f in ALPHABET for f in ALPHABET
]

"""
# It is time to do elitism, replication,
# crossover, and mutations to models

# Grace of god to unique models which
# had had success but got outmatched.
# Grace of god: 10%

# Peaceful mode
# Every species will go to the next stage
# but some will change
# Elitism: 10%
# Replication: 20%
# Breeding: 30%
# Parents get kids, kids go further
# Mutation: 40%

# Survival mode
# Some species will go extinct
# Parents are from the replications
# Elitism: 10%
# Replication: 20%
# Breeding: 30%
# Mutation: 40%
"""


class RotatingList:
    def __init__(self, data):
        self.data = data
        self.index = -1

    def next(self):
        self.index += 1
        if self.index > len(self.data):
            self.index = 0
        return self.data[self.index]


ROTATING_ALPHABET = RotatingList(ALPHABET_EXTENDED)


class ModelManager:
    """
    Class holds container of :class:`InOutModel` and handles the
    process of switching between different io configurations.
    """

    def __init__(self):
        self.models: list[controllers.InOutModel] = []

    def add_model(self, controller: controllers.InOutModel) -> None:
        """
        Adds an io controller to the manager configuration.

        :param controller: Any instance of InOutController
        :return: None
        """
        self.models.append(controller)

    def get_models(self) -> list[tuple]:
        """
        Returns the controls of all io models
        inside the manager configuration.

        :return: List of the current controller configuration
        """
        return [c.get_model() for c in self.models]

    def set_models(self, controls: list[tuple]) -> None:
        """
        Changes the controls of all io models which are
        inside the manager configuration.

        :param controls: List of new controller configurations
        :return: None
        """
        for controller, control in zip(self.models, controls):
            controller.set_model(control)

    def reset(self) -> None:
        """
        Resets the io models to start state.

        :return: None
        """
        for controller in self.models:
            controller.reset()

    def get_string(self) -> str:
        """
        Returns a representation of the current controller configuration.

        :return: Current controller configuration
        """
        text = '['
        for c in self.get_models():
            text += controllers.get_tuple_string(c) + ', '
        return text + ']'


class GeneticEvolutionController(controllers.LearningController,
                                 controllers.BaseManager):
    def __init__(self, size=10):
        controllers.LearningController.__init__(self, 'Genetic')
        self.genetic_population: list[dict] = [{
            'name': ROTATING_ALPHABET.next() + ':0',
            'gene': [],
            'rank': 1,
            'rewards': [],
        } for _ in range(size)]
        self.model_manager = ModelManager()
        self.genetic_index = -1
        self.rewards = []

    def explore(self) -> None:
        if self.genetic_index + 1 < len(self.genetic_population):
            self.next_controller()
            return
        ranked_pool = sorted(
            self.genetic_population, key=lambda g: g['rank'])
        progress_pool = []

        population = len(self.genetic_population)
        elitism = int(population * 0.1)
        breeding = int(population * 0.3)
        mutation = int(population * 0.4)
        replication = int(population * 0.2)

        while elitism + replication + breeding + mutation != population:
            rng = numpy.random.rand()
            if rng < 0.1:
                elitism += 1
                continue
            rng += -0.1
            if rng < 0.2:
                replication += 1
                continue
            rng += -0.2
            if rng < 0.3:
                breeding += 1
                continue
            rng += -0.3
            if rng < 0.4:
                mutation += 1
                continue

        genetics_replication(ranked_pool, progress_pool, replication)
        genetics_mutation(ranked_pool, progress_pool, mutation)
        genetics_breeding(ranked_pool, progress_pool, breeding)
        genetics_elitism(ranked_pool, progress_pool, elitism)
        self.genetic_population = progress_pool
        self.next_controller()

    def reflect(self) -> None:
        genetic_info = self.genetic_population[self.genetic_index]
        genetic_info['rewards'] = self.rewards
        self.rewards = []

        if self.genetic_index + 1 < len(self.genetic_population):
            return

        rewards = [g['rewards'] for g in self.genetic_population]
        max_reward = [max(r) for r in rewards]
        highest, lowest = max(max_reward), min(max_reward)
        genetics_rank = [(r - lowest) / (highest - lowest)
                         for r in max_reward]
        for genetic, rank in zip(self.genetic_population, genetics_rank):
            genetic['rank'] = rank
            genetic['rewards'] = []

    def resize_genetic_population(self, size):
        genetic_base = self.model_manager.get_models()
        for _ in range(size):
            self.genetic_population.append({
                'name': ROTATING_ALPHABET.next() + ':0',
                'gene': randomize_genetics(genetic_base),
                'rank': 1,
                'rewards': [],
            })
        self.next_controller()

    def add_controller(self, controller) -> None:
        """
        Adds an io controller to the genetic setup.

        :param controllers.InOutModel controller:
        """
        self.model_manager.add_model(controller)
        io_base = controller.get_model()
        for genetic in self.genetic_population:
            genetic['gene'].append(randomize_genetic_io(io_base))

    def next_controller(self):
        self.select_controller(self.genetic_index + 1)

    def select_controller(self, index):
        if not index < len(self.genetic_population):
            index = 0
        self.genetic_index = index
        genetic_io = self.genetic_population[index]
        self.model_manager.set_models(genetic_io['gene'])
        self.name = genetic_io['name']

    def get_size(self):
        return len(self.genetic_population)

    def reward(self, reward):
        self.rewards.append(reward)

    def reset(self) -> None:
        self.model_manager.reset()

    def get_string(self) -> str:
        return self.model_manager.get_string()


def genetics_elitism(ranked_pool, progress_pool, elitism):
    ranked_pool.reverse()
    for genetic_info in ranked_pool:
        progress_pool.append(genetic_info)
        elitism += -1
        if elitism > 0:
            continue
        break


def genetics_replication(ranked_pool, progress_pool, replication):
    while replication > 0:
        for genetic_info in ranked_pool:
            if numpy.random.rand() > genetic_info['rank']:
                continue
            if numpy.random.rand() > 0.2:
                continue
            progress_pool.append(genetic_info)
            replication += -1
            if replication > 0:
                continue
            break


def genetics_breeding(ranked_pool, progress_pool, breeding):
    while breeding > 0:
        pool_a = generate_breeding_pool(ranked_pool)
        pool_b = generate_breeding_pool(ranked_pool)
        for parent_a, parent_b in zip(pool_a, pool_b):
            if parent_a is parent_b:
                continue

            name_info_a: str = parent_a['name']
            info_a = name_info_a.split(':')[0]
            name_a = info_a.split('=')[0]

            name_info_b: str = parent_b['name']
            info_b = name_info_b.split(':')[0]
            name_b = info_b.split('=')[0]

            new_info = '=' + name_a + '+' + name_b + ':0'
            genetics = breed_genetics(
                parent_a['gene'], parent_b['gene'])
            progress_pool.append({
                'name': ROTATING_ALPHABET.next() + new_info,
                'gene': genetics,
                'rank': 0,
                'rewards': [],
            })
            breeding += -1
            if breeding > 0:
                continue
            break


def genetics_mutation(ranked_pool, progress_pool, mutation):
    while mutation > 0:
        for genetic_info in ranked_pool:
            if numpy.random.rand() > genetic_info['rank']:
                continue
            if numpy.random.rand() > 0.2:
                continue
            genetics = genetic_info['gene']
            genetics = mutate_genetics(genetics)
            name_info: str = genetic_info['name']
            info, iteration = name_info.split(':')
            iteration = ':' + str(int(iteration) + 1)
            progress_pool.append({
                'name': info + iteration,
                'gene': genetics,
                'rank': 0,
                'rewards': [],
            })
            mutation += -1
            if mutation > 0:
                continue
            break


def generate_breeding_pool(ranked_pool):
    pool_a = []
    for genetic_info in ranked_pool:
        if numpy.random.rand() > genetic_info['rank']:
            continue
        if numpy.random.rand() > 0.2:
            continue
        pool_a.append(genetic_info)
    return pool_a


def breed_genetics(genetics_a, genetics_b):
    genetics_new = []
    for a, b in zip(genetics_a, genetics_b):
        if numpy.random.rand() > 0.5:
            genetics_new.append(a)
        else:
            genetics_new.append(b)
    return genetics_new


def mutate_genetics(genetic_base):
    index = numpy.random.randint(len(genetic_base))
    io_controller = genetic_base[index]
    genetic_base[index] = mutate_io_controller_random(io_controller)
    return genetic_base


def randomize_genetics(genetic_base):
    return [
        randomize_genetic_io(base)
        for base in genetic_base
    ]


def randomize_genetic_io(base_io):
    for _ in range(10):
        base_io = mutate_io_controller_random(base_io)
    return base_io
