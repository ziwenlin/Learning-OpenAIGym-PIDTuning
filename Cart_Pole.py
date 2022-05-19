import gym
import numpy.random

from src import settings
from src import controllers


class CartPole(controllers.EnvironmentWorker):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode = 0
        self.progress = 0
        self.position = 0.0

    def reset(self, seed=None):
        if seed is not None:
            numpy.random.seed(seed)
        self.episode += 1
        episode = self.episode
        intervals = settings.EPISODE_LEARN
        progress = (episode // intervals) * intervals
        unluck = numpy.random.random() * 2 - 1
        self.progress = progress / settings.EPISODE_CAP
        self.difficulty = self.progress * unluck
        self.position = 2 * self.difficulty

    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        output = 0
        position = pid_point.get_output((node_point.get_output(observation, 0),), self.position)
        output += pid_pole.get_output((node_pole.get_output(observation, 0),), 0.0)
        output += pid_cart.get_output((node_cart.get_output(observation, 0),), position)
        return action_space(output)

    def get_reward(self, observation: gym.core.ObsType) -> float:
        reward = abs(self.difficulty)
        cart_p, cart_v, pole_p, pole_v = observation
        if manager.name in ('PID_POLE', 'NODE_POLE'):
            reward -= abs(pole_p * 0.80) ** 0.5
        elif manager.name in ('PID_CART', 'NODE_CART'):
            reward -= abs(cart_p - self.position) ** 0.5
        elif manager.name in ('PID_POINT', 'NODE_POINT'):
            reward -= abs(cart_p - self.position) ** 0.5
        else:
            reward -= abs(cart_p - self.position) ** 0.5
            reward -= abs(pole_p * 0.80) ** 0.5
        return reward


def action_space(output):
    if output > 0:
        action = 1
    else:
        action = 0
    return action


settings.EPISODE_CAP *= 10
settings.TIME_STEPS = 500

PID_CART = (-0.5298, 0.4768, -0.488)
PID_CART = (-0.2419, 1.0445, -0.393)
PID_CART = (-0.4752, 1.1208, -0.393)
PID_CART = (0, 0, 0)
PID_CART = (-0.2195, 0.7622, 0.0)
PID_CART = (-0.2931, 1.2504, 0.0)
PID_CART = (-0.5055, 1.3731, 0.4458)
PID_CART = (0, 0, 0)

PID_POLE = (-5.7916, 4.8044, -7.0568)
PID_POLE = (-5.7777, 4.8044, -6.6541)
PID_POLE = (-6.1015, 5.4192, -6.6313)
PID_POLE = (-6.3068, 5.3475, -7.2753)
PID_POLE = (-7.4381, 5.8662, -7.0927)
PID_POLE = (-8.1293, 6.5284, -6.528)
PID_POLE = (-7.8758, 6.8384, -6.6159)
PID_POLE = (0, 0, 0)

PID_POINT = (-0.3501, -0.0706, 0.2518)
PID_POINT = (-0.179, -0.0706, 0.3005)
PID_POINT = (0, 0, 0)

NODE_POLE = (0, 0, 0, 0)
NODE_CART = (0, 0, 0, 0)
NODE_POINT = (0, 0, 0, 0)

pid_pole: controllers.IOModel
pid_cart: controllers.IOModel
pid_point: controllers.IOModel
node_pole: controllers.IOModel
node_cart: controllers.IOModel
node_point: controllers.IOModel
manager: controllers.BaseManager | \
         controllers.LearningController


def generate_improving_model():
    global manager, pid_cart, pid_point, node_cart, node_point
    global pid_pole, node_pole
    pid_pole = controllers.ImprovingPIDModel('PID_POLE', PID_POLE)
    pid_cart = controllers.ImprovingPIDModel('PID_CART', PID_CART)
    pid_point = controllers.ImprovingPIDModel('PID_POINT', PID_POINT)
    node_pole = controllers.ImprovingNodeModel('NODE_POLE', NODE_POLE)
    node_cart = controllers.ImprovingNodeModel('NODE_CART', NODE_CART)
    node_point = controllers.ImprovingNodeModel('NODE_POINT', NODE_POINT)
    manager = controllers.ImprovingControllerManager()
    manager.add_controller(pid_cart)
    manager.add_controller(pid_pole)
    manager.add_controller(pid_point)
    manager.add_controller(node_cart)
    manager.add_controller(node_pole)
    manager.add_controller(node_point)
    pid_pole = pid_pole.model
    pid_cart = pid_cart.model
    pid_point = pid_point.model
    node_pole = node_pole.model
    node_cart = node_cart.model
    node_point = node_point.model


def main():
    generate_improving_model()
    env = gym.make('CartPole-v1')
    environment = controllers.EnvironmentManager(env, manager, CartPole(env))
    # environment.controller.episode = environment.episode = Settings.EPISODE_CAP // 10 * 7

    environment.run()
    for i in range(manager.get_size()):
        manager.select_controller(i)
        print(manager.name, '=', manager.get_string())


def main1():
    env = gym.make('CartPole-v1')
    env.reset(seed=1)
    for _ in range(1000):
        env.render()
        action = 1
        observation, reward, done, info = env.step(action)
        if done:
            env.reset(seed=1)
    env.close()


if __name__ == '__main__':
    main()
