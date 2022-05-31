import gym
import math

from src import controllers
from src import settings


class Pendulum(controllers.EnvironmentWorker):
    def get_reward(self, observation: gym.core.ObsType) -> float:
        # reward -= abs(observation[0] - 0.50)
        # reward += abs(observation[1] * 100)
        return 0

    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        output = 0
        n_point_x = node_point_x.get_output(observation, 0)
        n_point_y = node_point_y.get_output(observation, 0)
        # n_direct = node_direct.get_output(observation, 0)
        n_pendulum_x = node_pendulum_x.get_output(observation, 0)
        n_pendulum_y = node_pendulum_x.get_output(observation, 0)
        target_x = pid_point_x.get_output((n_point_x,), 0.0)
        target_y = pid_point_y.get_output((n_point_y,), 1.0)
        # output += pid_direct.get_output((n_direct,), 1.0)
        output += pid_pendulum_x.get_output((n_pendulum_x,), target_x)
        output += pid_pendulum_y.get_output((n_pendulum_y,), target_y)
        action = action_space(output)
        return action


def action_space(output):
    action = math.tanh(output) * 2
    return action,


settings.EPISODE.recalculate(10)
settings.EPISODE.CAP *= 10

PID_POINT_X = (0, 0, 0)
PID_POINT_Y = (0, 0, 0)
PID_PENDULUM_X = (0, 0, 0)
PID_PENDULUM_Y = (0, 0, 0)
PID_DIRECT = (0, 0, 0)
NODE_POINT_X = (0, 0, 0)
NODE_POINT_Y = (0, 0, 0)
NODE_PENDULUM_X = (0, 0, 0)
NODE_PENDULUM_Y = (0, 0, 0)
NODE_DIRECT = (0, 0, 0)

pid_point_x: controllers.IOModel
pid_point_y: controllers.IOModel
pid_direct: controllers.IOModel
pid_pendulum_x: controllers.IOModel
pid_pendulum_y: controllers.IOModel
node_point_x: controllers.IOModel
node_point_y: controllers.IOModel
node_direct: controllers.IOModel
node_pendulum_x: controllers.IOModel
node_pendulum_y: controllers.IOModel
manager: controllers.BaseManager | \
         controllers.LearningController


def generate_improving_model():
    global manager, pid_point_x, pid_point_y, node_point_x, node_point_y
    global pid_pendulum_x, pid_pendulum_y, pid_direct, node_pendulum_x, node_pendulum_y, node_direct
    pid_point_x = controllers.ImprovingPIDModel('PID_POINT_X', PID_POINT_X)
    pid_pendulum_x = controllers.ImprovingPIDModel('PID_PENDULUM_X', PID_PENDULUM_X)
    pid_point_y = controllers.ImprovingPIDModel('PID_POINT_Y', PID_POINT_Y)
    pid_pendulum_y = controllers.ImprovingPIDModel('PID_PENDULUM_Y', PID_PENDULUM_Y)
    pid_direct = controllers.ImprovingPIDModel('PID_DIRECT', PID_DIRECT)
    node_point_x = controllers.ImprovingNodeModel('NODE_POINT_X', NODE_POINT_X)
    node_point_y = controllers.ImprovingNodeModel('NODE_POINT_Y', NODE_POINT_Y)
    node_pendulum_x = controllers.ImprovingNodeModel('NODE_PENDULUM_X', NODE_PENDULUM_X)
    node_pendulum_y = controllers.ImprovingNodeModel('NODE_PENDULUM_Y', NODE_PENDULUM_Y)
    node_direct = controllers.ImprovingNodeModel('NODE_DIRECT', NODE_DIRECT)
    manager = controllers.ImprovingControllerManager()
    # manager.add_controller(pid_direct)
    manager.add_controller(pid_pendulum_x)
    manager.add_controller(pid_pendulum_y)
    manager.add_controller(pid_point_x)
    manager.add_controller(pid_point_y)
    # manager.add_controller(node_direct)
    manager.add_controller(node_pendulum_x)
    manager.add_controller(node_pendulum_y)
    manager.add_controller(node_point_x)
    manager.add_controller(node_point_y)
    pid_pendulum_x = node_pendulum_x.model
    pid_pendulum_y = node_pendulum_y.model
    # pid_direct = node_direct.model
    pid_point_x = pid_point_x.model
    pid_point_y = pid_point_y.model
    node_pendulum_x = node_pendulum_x.model
    node_pendulum_y = node_pendulum_y.model
    # node_direct = node_direct.model
    node_point_x = node_point_x.model
    node_point_y = node_point_y.model


def main():
    generate_improving_model()
    env = gym.make('Pendulum-v1')
    environment = controllers.EnvironmentManager(env, manager, Pendulum(env))

    environment.run()
    for i in range(manager.get_size()):
        manager.select_controller(i)
        print(manager.name, '=', manager.get_string())


if __name__ == '__main__':
    main()
