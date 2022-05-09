import gym
import settings
import controllers


class Pendulum(controllers.EnvironmentWorker):
    def get_reward(self, observation: gym.core.ObsType) -> float:
        # reward -= abs(observation[0] - 0.50)
        # reward += abs(observation[1] * 100)
        return 0

    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        output = 0
        n_point = node_point.get_output(observation, 0)
        n_direct = node_direct.get_output(observation, 0)
        n_pendulum = node_pendulum.get_output(observation, 0)
        target = pid_point.get_output((n_point,), 1.0)
        output += pid_direct.get_output((n_direct,), 1.0)
        output += pid_pendulum.get_output((n_pendulum,), target)
        action = action_space(output)
        return action


def action_space(output):
    if output > 2:
        action = 2
    elif output < -2:
        action = -2
    else:
        action = output
    return [action]


settings.recalculate(10)
settings.EPISODE_CAP *= 10

PID_POINT = (0, 0, 0)
PID_PENDULUM = (0, 0, 0)
PID_DIRECT = (0, 0, 0)

NODE_POINT = (0, 0, 0)
NODE_PENDULUM = (0, 0, 0)
NODE_DIRECT = (0, 0, 0)

pid_point = controllers.ImprovingPIDModel('PID_POINT', PID_POINT)
pid_pendulum = controllers.ImprovingPIDModel('PID_PENDULUM', PID_PENDULUM)
pid_direct = controllers.ImprovingPIDModel('PID_DIRECT', PID_DIRECT)
node_point = controllers.ImprovingNodeModel('NODE_POINT', NODE_POINT)
node_pendulum = controllers.ImprovingNodeModel('NODE_PENDULUM', NODE_PENDULUM)
node_direct = controllers.ImprovingNodeModel('NODE_DIRECT', NODE_DIRECT)
manager = controllers.ImprovingControllerManager()
manager.add_controller(pid_direct)
manager.add_controller(pid_pendulum)
manager.add_controller(pid_point)
manager.add_controller(node_direct)
manager.add_controller(node_pendulum)
manager.add_controller(node_point)


def main():
    env = gym.make('Pendulum-v1')
    environment = controllers.EnvironmentManager(env, manager, Pendulum(env))

    environment.run()
    for i in range(manager.get_size()):
        manager.select_controller(i)
        print(manager.name, '=', manager.get_string())


if __name__ == '__main__':
    main()
