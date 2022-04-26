import gym
import settings
import controllers


class Pendulum(controllers.EnvironmentController):
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

pid_point = controllers.LearningPIDController('PID_POINT', PID_POINT)
pid_pendulum = controllers.LearningPIDController('PID_PENDULUM', PID_PENDULUM)
pid_direct = controllers.LearningPIDController('PID_DIRECT', PID_DIRECT)
node_point = controllers.LearningNodeController('NODE_POINT', NODE_POINT)
node_pendulum = controllers.LearningNodeController('NODE_PENDULUM', NODE_PENDULUM)
node_direct = controllers.LearningNodeController('NODE_DIRECT', NODE_DIRECT)
learning_agent = controllers.RotatingLearningController()
learning_agent.add_controller(pid_direct)
learning_agent.add_controller(pid_pendulum)
learning_agent.add_controller(pid_point)
learning_agent.add_controller(node_direct)
learning_agent.add_controller(node_pendulum)
learning_agent.add_controller(node_point)


def main():
    env = gym.make('Pendulum-v1')
    environment = controllers.EnvironmentRunner(env, learning_agent, Pendulum(env))

    environment.start()
    while environment.running:
        try:
            environment.step_episode()
            environment.step_end()
        except KeyboardInterrupt:
            break
    environment.stop()


if __name__ == '__main__':
    main()
