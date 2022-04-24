import gym
import controllers


class MountainCar(controllers.EnvironmentController):
    position = 0.6

    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        output = 0
        position = point_agent.get_output(observation[0] + observation[1], self.position)
        output += cart_agent.get_output(observation[0] + observation[1], position)
        # if observation[1] < 0:
        #     output = cart_agent.get_output(observation[0], 0.6)
        # elif observation[1] > 0:
        #     output = cart_agent.get_output(observation[0], -1.2)
        action = action_space(output)
        return action

    def get_reward(self, observation: gym.core.ObsType) -> float:
        reward = 0
        if learning_agent.name == 'PID_CART':
            reward += abs(observation[1]) ** 0.5
        elif learning_agent.name == 'PID_POINT':
            reward += abs(observation[1]) ** 0.5
        # reward -= abs(observation[0] - 0.50)
        # reward += abs(observation[1] * 100)
        return reward


def action_space(output):
    if output > 0.1:
        action = 2
    elif output < -0.1:
        action = 0
    else:
        action = 1
    return action


PID_CART = (0, 0, 0)
# PID_CART = (-1.2978, -0.0252, -0.8364)
# PID_CART = (-1.8086, -0.0327, -0.7587)
PID_POINT = (0, 0, 0)

cart_agent = controllers.LearningPIDController('PID_CART', PID_CART)
point_agent = controllers.LearningPIDController('PID_POINT', PID_POINT)
learning_agent = controllers.MultiLearningController()
learning_agent.add_controller(cart_agent)
learning_agent.add_controller(point_agent)


def main():
    env = gym.make('MountainCar-v0')
    environment = controllers.Environment(env, learning_agent, MountainCar(env))

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
