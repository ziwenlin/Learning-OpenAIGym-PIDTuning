import gym
import controllers
import settings


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
        if learning_agent.name in ('PID_CART', 'NODE_CART'):
            reward += abs(observation[1]) ** 0.5
        elif learning_agent.name in ('PID_POINT', 'NODE_POINT'):
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


settings.EPISODE_CAP *= 10
settings.TIME_STEPS = 250

PID_CART = (0, 0, 0)
# PID_CART = (-1.2978, -0.0252, -0.8364)
# PID_CART = (-1.8086, -0.0327, -0.7587)
PID_POINT = (0, 0, 0)

NODE_CART = (0, 0)
NODE_POINT = (0, 0)

cart_agent = controllers.LearningPIDController('PID_CART', PID_CART)
point_agent = controllers.LearningPIDController('PID_POINT', PID_POINT)
cart_node = controllers.LearningNodeController('NODE_CART', NODE_CART)
point_node = controllers.LearningNodeController('NODE_POINT', NODE_POINT)
learning_agent = controllers.LearningMultiController()
learning_agent.add_controller(cart_agent)
learning_agent.add_controller(point_agent)
learning_agent.add_controller(cart_node)
learning_agent.add_controller(point_node)


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
