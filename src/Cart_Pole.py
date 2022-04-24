import gym
import numpy.random

import settings
import controllers


class CartPole(controllers.EnvironmentController):
    position = 0.0
    episode = 0

    def reset(self):
        self.episode += 1
        progress = self.episode / settings.EPISODE_CAP
        self.position = numpy.random.uniform(-2, 2) * progress

    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        output = 0
        position = point_agent.get_output(observation[0] + observation[1] + observation[3], self.position)
        output += pole_agent.get_output(observation[2] + observation[3] + observation[1], 0.0)
        output += cart_agent.get_output(observation[0] + observation[1] + observation[3], position)
        return action_space(output)

    def get_reward(self, observation: gym.core.ObsType) -> float:
        reward = 0
        if learning_agent.name == 'PID_POLE':
            reward -= abs(observation[2] * 0.80) ** 0.5
        elif learning_agent.name == 'PID_CART':
            reward -= abs(observation[0] - self.position) ** 0.5
        elif learning_agent.name == 'PID_POINT':
            reward -= abs(observation[0] - self.position) ** 0.5
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

pole_agent = controllers.LearningPIDController('PID_POLE', PID_POLE)
cart_agent = controllers.LearningPIDController('PID_CART', PID_CART)
point_agent = controllers.LearningPIDController('PID_POINT', PID_POINT)
learning_agent = controllers.MultiLearningController()
learning_agent.add_controller(cart_agent)
learning_agent.add_controller(pole_agent)
learning_agent.add_controller(point_agent)
learning_agent.is_rotating = True


def main():
    env = gym.make('CartPole-v1')
    environment = controllers.Environment(env, learning_agent, CartPole(env))
    # environment.controller.episode = environment.episode = Settings.EPISODE_CAP // 10 * 7

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
