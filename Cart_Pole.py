import Controllers
import Settings
import gym


class CartPole(Controllers.Environment_Controller):
    position = 0.0

    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        output = 0
        output += pole_agent.get_output(observation[2] + observation[3] + observation[1], 0.0)
        output += cart_agent.get_output(observation[0] + observation[1] + observation[3], self.position)
        return action_space(output)

    def get_reward(self, observation: gym.core.ObsType) -> float:
        reward = 0
        if learning_agent.name == 'PID_POLE':
            reward -= (observation[2] * 0.80) ** 2
        elif learning_agent.name == 'PID_CART':
            reward -= (observation[0] - self.position) * 10
        return reward


def action_space(output):
    if output > 0:
        action = 1
    else:
        action = 0
    return action


Settings.EPISODE_CAP *= 10
Settings.TIME_STEPS = 500

PID_CART = (0, 0, 0)
PID_CART = (0.2313, -0.0012, 1.2712)
PID_CART = (0.7395, -0.3309, 1.2712)
PID_CART = (1.0369, -0.5287, 1.132)
PID_CART = (0.3935, -0.3809, 2.9108)

# PID_POLE = (-1.8632, 0.0049, -5.6141)
# PID_POLE = (-0.9303, 0.0049, -5.7019)
# PID_POLE = (0, 0, 0)
# PID_POLE = (-0.6757, 0.0031, -3.5676)
# PID_POLE = (-0.1363, 0.0013, -0.638)
PID_POLE = (-1.0184, -0.0008, -5.5321)
PID_POLE = (-1.0139, 0.0668, -6.5261)
PID_POLE = (-1.2123, 0.0668, -6.7867)
PID_POLE = (-2.1976, 1.9243, -7.0012)

pole_agent = Controllers.Learning_PID_Controller(PID_POLE, 'PID_POLE')
cart_agent = Controllers.Learning_PID_Controller(PID_CART, 'PID_CART')
learning_agent = Controllers.Multi_Learning_Controller()
learning_agent.add_controller(cart_agent)
learning_agent.add_controller(pole_agent)
learning_agent.is_rotating = True

env = gym.make('CartPole-v1')
environment = Controllers.Environment(env, learning_agent, CartPole(env))

environment.start()
while environment.running:
    try:
        environment.step_episode()
        environment.step_end()
    except KeyboardInterrupt:
        break
environment.stop()
