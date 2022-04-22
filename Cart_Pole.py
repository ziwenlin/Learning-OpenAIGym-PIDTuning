import Controllers
import Settings
import gym


class CartPole(Controllers.Decider):
    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        action = pole_agent.get_output(observation[2], 0.0)
        return action_space(action)

    def get_reward(self, observation: gym.core.ObsType) -> float:
        return 0


PRESET_PID_CART = (0, 0, 0)
# PRESET_PID_CART = (-0.924861051852458, 0.03002569629632385, -0.2994526736003437)
# PRESET_PID_CART = (-0.924861051852458, 0.03002569629632385, -0.3848792315820444)

PRESET_PID_POLE = (0, 0, 0)
PRESET_PID_POLE = (-1.8394291134013734, 0.20138629211940734, -5.200392703676353)
# PRESET_PID_POLE = (-4.512285277021222, 3.7468386916186502, 7.743010122805896)


pole_agent = Controllers.PID_Learning_Controller(PRESET_PID_POLE)
# cart_agent = Controllers.PID_Learning_Controller(PRESET_PID_CART)
# agent = cart_agent
# agent.name = 'PRESET_PID_CART'
agent = pole_agent
agent.name = 'PRESET_PID_POLE'

env = gym.make('CartPole-v1')
environment = Controllers.Environment(env, agent, CartPole(env))


def action_space(output):
    if output > 0:
        action = 1
    else:
        action = 0
    return action


environment.start()
while environment.running:
    environment.step_episode()
    environment.step_end()
environment.stop()
