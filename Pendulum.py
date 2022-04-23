import gym
import Settings
import Controllers


class Pendulum(Controllers.Environment_Controller):
    def get_reward(self, observation: gym.core.ObsType) -> float:
        # reward -= abs(observation[0] - 0.50)
        # reward += abs(observation[1] * 100)
        return 0

    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        output = 0
        output += cos_agent.get_output(observation[1], 1.0)
        output += sin_agent.get_output(observation[0], 0.0)
        action = action_space(output)
        return action


def action_space(output):
    if output > 2:
        action = 2
    elif output < -2:
        action = 2
    else:
        action = output
    return [action]

Settings.EPISODE_MULTIPLIER = 1
Settings.recalculate()
Settings.EPISODE_PRINT = Settings.EPISODE_CAP

PID_COS = (0, 0, 0)
PID_COS = (-1.2978, -0.0252, -0.8364)
PID_COS = (-1.8086, -0.0327, -0.7587)
PID_COS = (-1.8086, 0.4038, -0.7587)

PID_SIN = (0, 0, 0)

env = gym.make('Pendulum-v1')
logger = Controllers.Environment_Monitor()

cos_agent = Controllers.Learning_PID_Controller(preset=PID_COS)
# agent = cos_agent
# agent.name = 'PID_COS'
sin_agent = Controllers.Learning_PID_Controller(preset=PID_SIN)
agent = sin_agent
agent.name = 'PID SIN'

environment = Controllers.Environment(env, agent, Pendulum(env))

environment.start()
while environment.running:
    environment.step_episode()
    environment.step_end()
environment.stop()
