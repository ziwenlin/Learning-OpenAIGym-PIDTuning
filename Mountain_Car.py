import gym
import Settings
import Controllers


PRESET_PID_CART = (0, 0, 0)
PRESET_PID_CART = (-1.2978, -0.0252, -0.8364)
PRESET_PID_CART = (-1.8086, -0.0327, -0.7587)


env = gym.make('MountainCar-v0')
logger = Controllers.Logger()

cart_agent = Controllers.PID_Learning_Controller(PRESET_PID_CART)
agent = cart_agent
agent.name = 'CART'
# agent = pole_agent
# agent.name = 'POLE'
sum_reward = 0.0


def action_space(output):
    if output > 1:
        action = 2
    elif output < -1:
        action = 0
    else:
        action = 1
    return action


for i_episode in range(Settings.EPISODES):
    observation = env.reset()
    for time_steps in range(Settings.TIME_STEPS):
        if (i_episode + 1) % Settings.EPISODE_SHOW == 0:
            env.render()
        output = 0
        # output += pole_agent.get_output(observation[2], 0.0)
        if observation[1] < 0:
            output += cart_agent.get_output(observation[0], 0.6)
        elif observation[1] > 0:
            output += cart_agent.get_output(observation[0], -1.2)
        action = action_space(output)
        observation, reward, done, info = env.step(action)
        reward -= abs(observation[0] - 0.50)
        reward += abs(observation[1] * 100)
        sum_reward += reward
        if done:
            if (i_episode + 1) % Settings.EPISODE_PRINT == 0 or (i_episode + 1) % Settings.EPISODE_SHOW == 0:
                print("Episode {} finished after {} timesteps".format(i_episode + 1, time_steps + 1))
            break
    else:
        if (i_episode + 1) % Settings.EPISODE_PRINT == 0 or (i_episode + 1) % Settings.EPISODE_SHOW == 0:
            print("Episode {} finished after {} timesteps".format(i_episode + 1, Settings.TIME_STEPS))

    agent.reward(sum_reward)
    logger.monitor(sum_reward)
    sum_reward = 0.0
    if (i_episode + 1) % Settings.EPISODE_LEARN == 0:
        logger.process(i_episode + 1, Settings.EPSILON, Settings.MULTIPLIER_EPSILON)
        agent.reflect()
        agent.explore()

    if Settings.MULTIPLIER_EPSILON > 1.0:
        Settings.MULTIPLIER_EPSILON *= Settings.EPSILON_DECAY
    if Settings.EPSILON > Settings.EPSILON_CAP:
        Settings.EPSILON *= Settings.EPSILON_DECAY

    if (i_episode + 1) % Settings.EPISODE_SHOW == 0:
        log = logger.get_log()
        log += f'PRESET_PID_{agent.name} = {agent.get_string()}' + '\n'
        print(log)

env.close()
