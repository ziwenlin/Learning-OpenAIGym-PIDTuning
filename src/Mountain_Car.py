import gym
import controllers
import settings


class MountainCar(controllers.EnvironmentController):
    position = 0.6

    def get_action(self, observation: gym.core.ObsType) -> gym.core.ActType:
        output = 0
        n_point = node_point.get_output(observation, 0.0)
        n_cart = node_cart.get_output(observation, 0.0)
        position = pid_point.get_output((n_point,), self.position)
        output += pid_cart.get_output((n_cart,), position)
        # if observation[1] < 0:
        #     output = cart_agent.get_output(observation[0], 0.6)
        # elif observation[1] > 0:
        #     output = cart_agent.get_output(observation[0], -1.2)
        action = action_space(output)
        return action

    def get_reward(self, observation: gym.core.ObsType) -> float:
        reward = 0
        position, speed = observation
        if learner.name in ('PID_CART'):
            reward += abs(speed) ** 0.5
            reward += abs(position - 0.5) ** 0.5
        elif learner.name in ('PID_POINT', 'NODE_POINT', 'NODE_CART'):
            reward += abs(speed) ** 0.5
            reward -= abs(position - 0.5) ** 0.5
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

pid_cart = controllers.ImprovingPIDController('PID_CART', PID_CART)
pid_point = controllers.ImprovingPIDController('PID_POINT', PID_POINT)
node_cart = controllers.ImprovingNodeController('NODE_CART', NODE_CART)
node_point = controllers.ImprovingNodeController('NODE_POINT', NODE_POINT)
learner = controllers.RotatingImprovingController()
learner.add_controller(pid_cart)
learner.add_controller(pid_point)
learner.add_controller(node_cart)
learner.add_controller(node_point)


def main():
    env = gym.make('MountainCar-v0')
    environment = controllers.EnvironmentRunner(env, learner, MountainCar(env))

    environment.start()
    while environment.running:
        try:
            environment.step_episode()
            environment.step_end()
        except KeyboardInterrupt:
            break
    environment.stop()
    for i in range(len(learner.controllers)):
        learner.select_controller(i)
        print(learner.selected.name, '=', learner.get_string())


if __name__ == '__main__':
    main()
