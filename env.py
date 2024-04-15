import gymnasium as gym


class MyEnv(gym.Env):
    def __init__(self):
        super(MyEnv, self).__init__()

    def reset(self, **kwargs):
        return 0

    def step(self, action):
        return 0, 0, 0, {}
