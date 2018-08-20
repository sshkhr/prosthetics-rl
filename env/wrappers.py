# Source : https://github.com/seungjaeryanlee/osim-rl-helper

import gym, numpy as np
from osim.http.client import Client

class EnvironmentWrapper:
    def __init__(self, env):
        """
        A base template for all environment wrappers.
        """
        self.env = env

        # Attributes
        self.observation_space = env.observation_space if hasattr(env, 'observation_space') else None
        self.action_space = env.action_space if hasattr(env, 'action_space') else None
        self.time_limit = env.time_limit if hasattr(env, 'time_limit') else None
        self.submit = env.submit if hasattr(env, 'submit') else None

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)

class Client_To_Env:
    def __init__(self, remote_base, crowdai_token):
        """
        Wrapper that reformats client environment to a local environment format,
        complete with observation_space, action_space, reset, step, submit, and
        time_limit.
        """
        
        self.client = Client(remote_base)
        self.crowdai_token = crowdai_token
        self.reset_ = self.client.env_reset
        self.step  = self.client.env_step
        self.submit = self.client.submit
        self.time_limit = 300
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(19, ),dtype=np.float32)

        self.first_reset = True

    def reset(self):
        if self.first_reset:
            self.first_reset = False
            obs = self.client.env_create(self.crowdai_token, env_id='ProstheticsEnv')
            return obs
        else:
            obs = self.reset_()
            return obs


class Env_With_JSONable_Actions(EnvironmentWrapper):
    def __init__(self, env):
        """
        Environment Wrapper that converts NumPy ndarray type actions to list.
        This wrapper is needed for communicating with the client for submission.
        """
        super().__init__(env)
        self.env = env

    def step(self, action):
        if type(action) == np.ndarray:
            return self.env.step(action.tolist())
        else:
            return self.env.step(action)

class Env_With_Dict_Observation(EnvironmentWrapper):
    def __init__(self, env):
        """
        Environment wrapper that wraps local environment to use dict-type
        observation by setting project=False. This can be deprecated once
        the default observation is dict-type rather than list-type.
        """
        super().__init__(env)
        self.env = env
        self.time_limit = 300
    
    def reset(self):
        return self.env.reset(project=False)

    def step(self, action):
        return self.env.step(action, project=False)

class Env_With_Discreet_Actions(EnvironmentWrapper):
    def __init__(self, env):
        """
        Environment wrapper that quantises actions to discreet values.
        """
        super().__init__(env)
        self.env = env
        self.time_limit = 300
    
    def reset(self):
        return self.env.reset(project=False)

    def discretise_action(action):
        bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        discrete_action = np.digitize(action, bins)
        return action

    def step(self, action):
        action = discretise_action(action)
        return self.env.step(action, project=False)


class Env_With_Reward_Shaping(EnvironmentWrapper):
    def __init__(self, env):
        """
        Environment wrapper that wraps local environment to use dict-type
        observation by setting project=False. This can be deprecated once
        the default observation is dict-type rather than list-type.
        """
        super().__init__(env)
        self.env = env
        self.time_limit = 300
    
    def reset(self):
        return self.env.reset(project=False)

    def shape_reward(observation, reward, done):
        shaped_reward = reward
        return shaped_reward

    def step(self, action):
        observation, reward, done, info = env.step(action, project=False)
        shaped_reward = shape_reward(observation, reward, done)
        return observation, shaped_reward, done, info
