from tensorforce.agents import PPOAgent

from .base_agent import Agent

class TensorforceAgent(Agent):
    def __init__(self, observation_space, action_space, directory):
        """
        Template class for agents using Keras RL library.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.directory = directory
        self.agent = None

    def train(self, env, nb_steps):

        self.load()
        
        print('[train] Training \'{}\''.format(type(self).__name__))
        step_count = 0
        episode_count = 1
        while step_count < nb_steps:
            episode_step_count = 0
            obs = env.reset()
            done = False
            total_rew = 0
            while not done:
                action = self.agent.act(obs)
                obs, rew, done, info = env.step(action)
                total_rew += rew
                self.agent.observe(reward=rew, terminal=done)
                episode_step_count += 1
            step_count += episode_step_count
            print('[train] Episode {:3} | Steps Taken: {:3} | Total Steps: Taken {:6}/{:6} | Total reward: {}'.format(
                episode_count, episode_step_count, step_count, nb_steps, total_rew))
            episode_count += 1
        print('[train] Finished training')

        self.save()

        

    def test(self, env):
        """
        Run agent locally.
        """
        self.load()

        print('[test] Running \'{}\''.format(type(self).__name__))
        obs = env.reset()
        done = False
        total_rew = 0
        while not done:
            action = self.agent.act(obs)
            obs, rew, done, info = env.step(action)
            total_rew += rew
            self.agent.observe(reward=rew, terminal=done)
        print('[test] Total reward: ' + str(total_rew))
        print('[test] Finished test.')

        self.save()

    def submit(self, env):
        """
        Submit agent to CrowdAI server.
        """
        self.load()

        print('[submit] Running \'{}\''.format(type(self).__name__))
        obs = env.reset()
        episode_count = 1
        step_count = 0
        total_rew = 0
        try:
            while True:
                action = self.act(obs)
                obs, rew, done, info = env.step(action)
                total_rew += rew
                step_count += 1
                if done:
                    print('[submit] Episode {} | Steps Taken: {:3} | Total reward: {}'.format(episode_count, step_count, total_rew))
                    obs = env.reset()
                    episode_count += 1
                    step_count = 0
                    total_rew = 0
        except TypeError:
            # When observation is None - no more steps left
            pass

        print('[submit] Finished running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        env.submit()
        print('[submit] Submitted results successfully!')

    def save(self):
        print('Saved weights to \'{}\''.format(self.directory))
        self.agent.save_model(directory=self.directory)
        print('Successfully saved weights to \'{}\''.format(self.directory))


    def load(self):
        """
        Load agent
        """
        try:
            print('Loading weights from {}'.format(self.directory))
            self.agent.restore_model(directory=self.directory)
            print('Successfully loaded weights from {}'.format(self.directory))
        except ValueError:
            print('Pretrained model {} not found. Starting from scratch.'.format(self.directory))

    def act(self, obs):
        return self.agent.act(obs) 




class TensorforcePPOAgent(TensorforceAgent):
    def __init__(self, observation_space, action_space,
                 directory='./TensorforcePPOAgent/'):
        # Create a Proximal Policy Optimization agent

        self.agent = PPOAgent(
            states=dict(type='float', shape=observation_space.shape),
            actions=dict(type='float', shape=action_space.shape, min_value=0, max_value=1),
            network=[
                dict(type='dense', size=256, activation='relu'),
                dict(type='dense', size=128, activation='relu'),
                dict(type='dense', size=64, activation='relu'),
                dict(type='dense', size=32, activation='relu'),
            ],
            batching_capacity=1000,
            step_optimizer=dict(type='adam', learning_rate=1e-2)
        )
        self.directory = directory