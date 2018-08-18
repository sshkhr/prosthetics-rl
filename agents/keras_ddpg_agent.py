from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam, RMSprop

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from .base_agent import Agent

class KerasAgent(Agent):
    def __init__(self, observation_space, action_space, filename):
        """
        Template class for agents using Keras RL library.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.filename = filename

    def train(self, env, nb_steps):
        """
        Train agent for nb_steps.
        """

        self.load(self.filename)

        print('[train] Training \'{}\''.format(type(self).__name__))
        self.agent.fit(env, nb_steps=nb_steps, action_repetition=5, visualize=False, verbose=1, nb_max_episode_steps=env.time_limit, log_interval=1000)
        print('[train] Finished training')

        self.save(self.filename)      

    def test(self, env):
        """
        Run agent locally.
        """
        self.load(self.filename)

        print('[test] Running \'{}\''.format(type(self).__name__))
        self.agent.test(env, nb_episodes=1, action_repetition=5, visualize=False, nb_max_episode_steps=env.time_limit)
        print('[test] Finished test')

    def submit(self, env):
        """
        Submit agent to CrowdAI server.
        """
        self.load(self.filename)

        print('[submit] Running on server environment \'{}\''.format(type(self).__name__))
        try:
            self.agent.test(env, nb_episodes=3, visualize=False, nb_max_episode_steps=300)
        except TypeError:
            # When observation is None - no more steps left
            pass

        print('[submit] Finished Running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        env.submit()
        print('[submit] Submitted results successfully!')

    def save(self,filename):
        """
        Save model file
        """
        print('[train] Saved weights to \'{}\''.format(self.filename))
        self.agent.save_weights(self.filename, overwrite=True)
        print('[train] Successfully saved weights to \'{}\''.format(self.filename))

    def load(self,filename):
        """
        Load agent
        """
        try:
            print('[train] Loading weights from {}'.format(self.filename))
            self.agent.load_weights(self.filename)
            print('[train] Successfully loaded weights from {}'.format(self.filename))
        except OSError:
            print('[train] Pretrained model {} not found. Starting from scratch.'.format(self.filename))


class KerasDDPGAgent(KerasAgent):
    """
    An DDPG agent using Keras library with Keras RL.
    For more details about Deep Deterministic Policy Gradient algorithm, check
    "Continuous control with deep reinforcement learning" by Lillicrap.
    https://arxiv.org/abs/1509.02971
    """

    def __init__(self, observation_space, action_space, filename='KerasDDPGAgent.h5f'):
        nb_actions = action_space.shape[0]

        # Actor network
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + observation_space.shape))
        actor.add(Dense(256))
        actor.add(Activation('relu'))
        actor.add(Dense(128))
        actor.add(Activation('relu'))
        actor.add(Dense(64))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('sigmoid'))
        print(actor.summary())

        # Critic network
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)

        x = concatenate([action_input, flattened_observation])
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())

        # Setup Keras RL's DDPGAgent
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=nb_actions)
        
        self.agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                               memory=memory, batch_size=128, nb_steps_warmup_critic=128, nb_steps_warmup_actor=128, 
                               random_process=random_process, gamma=.75, target_model_update=1e-2, delta_clip=2.)
        
        self.agent.compile(Adam(lr=.01, clipnorm=2.), metrics=['mae'])

        self.filename = filename