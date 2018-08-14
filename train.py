import argparse

from osim.env import ProstheticsEnv
from osim.http.client import Client

from env.wrappers import Client_To_Env, Env_With_Dict_Observation, Env_With_JSONable_Actions
from env.dict_to_list import Dict_To_List
from utils.config import remote_base, crowdai_token
from agents.keras_ddpg_agent import KerasDDPGAgent as SpecifiedAgent

from test import test
from submit import submit

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def train(agent, env, nb_steps = 100000):
    agent.train(env, nb_steps = nb_steps)
    

def main(saved_model=None):
    parser = argparse.ArgumentParser(description='Test or submit agent.')
    parser.add_argument('-t', '--test', action='store', default=True, help='test agent locally')
    parser.add_argument('-s', '--submit', action='store_true', default=True, help='submit agent to crowdAI server')
    parser.add_argument('-v', '--visualize', action='store_true', default=False, help='render the environment locally')
    args = parser.parse_args()

    # Create environment

    env = ProstheticsEnv(visualize=args.visualize)
    env = Env_With_Dict_Observation(env)
    env = Dict_To_List(env)
    env = Env_With_JSONable_Actions(env)

    # Specify Agent

    agent = SpecifiedAgent(env.observation_space, env.action_space)
    
    if saved_model:
    	agent.load(saved_model)

    train(agent, env)

    if args.test:
    	test(agent,env)

    if args.submit:
    	submit(agent)

if __name__ == '__main__':
	main()