import argparse

from osim.env import ProstheticsEnv
from osim.http.client import Client

from env.wrappers import Client_To_Env, Env_With_Dict_Observation, Env_With_JSONable_Actions
from env.dict_to_list import Dict_To_List
from utils.config import remote_base, crowdai_token
from agents.keras_ddpg_agent import KerasDDPGAgent as SpecifiedAgent


def submit(agent):
    # Create client environment

    client_env = Client_To_Env(remote_base, crowdai_token)
    client_env = Dict_To_List(client_env)
    client_env = Env_With_JSONable_Actions(client_env)
    
    agent.submit(client_env)

def main(saved_model=None):
    
    parser = argparse.ArgumentParser(description='Submit agent')
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

    submit(agent)
    

if __name__ == '__main__':
    main()