# Source : https://github.com/seungjaeryanlee/osim-rl-helper

import gym
import numpy as np

from .wrappers import EnvironmentWrapper

class Dict_To_List(EnvironmentWrapper):
    def __init__(self, env):
        """
        A wrapper that formats dict-type observation to list-type observation.
        Appends all meaningful unique numbers in the dict-type observation to a
        list. The resulting list has length 347.
        """
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-float('Inf'),
                                                high=float('Inf'),
                                                shape=(347, ),
                                                dtype=np.float32)

    def reset(self):
        state_desc = self.env.reset()
        return self._dict_to_list(state_desc)

    def step(self, action):
        state_desc, reward, done, info = self.env.step(action)
        return [self._dict_to_list(state_desc), reward, done, info]

    def _dict_to_list(self, state_desc):
        """
        Return observation list of length 347 given a dict-type observation.
        For more details about the observation, visit this page:
        http://osim-rl.stanford.edu/docs/nips2018/observation/
        """
        res = []

        # Body Observations
        for info_type in ['body_pos', 'body_pos_rot',
                          'body_vel', 'body_vel_rot',
                          'body_acc', 'body_acc_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += state_desc[info_type][body_part]
        
        # Joint Observations
        # Neglecting `back_0`, `mtp_l`, `subtalar_l` since they do not move
        for info_type in ['joint_pos', 'joint_vel', 'joint_acc']:
            for joint in ['ankle_l', 'ankle_r', 'back', 'ground_pelvis',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += state_desc[info_type][joint]

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r', 
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r', 
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(state_desc['muscles'][muscle]['activation'])
            res.append(state_desc['muscles'][muscle]['fiber_force'])
            res.append(state_desc['muscles'][muscle]['fiber_length'])
            res.append(state_desc['muscles'][muscle]['fiber_velocity'])

        # Force Observations
        # Neglecting forces corresponding to muscles as they are redundant with
        # `fiber_forces` in muscles dictionaries
        for force in ['AnkleLimit_l', 'AnkleLimit_r',
                      'HipAddLimit_l', 'HipAddLimit_r',
                      'HipLimit_l', 'HipLimit_r', 'KneeLimit_l', 'KneeLimit_r']:
            res += state_desc['forces'][force]

        # Center of Mass Observations
        res += state_desc['misc']['mass_center_pos']
        res += state_desc['misc']['mass_center_vel']
        res += state_desc['misc']['mass_center_acc']

        return res