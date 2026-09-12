import argparse
import gymnasium as gym
import numpy as np

import cyberbattle._env.cyberbattle_env as cyberbattle_env

from gymnasium import spaces
from stable_baselines3 import PPO



# =====================================================
# Adaptive Multi Objective Reward Wrapper
# =====================================================

class AdaptiveRewardWrapper(gym.Wrapper):

    def __init__(self, env):

        super().__init__(env)

        self.steps = 0



    def reset(self, **kwargs):

        self.steps = 0

        return self.env.reset(**kwargs)



    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps += 1



        # -----------------------------
        # Objective 1:
        # Attack success
        # -----------------------------

        success_reward = reward



        # -----------------------------
        # Objective 2:
        # Target criticality
        # -----------------------------

        critical_reward = 0


        if reward > 0:

            critical_reward = reward * 0.5



        # -----------------------------
        # Objective 3:
        # Attack efficiency
        # -----------------------------

        efficiency_reward = 0


        if reward > 0:

            efficiency_reward = 10 / self.steps



        # -----------------------------
        # Objective 4:
        # Ineffective action penalty
        # -----------------------------

        penalty = 0


        if reward <= 0:

            penalty = 1



        # -----------------------------
        # Adaptive Reward
        #
        # R =
        # 0.4 Success
        # +0.3 Criticality
        # +0.2 Efficiency
        # -0.1 Penalty
        #
        # -----------------------------

        adaptive_reward = (

            0.4 * success_reward

            +

            0.3 * critical_reward

            +

            0.2 * efficiency_reward

            -

            0.1 * penalty

        )


        return (

            obs,

            adaptive_reward,

            terminated,

            truncated,

            info

        )





# =====================================================
# Observation Wrapper
# =====================================================

class CyberBattleObservationWrapper(gym.ObservationWrapper):


    def __init__(self, env):

        super().__init__(env)


        self.observation_space = spaces.Box(

            low=0,

            high=1,

            shape=(100,),

            dtype=np.float32

        )




    def observation(self, observation):


        vector = np.zeros(

            100,

            dtype=np.float32

        )


        index = 0



        def extract(value):

            nonlocal index


            if index >= 100:

                return



            if isinstance(value, dict):

                for v in value.values():

                    extract(v)



            elif isinstance(value, (list, tuple)):

                for v in value:

                    extract(v)



            elif isinstance(value, (int,float,np.integer,np.floating)):

                vector[index] = float(value)

                index += 1



        extract(observation)


        return vector






# =====================================================
# Action Wrapper
# =====================================================

class CyberBattleActionWrapper(gym.ActionWrapper):


    def __init__(self, env):

        super().__init__(env)


        self.valid_actions = []


        self.action_space = spaces.Discrete(100)



    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        self.update_actions()

        return obs, info




    def update_actions(self):

        self.valid_actions = []

        seen_actions = set()



        for _ in range(100):

            action = self.env.unwrapped.sample_valid_action()


            key = str(action)



            if key not in seen_actions:

                seen_actions.add(key)

                self.valid_actions.append(action)




        if len(self.valid_actions) == 0:

            self.valid_actions.append(

                self.env.unwrapped.sample_valid_action()

            )





    def action(self, action):


        if action < len(self.valid_actions):

            return self.valid_actions[action]


        else:

            return self.valid_actions[0]





    def step(self, action):


        cyber_action = self.action(action)


        result = self.env.step(cyber_action)


        self.update_actions()


        return result





# =====================================================
# Main Training
# =====================================================


parser = argparse.ArgumentParser()



parser.add_argument(

    "--chain_size",

    default=10,

    type=int

)



parser.add_argument(

    "--training_timesteps",

    default=50000,

    type=int

)



args = parser.parse_args()



print("Starting Adaptive PPO Training")

print("Chain size:", args.chain_size)





# Create CyberBattle Environment

env = gym.make(

    "CyberBattleChain-v0",

    size=args.chain_size,

    attacker_goal=cyberbattle_env.AttackerGoal(

        own_atleast_percent=1.0,

        reward=2180

    )

)



print("Environment created")





# Apply Adaptive Reward

env = AdaptiveRewardWrapper(env)



# Apply PPO compatible wrappers

env = CyberBattleObservationWrapper(env)


env = CyberBattleActionWrapper(env)



print("Observation:")

print(env.observation_space)



print("Action:")

print(env.action_space)





# =====================================================
# PPO Agent
# =====================================================


model = PPO(

    "MlpPolicy",

    env,

    learning_rate=0.0003,

    gamma=0.99,

    verbose=1

)




# Training

model.learn(

    total_timesteps=args.training_timesteps

)




# Save model

model.save(

    "ppo_adaptive_cyberbattle"

)




print("Adaptive PPO training completed")