import argparse
import gymnasium as gym
import numpy as np

import cyberbattle._env.cyberbattle_env as cyberbattle_env

from gymnasium import spaces
from stable_baselines3 import PPO


# -----------------------------
# Observation Wrapper
# -----------------------------

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

        vector = np.zeros(100, dtype=np.float32)

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



# -----------------------------
# Action Wrapper
# -----------------------------

class CyberBattleActionWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.valid_actions = []

        # PPO-compatible action space
        self.action_space = spaces.Discrete(100)


    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        self.update_actions()

        return obs, info


    def update_actions(self):

        self.valid_actions = []

        seen_actions = set()

        for _ in range(100):

            action = self.env.sample_valid_action()

            # Convert action to comparable format
            action_key = str(action)

            if action_key not in seen_actions:

                seen_actions.add(action_key)

                self.valid_actions.append(action)


        # Safety check
        if len(self.valid_actions) == 0:

            self.valid_actions.append(
                self.env.sample_valid_action()
            )


    def action(self, action):

        # PPO action index -> CyberBattle action

        if action < len(self.valid_actions):

            selected_action = self.valid_actions[action]

        else:

            selected_action = self.valid_actions[0]


        return selected_action


    def step(self, action):

        cyber_action = self.action(action)

        result = self.env.step(cyber_action)

        self.update_actions()

        return result





# -----------------------------
# Main
# -----------------------------

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


print("Starting PPO training")
print("Chain size:", args.chain_size)



env = gym.make(
    "CyberBattleChain-v0",
    size=args.chain_size,
    attacker_goal=cyberbattle_env.AttackerGoal(
        own_atleast_percent=1.0,
        reward=2180
    )
)


print("Environment created")


env = CyberBattleObservationWrapper(env)

env = CyberBattleActionWrapper(env)


print("Observation:")
print(env.observation_space)

print("Action:")
print(env.action_space)



model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    gamma=0.99,
    verbose=1
)


model.learn(
    total_timesteps=args.training_timesteps
)


model.save(
    "ppo_cyberbattle_chain10"
)


print("PPO training completed")