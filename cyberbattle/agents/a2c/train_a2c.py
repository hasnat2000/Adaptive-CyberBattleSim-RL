import argparse
import gymnasium as gym
import numpy as np

# Register CyberBattle environments
import cyberbattle._env.cyberbattle_env as cyberbattle_env

from gymnasium import spaces
from stable_baselines3 import A2C


# ============================================================
# Observation Wrapper
# Converts CyberBattle structured observations
# into numerical vector for Stable-Baselines3
# ============================================================

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


            elif isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating
                )
            ):

                vector[index] = float(value)
                index += 1


        extract(observation)

        return vector



# ============================================================
# Action Wrapper
# Converts CyberBattle DiscriminatedUnion action space
# into Stable-Baselines3 Discrete action space
# ============================================================

class CyberBattleActionWrapper(gym.ActionWrapper):

    def __init__(self, env):

        super().__init__(env)

        self.valid_actions = []

        # Stable-Baselines3 compatible action space
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

            action_key = str(action)


            if action_key not in seen_actions:

                seen_actions.add(action_key)

                self.valid_actions.append(action)



        # Safety fallback

        if len(self.valid_actions) == 0:

            self.valid_actions.append(
                self.env.unwrapped.sample_valid_action()
            )



    def action(self, action):

        # Convert A2C integer action
        # into CyberBattle structured action

        if action < len(self.valid_actions):

            selected_action = self.valid_actions[action]

        else:

            selected_action = self.valid_actions[0]


        return selected_action



    def step(self, action):

        cyber_action = self.action(action)

        result = self.env.step(cyber_action)

        # update available actions after state change

        self.update_actions()

        return result



# ============================================================
# Main Program
# ============================================================


parser = argparse.ArgumentParser(
    description="Train A2C agent on Microsoft CyberBattleSim"
)


parser.add_argument(
    "--chain_size",
    default=10,
    type=int,
    help="CyberBattleChain topology size"
)


parser.add_argument(
    "--training_timesteps",
    default=50000,
    type=int,
    help="Total A2C training timesteps"
)



args = parser.parse_args()



print("================================")
print("Starting A2C Training")
print("Chain size:", args.chain_size)
print("Training timesteps:", args.training_timesteps)
print("================================")



# ------------------------------------------------------------
# Create CyberBattle Environment
# ------------------------------------------------------------


env = gym.make(
    "CyberBattleChain-v0",
    size=args.chain_size,
    attacker_goal=cyberbattle_env.AttackerGoal(
        own_atleast_percent=1.0,
        reward=2180
    )
)


print("Environment created successfully")



# ------------------------------------------------------------
# Apply wrappers
# ------------------------------------------------------------

env = CyberBattleObservationWrapper(env)

env = CyberBattleActionWrapper(env)



print("\nObservation Space:")
print(env.observation_space)


print("\nAction Space:")
print(env.action_space)



# ------------------------------------------------------------
# Create A2C Agent
# ------------------------------------------------------------


model = A2C(

    "MlpPolicy",

    env,

    learning_rate=0.0007,

    gamma=0.99,

    verbose=1

)



# ------------------------------------------------------------
# Training
# ------------------------------------------------------------


model.learn(
    total_timesteps=args.training_timesteps
)



# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------


model_name = (
    f"a2c_cyberbattle_chain{args.chain_size}"
)


model.save(model_name)



print("\n================================")
print("A2C Training Completed")
print("Saved Model:", model_name)
print("================================")