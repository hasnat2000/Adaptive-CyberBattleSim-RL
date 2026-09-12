import argparse
import gymnasium as gym
import numpy as np

import cyberbattle._env.cyberbattle_env as cyberbattle_env

from gymnasium import spaces
from stable_baselines3 import PPO



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


            elif isinstance(value,
                           (int, float,
                            np.integer,
                            np.floating)):

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

        seen = set()


        for _ in range(100):

            action = self.env.unwrapped.sample_valid_action()

            key = str(action)


            if key not in seen:

                seen.add(key)

                self.valid_actions.append(action)



        if len(self.valid_actions) == 0:

            self.valid_actions.append(
                self.env.unwrapped.sample_valid_action()
            )



    def action(self, action):

        action = int(action)


        if action < len(self.valid_actions):

            return self.valid_actions[action]


        return self.valid_actions[0]



    def step(self, action):

        cyber_action = self.action(action)

        result = self.env.step(cyber_action)

        self.update_actions()

        return result




# =====================================================
# Evaluation
# =====================================================


parser = argparse.ArgumentParser()


parser.add_argument(
    "--chain_size",
    default=10,
    type=int
)


parser.add_argument(
    "--episodes",
    default=10,
    type=int
)


parser.add_argument(
    "--max_steps",
    default=3000,
    type=int
)


parser.add_argument(
    "--model",
    default="ppo_cyberbattle_chain10"
)



args = parser.parse_args()



print("==============================")
print(" PPO Evaluation")
print("==============================")

print("Chain size:", args.chain_size)

print("Episodes:", args.episodes)



# =====================================================
# Create environment
# =====================================================

env = gym.make(

    "CyberBattleChain-v0",

    size=args.chain_size,

    attacker_goal=cyberbattle_env.AttackerGoal(

        own_atleast_percent=1.0,

        reward=2180

    )

)



# Apply same wrappers as training

env = CyberBattleObservationWrapper(env)

env = CyberBattleActionWrapper(env)



# =====================================================
# Load trained PPO
# =====================================================

model = PPO.load(

    args.model,

    env=env

)



print("Model loaded successfully")

print("------------------------------")



episode_rewards = []

episode_lengths = []

success_count = 0



# =====================================================
# Run Evaluation Episodes
# =====================================================

for episode in range(args.episodes):


    obs, info = env.reset()


    done = False

    truncated = False


    total_reward = 0

    steps = 0



    while not done and not truncated and steps < args.max_steps:


        action, _ = model.predict(

            obs,

            deterministic=True

        )


        obs, reward, done, truncated, info = env.step(action)



        total_reward += reward

        steps += 1




    episode_rewards.append(total_reward)

    episode_lengths.append(steps)



    # success condition

    if total_reward > 0:

        success_count += 1



    print(
        f"Episode {episode+1}: "
        f"Reward={total_reward:.2f}, "
        f"Steps={steps}"
    )




# =====================================================
# Final Results
# =====================================================

print("\n==============================")
print(" PPO Evaluation Results")
print("==============================")


print(
    "Average Reward:",
    round(np.mean(episode_rewards),2)
)


print(
    "Maximum Reward:",
    round(np.max(episode_rewards),2)
)


print(
    "Minimum Reward:",
    round(np.min(episode_rewards),2)
)


print(
    "Average Episode Length:",
    round(np.mean(episode_lengths),2)
)


print(
    "Success Rate:",
    round((success_count / args.episodes)*100,2),
    "%"
)


print("==============================")