#!/usr/bin/python3.10

import torch
import gymnasium as gym
import logging
import sys
import argparse

import cyberbattle._env.cyberbattle_env as cyberbattle_env

from typing import cast

from cyberbattle._env.cyberbattle_env import CyberBattleEnv

from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.learner as learner



# =====================================================
# Adaptive Reward Wrapper
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



        # -------------------------
        # Objective 1:
        # Attack success
        # -------------------------

        success_reward = reward



        # -------------------------
        # Objective 2:
        # Target importance
        # -------------------------

        critical_reward = 0


        if reward > 0:

            critical_reward = reward * 0.5



        # -------------------------
        # Objective 3:
        # Efficiency
        # -------------------------

        efficiency_reward = 0


        if reward > 0:

            efficiency_reward = 10 / self.steps



        # -------------------------
        # Objective 4:
        # Ineffective action penalty
        # -------------------------

        penalty = 0


        if reward <= 0:

            penalty = 1




        # Adaptive reward equation

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
# Arguments
# =====================================================


parser = argparse.ArgumentParser(
    description="DQN with Adaptive Reward"
)


parser.add_argument(
    "--training_episode_count",
    default=10,
    type=int
)


parser.add_argument(
    "--iteration_count",
    default=3000,
    type=int
)


parser.add_argument(
    "--chain_size",
    default=10,
    type=int
)


parser.add_argument(
    "--reward_goal",
    default=2180,
    type=int
)


parser.add_argument(
    "--ownership_goal",
    default=1.0,
    type=float
)



args = parser.parse_args()



logging.basicConfig(
    stream=sys.stdout,
    level=logging.ERROR
)



print("Starting Adaptive DQN Training")

print("Chain size:", args.chain_size)



# =====================================================
# Create Environment
# =====================================================


cyberbattlechain = cast(

    CyberBattleEnv,

    gym.make(

        "CyberBattleChain-v0",

        size=args.chain_size,

        attacker_goal=cyberbattle_env.AttackerGoal(

            own_atleast_percent=args.ownership_goal,

            reward=args.reward_goal

        )

    )

)



print("Environment created")



# Apply adaptive reward

cyberbattlechain = AdaptiveRewardWrapper(
    cyberbattlechain
)



# Environment properties

ep = w.EnvironmentBounds.of_identifiers(

    maximum_total_credentials=22,

    maximum_node_count=22,

    identifiers=cyberbattlechain.identifiers

)



# =====================================================
# Adaptive DQN Training
# =====================================================


dqn_learning_run = learner.epsilon_greedy_search(

    cyberbattle_gym_env=cyberbattlechain,

    environment_properties=ep,


    learner=dqla.DeepQLearnerPolicy(

        ep=ep,

        gamma=0.015,

        replay_memory_size=10000,

        target_update=10,

        batch_size=512,

        learning_rate=0.01

    ),


    episode_count=args.training_episode_count,


    iteration_count=args.iteration_count,


    epsilon=0.90,


    render=True,


    epsilon_exponential_decay=5000,


    epsilon_minimum=0.10,


    verbosity=Verbosity.Quiet,


    title="Adaptive DQN"

)



print("Adaptive DQN training completed")