import pickle

import pfrl
import torch
import numpy as np

import bendplanner.bend_utils as bu
from rf_planner.cal_seq.bend_env import BendEnv

goal_pseq = pickle.load(open('goal_pseq.pkl', 'rb'))
init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
init_rotseq = [np.eye(3), np.eye(3)]

env = BendEnv(goal_pseq=goal_pseq, pseq=init_pseq, rotseq=init_rotseq)

# obs = env.reset()
# print('initial observation:', obs)
#
# action = env.sample_action()
# obs, r, done, info = env.step(action)
# print('next observation:', obs)
# print('reward:', r)
# print('done:', done)
# print('info:', info)

# TODO: use point net to extract point features here, for now we just flatten the input to be 1D array
q_func = torch.nn.Sequential(
    torch.nn.Linear(200 * 200 * 200, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 1),
    pfrl.q_functions.DiscreteActionValueHead(),
)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.sample_action)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = -1

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
    gpu=gpu,
)

n_episodes = 10
max_episode_len = 20
for i in range(1, n_episodes + 1):
    env.reset()
    obs = env.get_observation()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        # TODO: here we use flattened observation, in the future, the agent should use pointnet to extract point feature
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 10 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')

with agent.eval_mode():
    for i in range(10):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            reset = t == 200
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)

# Save an agent to the 'agent' directory
agent.save('agent')

# Uncomment to load an agent from the 'agent' directory
# agent.load('agent')

# Set up the logger to print info messages for understandability.
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=2000,  # Train the agent for 2000 steps
    eval_n_steps=None,  # We evaluate for episodes, not time
    eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
    train_max_episode_len=200,  # Maximum length of each episode
    eval_interval=1000,  # Evaluate the agent after every 1000 steps
    outdir='result',  # Save everything to 'result' directory
)
