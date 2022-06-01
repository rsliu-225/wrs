import pickle

import pfrl
import torch
import torch.nn as nn
import numpy as np

import bendplanner.bend_utils as bu
from rf_planner.opt_value.bend_env import BendEnv

goal_pseq = pickle.load(open('goal_pseq.pkl', 'rb'))
init_pseq = [(0, 0, 0), (0, bu.cal_length(goal_pseq), 0), (0, bu.cal_length(goal_pseq) + .1, 0)]
init_rotseq = [np.eye(3), np.eye(3), np.eye(3)]

env = BendEnv(goal_pseq=goal_pseq, pseq=init_pseq, rotseq=init_rotseq)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Linear(200 * 200 * 200, 128),
    nn.LeakyReLU(0.2),
    nn.Linear(128, 64),
    nn.LeakyReLU(0.2),
    nn.Linear(64, 4),
    pfrl.policies.GaussianHeadWithStateIndependentCovariance(action_size=4),
)
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

agent = pfrl.agents.REINFORCE(
    model,
    opt,
    gpu=0,
    beta=1e-4,
    batchsize=10,
    max_grad_norm=1.0,
)

n_episodes = 2000
max_episode_len = 30
for i in range(1, n_episodes + 1):
    env.reset()
    obs = env.get_observation()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # TODO: here we use flattened observation, in the future, the agent should use pointnet to extract point feature
        action = agent.act(obs.flatten().astype(np.float32))
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 5 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
env.visualize_observation()
print('Finished.')

with agent.eval_mode():
    for i in range(10):
        env.reset()
        obs = env.get_observation()
        R = 0
        t = 0
        while True:
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
