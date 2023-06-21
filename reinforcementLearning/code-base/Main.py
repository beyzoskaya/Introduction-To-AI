"""
    Name:
    Surname:
    Student ID:
"""

import os.path

import numpy as np
import matplotlib.pyplot as plt
import rl_agents
from Environment import Environment
import time



GRID_DIR = "grid_worlds/"


if __name__ == "__main__":
    file_name = input("Enter file name: ")

    assert os.path.exists(os.path.join(GRID_DIR, file_name)), "Invalid File"

    env = Environment(os.path.join(GRID_DIR, file_name))

    # Type your parameters
    #agents = [rl_agents.QLearningAgent(env, seed=42, discount_rate=0.9, epsilon=1, epsilon_decay=1e-4, alpha=0.01, max_episode=300, epsilon_min=0.01)]
    agents = [rl_agents.SARSAAgent(env, seed=42, discount_rate=0.99, epsilon=1, epsilon_decay=1e-4, alpha=0.01, max_episode=500, epsilon_min=0.01)]
    

    hyperparameters = {
        "discount_rate": [0.9, 0.95, 0.99],
        "epsilon": [1.0, 0.5, 0.1],
        "alpha": [0.01, 0.05, 0.1]
    }

    actions = ["UP", "LEFT", "DOWN", "RIGHT"]

    for agent in agents:
        print("*" * 50)
        print()

        env.reset()

        start_time = time.time_ns()

        episode_scores = []
        td_values = []
        for episode in range(agent.max_episode):
            agent.train()
            path, score = agent.validate()
            episode_scores.append(score)
            td_values.append(np.max(agent.Q))

        

        end_time = time.time_ns()

        path, score = agent.validate()

        print("Actions:", [actions[i] for i in path])
        print("Score:", score)
        print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)

        plt.plot(range(agent.max_episode), episode_scores)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Iteration Number vs. Score')
        plt.savefig(os.path.join(os.getcwd(), 'iteration_score.png'))  # Save the plot in the current working directory
        plt.show()

        # Plot and save the iteration-td_value graph
        plt.plot(range(agent.max_episode), td_values)
        plt.xlabel('Iteration')
        plt.ylabel('TD Value')
        plt.title('Iteration Number vs. TD Value')
        plt.savefig(os.path.join(os.getcwd(), 'td_value_score.png'))  # Save the plot in the current working directory
        plt.show()


        print("*" * 50)
        
