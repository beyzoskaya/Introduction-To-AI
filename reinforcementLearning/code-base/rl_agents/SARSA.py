"""
    Name:
    Surname:
    Student ID:
"""

from Environment import Environment
from rl_agents.RLAgent import RLAgent
import numpy as np
import matplotlib.pyplot as plt
import os


class SARSAAgent(RLAgent):
    epsilon: float          # Current epsilon value for epsilon-greedy
    epsilon_decay: float    # Decay ratio for epsilon
    epsilon_min: float      # Minimum epsilon value
    alpha: float            # Alpha value for soft-update
    max_episode: int        # Maximum iteration
    Q: np.ndarray           # Q-Table as Numpy Array

    def __init__(self, env: Environment, seed: int, discount_rate: float, epsilon: float, epsilon_decay: float,
                 epsilon_min: float, alpha: float, max_episode: int):
        """
        Initiate the Agent with hyperparameters.
        :param env: The Environment where the Agent plays.
        :param seed: Seed for random
        :param discount_rate: Discount rate of cumulative rewards. Must be between 0.0 and 1.0
        :param epsilon: Initial epsilon value for e-greedy
        :param epsilon_decay: epsilon = epsilon * epsilonDecay after all e-greedy. Less than 1.0
        :param epsilon_min: Minimum epsilon to avoid overestimation. Must be positive or zero
        :param max_episode: Maximum episode for training
        :param alpha: To update Q values softly. 0 < alpha <= 1.0
        """
        super().__init__(env, discount_rate, seed)

        assert epsilon >= 0.0, "epsilon must be >= 0"
        self.epsilon = epsilon

        assert 0.0 <= epsilon_decay <= 1.0, "epsilonDecay must be in range [0.0, 1.0]"
        self.epsilon_decay = epsilon_decay

        assert epsilon_min >= 0.0, "epsilonMin must be >= 0"
        self.epsilon_min = epsilon_min

        assert 0.0 < alpha <= 1.0, "alpha must be in range (0.0, 1.0]"
        self.alpha = alpha

        assert max_episode > 0, "Maximum episode must be > 0"
        self.max_episode = max_episode

        self.Q = np.zeros((self.state_size, self.action_size))

        # If you want to use more parameters, you can initiate below

    def train(self, **kwargs):
        """
        DO NOT CHANGE the name, parameters and return type of the method.

        You will fill the Q-Table with SARSA algorithm.

        :param kwargs: Empty
        :return: Nothing
        """
        
        # iter = 0
        # epsilon = 1.0

        # while iter <= self.max_episode:
        #     state = self.env.reset()  # Reset the environment for a new episode
        #     done = False
        #     action = self.act(state, epsilon)

        #     while not done:
        #         next_state, reward, done = self.env.move(action)
        #         next_action = self.act(next_state, epsilon)

        #         td_target = reward + self.discount_rate * np.max(self.Q[next_state, :])
        #         td_error = td_target - self.Q[state, action]
        #         self.Q[state, action] += self.alpha * td_error

        #         state = next_state
        #         action = next_action
                
        #         print("Q-values:")
        #         for i in range(self.state_size):
        #             print(f"State {i}: {self.Q[i, :]}")

        #     if epsilon > self.epsilon_min:
        #         epsilon *= self.epsilon_decay

        #     iter += 1
        #     if iter > self.max_episode:
        #         break
        
        episode_rewards = []
        for episode in range(self.max_episode):
            state = self.env.reset()
            action = self.act(state, is_training=True)
            done = False
            total_reward = 0

            while not done:
                next_state, reward, done = self.env.move(action)
                next_action = self.act(next_state, is_training=True)

                
                target = reward + self.discount_rate * self.Q[next_state, next_action]
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])

                state = next_state
                action = next_action
                total_reward += reward
            episode_rewards.append(total_reward)
       

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # plt.plot(range(self.max_episode), episode_rewards)
        # plt.xlabel('Iteration')
        # plt.ylabel('Score')
        # plt.title('Iteration Number vs. Score')
        # plt.show()

    def act(self, state: int, is_training: bool) -> int:
        """
        DO NOT CHANGE the name, parameters and return type of the method.

        This method will decide which action will be taken by observing the given state.

        In training, you should apply epsilon-greedy approach. In validation, you should decide based on the Policy.

        :param state: Current State as Integer not Position
        :param is_training: If training use e-greedy, otherwise decide action based on the Policy.
        :return: Action as integer
        """

        if is_training:
            if np.random.random() < (1 - self.epsilon):
                
                action = np.argmax(self.Q[state, :])

            else:
                action = np.random.randint(self.action_size)

            return action
        else:
            
            action = np.argmax(self.Q[state, :])
        return action
    

    
