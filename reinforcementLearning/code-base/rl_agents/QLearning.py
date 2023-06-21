"""
    Name:
    Surname:
    Student ID:
"""

from Environment import Environment
from rl_agents.RLAgent import RLAgent
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import os

figure_folder = "figures"
os.makedirs(figure_folder, exist_ok=True)


class QLearningAgent(RLAgent):
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

    def train(self, technique: str):
        """
        DO NOT CHANGE the name, parameters and return type of the method.

        You will fill the Q-Table with Q-Learning algorithm.

        :param kwargs: Empty
        :return: Nothing
        """
        if technique == 'average_reward':
            self.train_with_average_reward()
        elif technique == 'softmax':
            self.train_with_softmax()
        else:
            self.train_with_epsilon_greedy()
        
        
    
    def train_with_average_reward(self):
        episode_rewards = []
        iteration = 0
    
        while iteration <= self.max_episode:
            state = self.env.reset()
            done = False

            while not done:
                action = self.act(state, is_training=True,strategy='average_reward')
                next_state, reward, done = self.env.move(action)

                td_target = reward + self.discount_rate * np.max(self.Q[next_state, :])
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error

                state = next_state
                episode_rewards.append(reward)
                

                average_reward = self.calculate_average_reward(episode_rewards)
                self.decay_epsilon(average_reward)
                # print(f"Episode {iteration}: Reward: {sum(episode_rewards)}")
                # print(f"Epsilon: {self.epsilon}")
                # print("Q-values:")
                # for i in range(self.state_size):
                #     print(f"State {i}: {self.Q[i, :]}")
                
                iteration += 1

        print("Training complete!")
    
    def train_with_softmax(self):
        episode_rewards = []
        iteration = 0

        while iteration <= self.max_episode:
            state = self.env.reset()
            done = False

            while not done:
                action = self.act_with_softmax(state, is_training=True)
                next_state, reward, done = self.env.move(action)

                td_target = reward + self.discount_rate * np.max(self.Q[next_state, :])
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error

                state = next_state
                episode_rewards.append(reward)

            average_reward = self.calculate_average_reward(episode_rewards)
            self.decay_epsilon(average_reward)
            # print(f"Episode {iteration}: Reward: {sum(episode_rewards)}")
            # print(f"Epsilon: {self.epsilon}")
            # print("Q-values:")
            # for i in range(self.state_size):
            #     print(f"State {i}: {self.Q[i, :]}")

            iteration += 1

        print("Training complete!")
    def train_with_epsilon_greedy(self):
        episode_rewards = []
        total_reward = 0
        td_values = []
    
        for episode in range(self.max_episode):
            state = self.env.reset()
            done = False

            while not done:
                action = self.act(state, is_training=True, strategy='epsilon_greedy')
                next_state, reward, done = self.env.move(action)

                td_target = reward + self.discount_rate * np.max(self.Q[next_state, :])
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error

                state = next_state
                total_reward += reward
            episode_rewards.append(total_reward)
            td_values.append(abs(td_error))
           
            average_reward = self.calculate_average_reward(episode_rewards)
            self.decay_epsilon(average_reward)
            # print(f"Episode {episode}: Reward: {sum(episode_rewards)}")
            # print(f"Epsilon: {self.epsilon}")
            # print("Q-values:")
            # for i in range(self.state_size):
            #     print(f"State {i}: {self.Q[i, :]}")
        plt.figure(1)
        plt.plot(range(self.max_episode), episode_rewards)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Iteration Number vs. Score')
        plt.savefig(os.path.join(figure_folder, 'iteration_vs_score.png'))  # Save the graph in the figures folder
        plt.show()

        
        plt.figure(2)
        plt.plot(range(len(td_values)), td_values)
        plt.xlabel('Step')
        plt.ylabel('TD Value')
        plt.title('TD Values over Iterations')
        plt.savefig(os.path.join(figure_folder, 'td_values.png'))  # Save the graph in the figures folder
        plt.show()

        print("Training complete!")




        # iteration = 0
        # self.epsilon = 1
        # episode_rewards = []

        # # while iteration <= self.max_episode:
        # for episode in range(self.max_episode):
        #     state = self.env.reset()  
        #     done = False  
            
        #     while not done:
        #         action = self.act(state, is_training=True)
        #         next_state, reward, done = self.env.move(action)

        #         td_target = reward + self.discount_rate * np.max(self.Q[next_state, :])
        #         td_error = td_target - self.Q[state, action]
        #         self.Q[state, action] += self.alpha * td_error

        #         state = next_state

        #         episode_rewards.append(reward)
        #         self.decay_epsilon(episode_rewards)

            # Decay the exploration rate epsilon
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay

            #     average_reward = self.calculate_average_reward(episode_rewards)
            #     self.decay_epsilon(average_reward)
            #     print(f"Episode {episode}: Reward: {episode_rewards}")
            #     print(f"Epsilon: {self.epsilon}")
            #     print("Q-values:")
            #     for i in range(self.state_size):
            #         print(f"State {i}: {self.Q[i, :]}")
           
            # print("Training complete!")

    
    def act(self, state: int, is_training: bool, strategy: bool) -> int:
        """
        DO NOT CHANGE the name, parameters and return type of the method.

        This method will decide which action will be taken by observing the given state.

        In training, you should apply epsilon-greedy approach. In validation, you should decide based on the Policy.

        :param state: Current State as Integer not Position
        :param is_training: If training use e-greedy, otherwise decide action based on the Policy.
        :return: Action as integer
        """

        if strategy == 'average_reward':
            return self.act_with_average_reward(state, is_training)
        elif strategy == 'softmax':
            return self.act_with_softmax(state, is_training)
        else:
            return self.act_with_epsilon_greedy(state, is_training)
    
    def act_with_average_reward(self, state: int, is_training: bool) -> int:
        if is_training:
            if np.random.random() < (1 - self.epsilon):
                action = np.argmax(self.Q[state, :])
            else:
                action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def act_with_softmax(self, state: int, is_training: bool) -> int:
        if is_training:
            logits = np.log(np.sum(np.exp(self.Q[state, :] / self.epsilon)))
            max_logit = np.max(logits)
            shifted_logits = logits - max_logit  # Subtract the maximum value for numerical stability
            probabilities = np.exp(shifted_logits)
            probabilities_sum = np.sum(probabilities)
            if probabilities_sum > 0:
                probabilities /= probabilities_sum
            else:
                probabilities = np.ones_like(probabilities) / len(probabilities)  # Uniform distribution if sum is zero
            action = np.random.choice(self.action_size, p=probabilities)
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def act_with_epsilon_greedy(self, state: int, is_training: bool) -> int:
        if is_training:
            if np.random.random() < (1 - self.epsilon):
                action = np.argmax(self.Q[state, :])
            else:
                action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.Q[state, :])
        return action
    def calculate_average_reward(self, rewards: List[float]) -> float:
        """
        Calculate the average reward for a given list of rewards.

        :param rewards: List of rewards
        :return: Average reward
        """
        return sum(rewards) / len(rewards)

    def decay_epsilon(self, average_reward: float) -> None:
        """
        Decay the epsilon value based on the average reward.

        :param average_reward: Average reward
        """
        if average_reward > 0:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
        
        
        
        
        # This is epsilon greedy approach
        # if is_training:
        #     if np.random.random() < (1 - self.epsilon):
                
        #         action = np.argmax(self.Q[state, :])

        #     else:
        #         action = np.random.randint(self.action_size)

        #     return action
        # else:
            
        #     action = np.argmax(self.Q[state, :])
        #     return action
        # if is_training:
        #     # Softmax action selection
        #     probabilities = np.exp(self.Q[state, :] / self.epsilon)
        #     probabilities /= np.sum(probabilities)
        #     action = np.random.choice(self.action_size, p=probabilities)
        # else:
        
        #     action = np.argmax(self.Q[state, :])
        # return action
