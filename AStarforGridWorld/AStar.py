"""
    Name: Beyza
    Surname: Kaya
    Student ID: S021747
"""


from tree_search_agents.TreeSearchAgent import *
from tree_search_agents.PriorityQueue import PriorityQueue
import time


class AStarAgent(TreeSearchAgent):
    def run(self, env: Environment) -> (List[int], float, list):
        """
            You should implement this method for A* algorithm.

            DO NOT CHANGE the name, parameters and output of the method.
        :param env: Environment
        :return: List of actions and total score
        """
        path = []
        visited_nodes = []
        queue = PriorityQueue()
        current_state = {
            'current_position': [env.to_state(env.current_position)], # S -> A -> B -> C
            'current_reward': 0, #expanded nodes reward
            'action_list':[]  # which action I am choosing as a list --> up-down-left-right
        }
        queue.enqueue(current_state ,0) # starting node'u current node yaptık

        all_goals = []
        while not queue.is_empty(): #recursive devam
            current_state = queue.dequeue() #queue'da bir sey varsa en yüksek priori node'u aldım(costu az olanı)
            current_position_state = current_state['current_position'][-1]
            #print(f"current_position_state after dequeue {env.to_position(current_position_state)}")
            # print(f"current states after dequeue {current_state}")
            if env.is_done(env.to_position(current_position_state)):
                all_goals.append((current_state['current_reward'], current_state['current_position'][-1]))
                # print(all_goals)
                return current_state['action_list'], current_state['current_reward'], current_state['current_position']
            visited_nodes.append(current_position_state)
            env.set_current_state(current_position_state)
            for action in range(4):
                #print(f"current state : {env.to_position(current_state['current_position'][-1])}")
                new_next_state, new_reward, is_done = env.move(action) 
                #print(f"next state: {env.to_position(next_state)}")
                # print(current_state['current_position'])
                # visited_nodes.append(current_state['current position'])
                if new_next_state not in visited_nodes:
                    # current_state['current_position'].append(next_state)
                    new_total_reward = new_reward + current_state['current_reward']
                    # current_state['action_list'].append(action)
                    # queue.enqueue(current_state, current_state['current_reward'])
                    # print(f"next state {env.to_state(next_state)}")
                    #print(f"position meaning for 15 {env.to_position(15)}")
                    #next_state_cost = current_state['current_reward'] + new_reward
                    #next_state_action_list = current_state['action_list'].append(action)
                    new_current_list = current_state['current_position'].copy()
                    new_current_list.append(new_next_state)
                    new_action_list = current_state['action_list'].copy()
                    new_action_list.append(action)

                    
                    next_state = {
                        'current_position':new_current_list,
                        'current_reward':new_total_reward,
                        'action_list':new_action_list
                    }
                    
                    remaining_cost = - self.get_heuristic(env, new_next_state) + new_total_reward
                    #print(f"last element of current position {next_state['current_position'][-1]}")
                    # total_cost = new_total_reward + remaining_cost
                    queue.enqueue(next_state, remaining_cost)
                env.set_current_state(current_position_state)
        



        return [], 0., []

    def get_heuristic(self, env: Environment, state: int) -> float:
        """
            You should implement your heuristic calculation for A*

            DO NOT CHANGE the name, parameters and output of the method.

            Note that you can use kwargs to get more parameters :)
        :param env: Environment object
        :param state: Current state
        :param kwargs: More parameters
        :return: Heuristic score
        """
        current_position = env.to_position(state)
        return min(abs(current_position[0] - env.to_position(goal_state)[0]) + abs(current_position[1] - env.to_position(goal_state)[1]) for goal_state in env.get_goals())

        

    @property
    def name(self) -> str:
        return "AStar"
