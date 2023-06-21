"""
    Name: Beyza
    Surname: Kaya
    Student ID: S021747
"""


from tree_search_agents.TreeSearchAgent import *
from tree_search_agents.PriorityQueue import PriorityQueue
import time


class GreedyBestAgent(TreeSearchAgent):
    def run(self, env: Environment) -> (List[int], float, list):

        all_goals = []
        queue = PriorityQueue()
        state = {
            'current_position': [env.to_state(env.current_position)], # S -> A -> B -> C
            'current_heuristic': 0, #expanded nodes reward
            'current_reward' : 0,
            'action_list':[]  # which action I am choosing as a list --> up-down-left-right
        }
        queue.enqueue(state,0)
        expand_list = set()

        while not queue.is_empty():
            state = queue.dequeue()
            current_position= state['current_position'][-1] 
            if env.is_done(env.to_position(current_position)):
                # all_goals.append((state['current_heuristic'], state['current_position'][-1]))
                # print(all_goals)
                return state['action_list'], state['current_heuristic'], state['current_position']
            # expand_list.add(current_position)
            env.set_current_state(current_position)
            for action in range (4):
                new_next_state, new_reward, is_done = env.move(action)

                if new_next_state not in expand_list:
                    new_total_heuristic = -state['current_heuristic'] -self.get_heuristic(env, new_next_state) 
                    new_total_reward = state['current_reward'] + new_reward
                    new_current_list = state['current_position'].copy()
                    new_current_list.append(new_next_state)
                    new_action_list = state['action_list'].copy()
                    new_action_list.append(action)

                    
                    next_state = {
                        'current_position':new_current_list,
                        'current_heuristic':new_total_reward,
                        'action_list':new_action_list,
                        'current_reward': new_total_reward
                    }
                    queue.enqueue(next_state, new_total_heuristic)
                env.set_current_state(current_position)    

        return [], 0., []

    def get_heuristic(self, env: Environment, state: int) -> float:
        current_position = env.to_position(state)
        return min(abs(current_position[0] - env.to_position(goal_state)[0]) + abs(current_position[1] - env.to_position(goal_state)[1]) for goal_state in env.get_goals())
    @property
    def name(self) -> str:
        return "GreedyBest"





