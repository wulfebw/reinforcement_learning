import sys
import copy
import random
import collections

###########################################################################
# From Stanford CS221 Artificial Intelligence 
# An abstract class representing a Markov Decision Process (MDP).
class MDP(object):
    # Return the start state.
    def startState(self): 
        raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): 
        raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): 
        raise NotImplementedError("Override me")

    def discount(self): 
        raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.start_state)
        queue.append(self.start_state)
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

###########################################################################

class GridMDP(MDP):
    """
    :description: a 2d grid world MDP
    """
    GOAL_REWARD = 10
    NONGOAL_REWARD = 0

    def __init__(self, side_length, terminal_states=None, rewards=None, discount=0.9):
        self.side_length = side_length
        self.discount = discount
        self.grid = None

        x = random.randint(0, self.side_length - 1)
        y = random.randint(0, self.side_length - 1)
        self.start_state = x, y

        if terminal_states:
            self.terminal_states = terminal_states
        else:
            self.terminal_states = []
            x = random.randint(0, self.side_length - 1)
            y = random.randint(0, self.side_length - 1)
            self.terminal_states.append((x,y))
        
        self.rewards = collections.defaultdict(lambda: self.NONGOAL_REWARD)
        if rewards:
            self.rewards.update(rewards)
        else:
            for terminal in self.terminal_states:
                self.rewards[terminal] = self.GOAL_REWARD


    def actions(self, state):
        return [(0,1),(0,-1),(1,0),(-1,0)]

    def succAndProbReward(self, state, action): 
        # if terminal state return empty successors and final reward
        if state in self.terminal_states:
            return []
        x, y = state
        dx, dy = action
        new_x = max(0, min(x + dx, self.side_length - 1))
        new_y = max(0, min(y + dy, self.side_length - 1))
        new_state = new_x, new_y
        prob = .8
        reward = self.rewards[new_state]
        old_reward = self.rewards[state]
        return [((new_x, new_y), prob, reward), ((x,y), 1 - prob, old_reward)]

    def build_grid(self):
        row = ['-'] * self.side_length
        grid = [copy.deepcopy(row) for i in range(self.side_length)]
        x, y = self.start_state
        grid[x][y] = 'S'
        for x, y in self.terminal_states:
            grid[x][y] = 'E'
        self.grid = grid

    def print_grid(self):
        if self.grid is None:
            self.build_grid()
        for row in self.grid:
            print '|',
            for col in row:
                print col,
                print '\t',
                print '|',
            print '\n'
        print '\n'

    def print_pi_grid(self, pi):
        if self.grid is None:
            self.build_grid()

        for ridx, row in enumerate(self.grid):
            for cidx, col in enumerate(row):
                if col != 'S' and col != 'E':
                    y, x = pi[(ridx, cidx)]
                    if x == 1:
                        self.grid[ridx][cidx] = '>'
                    elif x == -1:
                        self.grid[ridx][cidx] =  '<'
                    elif y == 1:
                        self.grid[ridx][cidx] =  'd'
                    else:
                        self.grid[ridx][cidx] =  '^'

        self.print_grid()

    def print_v_grid(self, V):
        if self.grid is None:
            self.build_grid()

        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col != 'S' and col != 'E':
                    if (i,j) in V:
                        self.grid[i][j] = round(V[(i, j)], 2)

        self.print_grid()

















