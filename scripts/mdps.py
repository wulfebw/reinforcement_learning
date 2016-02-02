
import collections
import copy
import numpy as np
import random
import sys

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

    def get_default_action(self):
        return (-1, 0)

    def succAndProbReward(self, state, action): 
        # if terminal state return empty successors and final reward
        if state in self.terminal_states:
            return []

        successors = []

        x, y = state
        dx, dy = action
        new_x = max(0, min(x + dx, self.side_length - 1))
        new_y = max(0, min(y + dy, self.side_length - 1))
        new_state = new_x, new_y
        reward = self.rewards[new_state]
        move_prob = .6
        successors.append(((new_x, new_y), move_prob, reward))

        mistakes = [(1,0),(-1,0),(0,1),(0,-1)]
        slip_prob = (1.0 - move_prob) / len(mistakes)
        for dx, dy in mistakes:
            new_x = max(0, min(x + dx, self.side_length - 1))
            new_y = max(0, min(y + dy, self.side_length - 1))
            new_state = new_x, new_y
            reward = self.rewards[new_state]
            successors.append(((new_x, new_y), slip_prob, reward))

        return successors

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

    def print_pi(self, pi):
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

    def print_v(self, V):
        if self.grid is None:
            self.build_grid()

        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col != 'S' and col != 'E':
                    if (i,j) in V:
                        self.grid[i][j] = round(V[(i, j)], 2)

        self.print_grid()

class LineMDP(MDP):
    """
    :description: A line mdp is just an x axis. Here the rewards are all -1 except for the last state on the right which is +1.
    """
    
    def __init__(self, length):
        self.length = length
        self.start_state = 0
        self.discount = 1

    def get_default_action(self):
        return 1

    def actions(self, state):
        return [-1, 1]

    def succAndProbReward(self, state, action): 
        if state == self.length:
            return []

        nextState = max(-self.length, state + action)
        reward = 1 if nextState == self.length else -1
        return [(nextState, 1, reward)]

    def print_v(self, V):
        line = ['-'] * (self.length * 2)
        for vidx, lidx in zip(range(-self.length, self.length), range(self.length * 2)):
            if vidx in V:
                line[lidx] = round(V[vidx], 2)
        print line

    def print_pi(self, pi):
        line = ['-'] * (self.length * 2)
        for pidx, lidx in zip(range(-self.length, self.length), range(self.length * 2)):
            if pidx in pi:
                line[lidx] = round(pi[pidx], 2)
        print line

class TriggerMDP(MDP):
    """
    :description: A trigger mdp where if the agent hits the most negative position then on completing the mdp to right it receives a much higher reward
    """
    
    def __init__(self, length):
        self.length = length
        self.start_state = (0, False)
        self.discount = 1
        self.triggered = False

    def get_default_action(self):
        return 1

    def actions(self, state):
        return [-1, 1]

    def succAndProbReward(self, state, action): 
        # if we reach the far right then this episode ends
        if state[0] == self.length:
            self.triggered = False
            return []

        if state[0] == -self.length:
            self.triggered = True

        # o/w return a single, deterministic transition in the direction of the action
        nextState = max(-self.length, state[0] + action)

        reward = -1
        # if we have triggered the switch then the reward is must higher
        if nextState == self.length:
            if self.triggered == True:
                reward = 1000
            else:
                reward = 1
        nextState = (nextState, self.triggered)

        return [(nextState, 1, reward)]

    def print_v(self, V):
        line = np.zeros(self.length * 4).reshape(2, self.length * 2)
        for ridx, row in enumerate(line):
            for vidx, cidx in zip(range(-self.length, self.length), range(self.length * 2)):
                triggered = False if ridx == 0 else True
                if (vidx, triggered) in V:
                    line[ridx, cidx] = round(V[(vidx, triggered)], 2)
        print line

    def print_pi(self, pi):
        line = ['-'] * (self.length * 2)
        for pidx, lidx in zip(range(-self.length, self.length), range(self.length * 2)):
            if pidx in pi:
                line[lidx] = round(pi[pidx], 2)
        print line

class MazeMDP(MDP):
    """
    :description: an MDP specifying a maze, where that maze is a square, consists of num_rooms and each room having room_size discrete squares in it. So can have 1x1, 2x2, 3x3, etc size mazes. Rooms are separated by walls with a single entrance between them. The start state is always the bottom left of the maze, 1 position away from each wall of the first room. The end state is always in the top right room of the maze, again 1 position away from each wall. So the 1x1 maze looks like:

         _______
        |       |
        |     E |
        | S     |
        |       |
         -------

         the 2x2 maze would be

         _______ _______   
        |       |       |
        |             E |
        |               |
        |       |       |
         --   -- --   --
         __   __ __   __
        |       |       |
        |               |
        | S             |
        |       |       |
         ------- -------


         state is represented in absolute terms, so the bottom left corner of all mazes is (0,0) and to top right corner of all mazes is (room_size * num_rooms - 1, room_size * num_rooms - 1). In other words, the state ignores the fact that there are rooms or walls or anything, it's just the coordinates.

         actions are N,E,S,W movement by 1 direction. No stochasticity for now. moving into a wall leaves agent in place. Rewards are nothing except finding the exit is worth a lot 

         room_size must be odd
    """

    EXIT_REWARD = 100
    MOVE_REWARD = 0

    def __init__(self, room_size, num_rooms):
        self.room_size = room_size
        self.num_rooms = num_rooms
        self.discount = 1
        self.start_state = (1,1)
        self.max_position = self.room_size * self.num_rooms - 1
        self.end_state = (self.max_position - 1, self.max_position - 1)

    def get_default_action(self):
        return (1,0)

    def actions(self, state):
        return [(1,0),(-1,0),(0,1),(0,-1)]   

    def calculate_next_state(self, state, action):
        return state[0] + action[0], state[1] + action[1]

    def runs_into_wall(self, state, action):
        next_state = self.calculate_next_state(state, action)

        # 1. check for leaving the maze
        if next_state[0] > self.max_position or next_state[0] < 0 \
                            or next_state[1] > self.max_position or next_state[1] < 0:
            return True

        # 2. check if movement was through doorway and if so return false
        doorway_position = (self.room_size) / 2
        # check horizontal movement through doorway
        if next_state[0] != state[0]:
            if next_state[1] % self.room_size == doorway_position:
                return False

        # check vertical movement through doorway
        if next_state[1] != state[1]:
            if next_state[0] % self.room_size == doorway_position:
                return False

        # 3. check if movement was through a wall
        room_size = self.room_size
        # move right to left through wall
        if state[0] % room_size == room_size - 1 and next_state[0] % room_size == 0:
            return True

        # move left to right through wall
        if next_state[0] % room_size == room_size - 1 and state[0] % room_size == 0:
            return True

        # move up through wall
        if state[1] % room_size == room_size - 1 and next_state[1] % room_size == 0:
            return True

        # move down through wall
        if next_state[1] % room_size == room_size - 1 and state[1] % room_size == 0:
            return True

        # if none of the above conditions meet, then have not passed through wall
        return False

    def succAndProbReward(self, state, action): 

        # if we reach the end state then the episode ends
        if np.array_equal(state, self.end_state):
            return []

        if self.runs_into_wall(state, action):
            # if the action runs us into a wall do nothing
            next_state = state
        else:
            # o/w determine the next position
            next_state = self.calculate_next_state(state, action)

        # if next state is exit, then set reward
        reward = self.MOVE_REWARD
        if np.array_equal(next_state, self.end_state):
            reward = self.EXIT_REWARD

        return [(next_state, 1, reward)]

    def print_v(self, V):
        for ridx in reversed(range(self.max_position + 1)):
            for cidx in range(self.max_position + 1):
                if (ridx, cidx) in V:
                    print round(V[(ridx, cidx)], 1),
            print('\n')

    def print_maze(self, coordinates):
        for row in range(self.room_size):
            for col in range(self.room_size):
                if coordinates == (row,col):
                    print '*',
                elif self.end_state == (row,col):
                    print 'e',
                else:
                    print '-',
            print '\n'
        print '\n'

    def print_trajectory(self, actions):
        coordinates = self.start_state
        self.print_maze(coordinates)
        for action in actions:
            coordinates = (coordinates[0] + action[0], coordinates[1] + action[1])
            self.print_maze(coordinates)

if __name__ == '__main__':
    mdp = MazeMDP(5,1)
    print mdp.end_state
    trajectory = [(1,0),(1,0),(0,1)]
    mdp.print_trajectory(trajectory)