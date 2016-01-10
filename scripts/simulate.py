import random
import collections

import mdps
import algorithms
import replay_memory


def simulate_MDP_Algorithm():
    mdp = mdps.GridMDP(side_length=5)
    solver = algorithms.PolicyIteration(max_iterations=1000, epsilon=0.001)
    solver.solve(mdp)
    mdp.print_v_grid(solver.V)

def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

def simulate_online_RL_Algorithm(numTrials=10000, maxIterations=1000):
    mdp = mdps.GridMDP(side_length=5)
    mdp.computeStates()
    actions = mdp.actions(None)
    discount = 0.9
    explorationProb = 0.5
    stepSize = 0.1
    action = None
    rl = algorithms.SARSALearningAlgorithm(actions, discount, explorationProb, stepSize)

    for trial in range(numTrials):
        state = mdp.start_state

        for _ in range(maxIterations):
            if action is None:
                action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action)

            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            action = rl.incorporateFeedback(state, action, reward, newState)
            state = newState
    
    V = {}
    pi = {}
    rl.explorationProb = 0
    for state in mdp.states:
        action = rl.getAction(state)
        pi[state] = action
        V[state] = rl.getQ(state, action)
    mdp.print_v_grid(V)
    mdp.print_pi_grid(pi)

def gather_data(mdp, numTrials=10000, maxIterations=1000):
    mdp.computeStates()
    actions = mdp.actions(None)
    replay = replay_memory.ReplayMemory()

    for trial in range(numTrials):
        state = mdp.start_state
        if replay.isFull():
            break
        for _ in range(maxIterations):
            action = random.choice(actions)
            transitions = mdp.succAndProbReward(state, action)
            if len(transitions) == 0:
                break
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            replay.store((state, action, reward, newState))
            state = newState
    return replay

def simulate_offline_RL_Algorithm():
    rewards = collections.defaultdict(lambda: 0)
    mdp = mdps.GridMDP(side_length=10, terminal_states=[(0,0),(1,1)], rewards={(0,0):10, (1,1):-10})
    replay = gather_data(mdp)
    solver = algorithms.FittedQIteration(max_iterations=10, epsilon=0.01, discount=.9)
    solver.fit(replay)
    mdp.print_v_grid(solver.V)

if __name__ == '__main__':
    simulate_offline_RL_Algorithm()

