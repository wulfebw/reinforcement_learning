import random
import collections

import algorithms
import mdps
import learning_utils
import replay_memory

def simulate_MDP_algorithm(mdp):
    solver = algorithms.PolicyIteration(max_iterations=1000, epsilon=0.01)
    solver.solve(mdp)
    mdp.print_v(solver.V)

def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

def simulate_online_RL_algorithm(mdp, numTrials=10000, maxIterations=1000):
    mdp.computeStates()
    actions = mdp.actions(None)
    discount = mdp.discount
    explorationProb = 0.5
    stepSize = 0.05
    action = None
    rl = algorithms.SARSALearningAlgorithm(actions, discount, explorationProb, stepSize)

    total_rewards = []
    total_reward = 0

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
            total_reward += reward
            action = rl.incorporateFeedback(state, action, reward, newState)
            state = newState

        total_rewards.append(total_reward)
        total_reward = 0
    
    V = {}
    pi = {}
    rl.explorationProb = 0
    for state in mdp.states:
        action = rl.getAction(state)
        pi[state] = action
        V[state] = rl.getQ(state, action)

    return total_rewards, V

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

def simulate_offline_RL_algorithm():
    mdp = mdps.GridMDP(side_length=5)
    replay = gather_data(mdp)
    solver = algorithms.KADP(max_iterations=10, epsilon=0.001, discount=0.9)
    solver.fit(replay)
    mdp.print_v(solver.V)

def run():
    mdp = mdps.MazeMDP(room_size=5, num_rooms=5)
    print 'online RL algorithm: '
    total_rewards, V = simulate_online_RL_algorithm(mdp)
    mdp.print_v(V)
    learning_utils.plot_rewards(total_rewards)
    print 'DP algorithm: '
    simulate_MDP_algorithm(mdp)

if __name__ == '__main__':
    run()

