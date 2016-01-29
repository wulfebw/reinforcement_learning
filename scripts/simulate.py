import random
import collections

import numpy as np
import theano
import theano.tensor as T

import algorithms
import mdps
import learning_utils
import replay_memory

from qnetwork import QNetwork, HiddenLayer

########################################################
# DP algorithms:
# 1. policy iteration
# 2. value iteration

def simulate_MDP_algorithm(mdp):
    solver = algorithms.PolicyIteration(max_iterations=1000, epsilon=0.01)
    solver.solve(mdp)
    mdp.print_v(solver.V)

########################################################
# online RL algorithms
# 1. Sarsa
# 2. Q-learning

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

########################################################
# symbolic RL algorithms (i.e., those with neural net function approx):
# 1. Q-learning

INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 4
MIN_EXPLORATION_PROB = .05
MIN_LEARNING_RATE = 1e-4

def simulate_symbolic_online_RL_algorithm(mdp, num_episodes, max_iterations):
    
    real_actions = mdp.actions(None)
    actions = np.arange(len(real_actions))

    # these theano variables are used to define the symbolic input of the network
    features = T.dvector('features')
    action = T.lscalar('action')
    reward = T.dscalar('reward')
    next_features = T.dvector('next_features')
    learning_rate_symbol = T.dscalar('learning_rate')

    h1 = HiddenLayer(n_vis=INPUT_DIM,  n_hid=HIDDEN_DIM, layer_name='h1')
    h2 = HiddenLayer(n_vis=HIDDEN_DIM, n_hid=HIDDEN_DIM, layer_name='h2')
    h3 = HiddenLayer(n_vis=HIDDEN_DIM, n_hid=HIDDEN_DIM, layer_name='h3')
    h4 = HiddenLayer(n_vis=HIDDEN_DIM, n_hid=OUTPUT_DIM, layer_name='h4')
    # h5 = HiddenLayer(n_vis=HIDDEN_DIM, n_hid=HIDDEN_DIM, layer_name='h5')
    # h6 = HiddenLayer(n_vis=HIDDEN_DIM, n_hid=OUTPUT_DIM, layer_name='h6')
        
    layers = [h1, h2, h3, h4] #, h3, h4, h5, h6]
    learning_rate = 1e-2
    explorationProb = .4
    regularization_weight = 1e-5
    momentum_rate = 9e-1
    qnetwork = QNetwork(layers, discount=mdp.discount,  
                            momentum_rate=momentum_rate,
                            regularization_weight=regularization_weight)

    exploration_reduction = (explorationProb - MIN_EXPLORATION_PROB) / num_episodes
    learning_rate_reduction = (learning_rate - MIN_LEARNING_RATE) / num_episodes

    # this call gets the symbolic output of the network along with the parameter updates
    loss, updates = qnetwork.get_loss_and_updates(features, action, reward, next_features, learning_rate_symbol)

    print 'Building Training Function...'
    # this defines the theano symbolic function used to train the network
    # 1st argument is a list of inputs, here the symbolic variables above
    # 2nd argument is the symbolic output expected
    # 3rd argument is the dictionary of parameter updates
    # 4th argument is the compilation mode
    train_model = theano.function(
                    [theano.Param(features, default=np.zeros(INPUT_DIM)),
                    theano.Param(action, default=0),
                    theano.Param(reward, default=0),
                    theano.Param(next_features, default=np.zeros(HIDDEN_DIM)),
                    learning_rate_symbol],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_RUN')

    get_action = theano.function([features], qnetwork.get_action(features))

    total_rewards = []
    total_losses = []
    weight_magnitudes = []

    print 'Starting Training...'
    replay_mem = replay_memory.ReplayMemory()
    for episode in xrange(num_episodes):
        
        state = np.array(mdp.start_state)
        total_reward = 0
        total_loss = 0
        for iteration in xrange(max_iterations):
            
            if random.random() < explorationProb:
                action = random.choice(actions)
            else:
                action = get_action(state)

            real_action = real_actions[action]
            transitions = mdp.succAndProbReward(state, real_action)

            if len(transitions) == 0:
                # loss += train_model(state, action, 0, next_features)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            newState = np.array(newState)

            sars_tuple = (state, action, np.clip(reward,-1,1), newState)
            replay_mem.store(sars_tuple)
            num_samples = 5 if replay_mem.isFull() else 1
            for i in range(0, num_samples):
                random_train_tuple = replay_mem.sample()
                sample_state = random_train_tuple[0]
                sample_action = random_train_tuple[1]
                sample_reward = random_train_tuple[2]
                sample_new_state = random_train_tuple[3]

                total_loss += train_model(sample_state, sample_action, sample_reward, sample_new_state, learning_rate)

            total_reward += reward
            state = newState

        explorationProb -= exploration_reduction
        learning_rate -= learning_rate_reduction


        total_rewards.append(total_reward * mdp.discount ** iteration)
        total_losses.append(total_loss)
        weight_magnitude = qnetwork.get_weight_magnitude()
        weight_magnitudes.append(weight_magnitude)

        print 'episode: {}\t\t loss: {}\t\t reward: {}\t\tweight magnitude: {}'.format(episode, round(total_loss, 2), total_reward, weight_magnitude)

    # return the list of rewards attained
    return total_rewards, total_losses

########################################################
# offline RL algorithms
# 1. neural fitted Q-iteration
# 2. KADP

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

########################################################
# main functions

def run_nnet():
    mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
    total_rewards, total_losses = simulate_symbolic_online_RL_algorithm(mdp=mdp, num_episodes=200, max_iterations=500)
    learning_utils.plot_rewards(total_rewards)
    learning_utils.plot_rewards(total_losses)

def run():
    mdp = mdps.MazeMDP(room_size=5, num_rooms=5)
    print 'online RL algorithm: '
    total_rewards, V = simulate_online_RL_algorithm(mdp)
    mdp.print_v(V)
    learning_utils.plot_rewards(total_rewards)
    print 'DP algorithm: '
    simulate_MDP_algorithm(mdp)

if __name__ == '__main__':
    run_nnet()

