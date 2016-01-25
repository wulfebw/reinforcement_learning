import sys
import copy
import collections
import operator
import random

def has_converged(V, prev_V, epsilon):
    total = 0
    for k, v in V.iteritems():
        prev_v = prev_V[k]
        total += abs(v - prev_v)

    if total < epsilon:
        return True
    return False

###########################################################################
### From Stanford CS221 Artificial Intelligence ###########################

# An algorithm that solves an MDP (i.e., computes the optimal
# policy).
class MDPAlgorithm(object):
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): 
        raise NotImplementedError("Override me")

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm(object):
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): 
        raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): 
        raise NotImplementedError("Override me")

###########################################################################
### Dynamic Programming methods ###########################################

class PolicyIteration(MDPAlgorithm):

    def __init__(self, max_iterations, epsilon):
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def policy_evaluation(self, mdp):
        for t in range(self.max_iterations):
            prev_V = self.V
            for state in mdp.states:
                action = self.pi[state]
                discount = mdp.discount
                self.V[state] = sum(prob * (reward + discount * self.V[newState]) 
                                    for newState, prob, reward in mdp.succAndProbReward(state, action))
            if has_converged(self.V, prev_V, self.epsilon):
                break

    def policy_improvement(self, mdp):
        # calcualte Q(s,a)
        Q = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        discount = mdp.discount
        for state in mdp.states:
            for action in mdp.actions(state):
                Q[state][action] = sum(prob * (reward + discount * self.V[newState]) 
                            for newState, prob, reward in mdp.succAndProbReward(state, action))

        # update policy
        for state, actions in Q.iteritems():
            new_pi = max(actions.iteritems(), key=operator.itemgetter(1))[0]
            self.pi[state] = new_pi

    def solve(self, mdp):
        mdp.computeStates()
        self.pi = {}
        default_action = mdp.get_default_action()
        for state in mdp.states:
            self.pi[state] = default_action
        self.V = collections.defaultdict(lambda: 0)
        prev_V = collections.defaultdict(lambda: 0)

        for t in range(self.max_iterations):
            self.policy_evaluation(mdp)
            self.policy_improvement(mdp)
            if has_converged(self.V, prev_V, self.epsilon):
                break

class ValueIteration(MDPAlgorithm):

    def __init__(self, max_iterations, epsilon):
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def max_policy_evaluation(self, mdp):
        for state in mdp.states:
            discount = mdp.discount
            value, action = max((sum(prob * (reward + discount * self.V[newState]) 
                                for newState, prob, reward in mdp.succAndProbReward(state, action)), 
                                    action) for action in mdp.actions(state))
            self.V[state] = value
            self.pi[state] = action

    def solve(self, mdp):
        mdp.computeStates()
        self.pi = {}
        default_action = mdp.get_default_action()
        for state in mdp.states:
            self.pi[state] = default_action
        self.V = collections.defaultdict(lambda: 0)
        prev_V = collections.defaultdict(lambda: 0)

        for t in range(self.max_iterations):
            self.max_policy_evaluation(mdp)
            if has_converged(self.V, prev_V, self.epsilon):
                break

###########################################################################
### Reinforcement learning methods ########################################

### Temporal Difference Methods ###########################################
class RLAlgorithm(object):
    """
    :description: abstract class defining the interface of a RL algorithm
    """

    def getAction(self, state):
        raise NotImplementedError("Override me")

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("Override me")

class ValueLearningAlgorithm(RLAlgorithm):
    """
    :description: base class for RL algorithms that approximate the value function.
    """
    def __init__(self, actions, discount, explorationProb, stepSize):
        """
        :type: actions: list
        :param actions: possible actions to take

        :type discount: float
        :param discount: the discount factor

        :type explorationProb: float
        :param explorationProb: probability of taking a random action

        :type stepSize: float
        :param stepSize: learning rate
        """
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 1
        self.stepSize = stepSize

    def feature_extractor(self, state, action):
        """
        :description: this is the identity feature extractor, so we use tables here for the function
        """
        return [((state, action), 1)]

    def getQ(self, state, action):
        """
        :description: returns the Q value associated with this state-action pair

        :type state: dictionary
        :param state: the state of the game

        :type action: int
        :param action: the action for which to retrieve the Q-value
        """
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        """
        :description: returns an action accoridng to epsilon-greedy policy

        :type state: dictionary
        :param state: the state of the game
        """
        self.numIters += 1

        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            maxAction = max((self.getQ(state, action), action) for action in self.actions)[1]
        return maxAction

    def getStepSize(self):
        """
        :description: return the step size
        """
        return self.stepSize

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("Override me")

class QLearningAlgorithm(ValueLearningAlgorithm):
    """
    :description: Class implementing the Q-learning algorithm
    """
    def __init__(self, actions, discount, explorationProb, stepSize):
        super(QLearningAlgorithm, self).__init__(actions, discount, explorationProb, stepSize)

    def incorporateFeedback(self, state, action, reward, newState):
        """
        :description: performs a Q-learning update

        :type state: dictionary
        :param state: the state of the game

        :type action: int
        :param action: the action for which to retrieve the Q-value

        :type reward: float
        :param reward: reward associated with being in newState

        :type newState: dictionary
        :param newState: the new state of the game

        :type rval: int or None
        :param rval: if rval returned, then this is the next action taken
        """
        stepSize = self.stepSize
        prediction = self.getQ(state, action)
        target = reward
        if newState != None:
            target += self.discount * max(self.getQ(newState, newAction) for newAction in self.actions)

        for f, v in self.feature_extractor(state, action):
            self.weights[f] = self.weights[f] + stepSize * (target - prediction) * v

class SARSALearningAlgorithm(ValueLearningAlgorithm):
    """
    :description: Class implementing the SARSA algorithm
    """
    def __init__(self, actions, discount, explorationProb, stepSize):
        super(SARSALearningAlgorithm, self).__init__(actions, discount, explorationProb, stepSize)

    def incorporateFeedback(self, state, action, reward, newState):
        """
        :description: performs a SARSA update

        :type state: dictionary
        :param state: the state of the game

        :type action: int
        :param action: the action for which to retrieve the Q-value

        :type reward: float
        :param reward: reward associated with being in newState

        :type newState: dictionary
        :param newState: the new state of the game

        :type rval: int or None
        :param rval: if rval returned, then this is the next action taken
        """
        stepSize = self.stepSize
        prediction = self.getQ(state, action)
        target = reward
        newAction = None

        if newState != None:
            newAction = self.getAction(newState)
            target += self.discount * self.getQ(newState, newAction)

        for f, v in self.feature_extractor(state, action):
            self.weights[f] = self.weights[f] + stepSize * (target - prediction) * v

        # return newAction to denote that this is an on-policy algorithm
        return newAction

### Batch Methods #########################################################

class KADP(object):

    def __init__(self, max_iterations, epsilon, discount):
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.discount = discount

    def build_kernel(self, replay):
        """
        :description: builds a kernel to use as a similarity metric. Must define a convex combination of values within individual actions. This needs to be improved.
        """
        distances = collections.defaultdict(lambda: 0)
        actions = collections.defaultdict(lambda: 0)
        counts = collections.defaultdict(lambda: 0)
        self._kernel = collections.defaultdict(lambda: 0)
        for key, (sigma, action, reward, state) in replay.memory.iteritems():
                dist = (state[0] - sigma[0]) ** 2 + (state[1] - sigma[1]) ** 2
                distances[(sigma, state)] = 2 ** -dist
                actions[action] += 2 ** -dist

        self._kernel = collections.defaultdict(lambda: 0)
        for (state, newState), dist in distances.iteritems():
            for action, total in actions.iteritems():
                self._kernel[(state, newState, action)] = dist / 4.0

        print self._kernel

    def kernel(self, state, cur_state, action):
        return self._kernel[(state, cur_state, action)]

    def set_actions_states(self, replay):
        actions = set()
        states = set()
        for key, (state, action, reward, newState) in replay.memory.iteritems():
            actions.add(action)
            states.add(state)
        self.actions = list(actions)
        self.states = list(states)

    def fit(self, replay):

        self.V = collections.defaultdict(lambda: 0)
        self.set_actions_states(replay)
        self.num_states = len(self.states)
        self.build_kernel(replay)

        for iteration in range(self.max_iterations):
            prev_V = copy.deepcopy(self.V)
            
            for sigma in self.states:
                action_totals = collections.defaultdict(lambda: 0)
                for key, (state, action, reward, newState) in replay.memory.iteritems():

                    weight = self.kernel(state, sigma, action)
                    value = reward + self.discount * prev_V[newState]
                    action_totals[action] += weight * value

                new_value = max(action_totals.iteritems(), key=lambda (k,v): v)[1]
                self.V[sigma] = new_value

            if has_converged(self.V, prev_V, self.epsilon):
                break

class FittedQIteration(object):

    def __init__(self, max_iterations, epsilon, discount, learning_rate=0.1):
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate
        self.weights = collections.defaultdict(lambda: 0)

    def feature_extractor(self, state, action):
        """
        :description: this is the identity feature extractor, so we use tables here for the function
        """
        return [((state, action), 1)]

    def getQ(self, state, action):
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += self.weights[f] * v
        return score

    def build_dataset(self, replay):
        dataset = []
        for key, (state, action, reward, newState) in replay.memory.iteritems():
            # over the newState and newAction because it's what's next
            target = reward + self.discount * max(self.getQ(newState, newAction) 
                        for newAction in self.actions)
            dataset.append((state, action, target))
        return dataset

    def update_weights(self, dataset):
        for state, action, target in dataset:
            prediction = self.getQ(state, action)
            for f, v in self.feature_extractor(state, action):
                self.weights[f] = self.weights[f] + self.learning_rate * (target - prediction) * v

    def set_V(self):
        V = collections.defaultdict(lambda: 0)
        for state in self.states:
            V[state] = max(self.getQ(state, action) for action in self.actions)
        self.V = V

    def set_actions_states(self, replay):
        actions = set()
        states = set()
        for key, (state, action, reward, newState) in replay.memory.iteritems():
            actions.add(action)
            states.add(state)
        self.actions = list(actions)
        self.states = list(states)

    def fit(self, replay):
        self.set_actions_states(replay)
        prev_weights = collections.defaultdict(lambda: 0)
        for iteration in range(self.max_iterations):
            dataset = self.build_dataset(replay)
            self.update_weights(dataset)
            if has_converged(self.weights, prev_weights, self.epsilon):
                break
        self.set_V()

### Hierarchical Methods ##################################################

class MaxQQ(object):

    def __init__(self, actions, discount, explorationProb, stepSize):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 1
        self.stepSize = stepSize

    def feature_extractor(self, state, action):
        """
        :description: this is the identity feature extractor, so we use tables here for the function
        """
        return [((state, action), 1)]

    def getAction(self, state):
        pass

    def incorporateFeedback(self, state, action, reward, newState): 
        pass




