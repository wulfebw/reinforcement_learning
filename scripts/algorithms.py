import sys
import collections
import operator

###########################################################################
# From Stanford CS221 Artificial Intelligence 

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

############################################################

class PolicyIteration(MDPAlgorithm):

    def __init__(self, max_iterations, epsilon):
        self.max_iterations = max_iterations
        self.epsilon = epsilon


    def has_converged(self, V, prev_V):
        total = 0
        for k, v in V.iteritems():
            prev_v = prev_V[k]
            total += abs(v - prev_v)

        if total < self.epsilon:
            return True
        return False

    def policy_evaluation(self, mdp):
        for t in range(self.max_iterations):
            prev_V = self.V
            for state in mdp.states:
                action = self.pi[state]
                discount = mdp.discount
                self.V[state] = sum(prob * (reward + discount * self.V[newState]) 
                                    for newState, prob, reward in mdp.succAndProbReward(state, action))
            if self.has_converged(self.V, prev_V):
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
        for state in mdp.states:
            self.pi[state] = (-1, 0)
        self.V = collections.defaultdict(lambda: 0)
        prev_V = collections.defaultdict(lambda: 0)

        for t in range(self.max_iterations):
            self.policy_evaluation(mdp)
            self.policy_improvement(mdp)
            if self.has_converged(self.V, prev_V):
                break

        



