import mdps
import algorithms

def simulate_MDP_Algorithm():
    mdp = mdps.GridMDP(side_length=5)
    solver = algorithms.PolicyIteration(max_iterations=1000, epsilon=0.01)
    solver.solve(mdp)
    mdp.print_v_grid(solver.V)
    mdp.print_pi_grid(solver.pi)

def simulate_RL_Algorithm():
    pass

if __name__ == '__main__':
    simulate_MDP_Algorithm()
