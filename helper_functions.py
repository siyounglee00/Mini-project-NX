from neurodynex3.hopfield_network import network, pattern_tools, plot_tools
from matplotlib import pyplot as plt
import numpy as np

def squarest_pattern(N):
    min_diff = float('inf')
    squarest_values = None
    
    for w in range(1, int(N**0.5) + 1):
        if N % w == 0:
            h = N // w
            diff = abs(w - h)
            if diff < min_diff:
                min_diff = diff
                squarest_values = (w, h)
    
    return squarest_values

def generate_random_patterns(M, N):
    width, height = squarest_pattern(N)
    hopfield_net = network.HopfieldNetwork(nr_neurons=N)
    factory = pattern_tools.PatternFactory(width, height)
    pattern_list = factory.create_random_pattern_list(nr_patterns=M, on_probability=0.5)
    hopfield_net.store_patterns(pattern_list)
    return hopfield_net, factory, pattern_list

def custom_function(function_name, beta, N):
    if function_name == "phi":
        def custom_f(state_s0, weights):
            h = np.sum(weights * state_s0, axis=1)
            state_s1 = np.tanh(beta * h)
            return state_s1
    elif function_name == "phi_opti":
        def custom_f(state_s0, pattern_list):
            m_list = []
            flattened_pattern_list = np.array([pattern.flatten() for pattern in pattern_list])
            for pattern in flattened_pattern_list:
                m_list.append((1/N) * np.sum(np.dot(pattern, state_s0)))

            h = np.sum(flattened_pattern_list * np.array(m_list)[:, None], axis=0)
            state_s1 = np.tanh(beta * h)
            return state_s1
    else:
        raise ValueError("The function must be 'phi' or 'phi_opti'.")
    return custom_f

def study_overlap(states_as_patterns, pattern_list, nr_steps):
    overlap = pattern_tools.compute_overlap(states_as_patterns[nr_steps], pattern_list[0])
    if overlap == 1:
        print("With {} steps, the network converged to the stored pattern.".format(nr_steps))
    elif np.round(overlap*100)/100 == 1.0:
        print("With {} steps, the network approximatively converged to the stored pattern.".format(nr_steps))
    else:
        print("With {} steps, the network did not converge to the stored pattern.".format(nr_steps))
    print("The overlap is {}".format(overlap))
    return overlap

def flip_and_iterate(hopfield_net, factory, pattern_list, nr_of_flips, nr_steps):
    noisy_init_state = pattern_tools.flip_n(pattern_list[0], nr_of_flips=nr_of_flips)
    hopfield_net.set_state_from_pattern(noisy_init_state)
    states = hopfield_net.run_with_monitoring(nr_steps=nr_steps)
    states_as_patterns = factory.reshape_patterns(states)
    return noisy_init_state, states, states_as_patterns

def hamming_distance(pattern1, pattern2, N):
    return (N-np.dot(pattern1.copy().flatten(), pattern2.copy().flatten()))/(2*N)

def compute_hamming_distances(states_as_patterns, pattern_list, M, T, N):
    hamming_distances = []
    for mu in range(M):
        hamming_distance_list = []
        for t in range(T + 1):
            hamming_distance_list.append(hamming_distance(states_as_patterns[t], pattern_list[mu], N))
        hamming_distances.append(hamming_distance_list)
    return hamming_distances

def plot_hamming_distances(hamming_distances, M, T):
    plt.figure()
    for mu in range(M):
        plt.plot(np.arange(T + 1), hamming_distances[mu], label="Pattern {}".format(mu))
    plt.xlabel("Time step")
    plt.ylabel("Hamming distance")
    plt.legend()
    plt.show()

def study_retrieval(hamming_distances, M, c_f, init_id, silent=False):
    hamming_distances_f = []
    if not silent:
        print("The pattern used to initialise the first state S(t=0) is P{}.".format(init_id))
    for hamming_list in hamming_distances:
        hamming_distances_f.append(hamming_list[-1])
    # The error retrieval needs to be computed with S(t=T) initialised with P_mu for all mu.
    # error_retrieval = 1/M * np.sum(hamming_distances_f, axis=0)
    retrieved_patterns = []
    for mu in range(M):
        if hamming_distances_f[mu] <= c_f:
            if not silent:
                print("The network retrieved the pattern P{}.".format(mu))
            retrieved_patterns.append(mu)
    # if not silent:
        # print("The number of retrieved patterns is {}.".format(len(retrieved_patterns)))
        # print("The error retrieval is {}.".format(error_retrieval))
    return retrieved_patterns

def study_simple_retrieval(state_f_as_pattern, init_pattern, init_id, N, c_f, silent=False):
    hamming_dist = hamming_distance(np.array(state_f_as_pattern), init_pattern, N)
    if hamming_dist <= c_f:
        if not silent:
            print("The network correctly retrieved the initial pattern P{}.".format(init_id))
            print("The hamming distance is {}.".format(hamming_dist))
        return hamming_dist, init_id
    else:
        return hamming_dist, None

def custom_iterate(initial_state, var_list, function_name, beta, N):
    """Executes one timestep of the dynamics using weights OR patterns."""
    custom_f = custom_function(function_name, beta, N)
    next_state = custom_f(initial_state, var_list)
    return next_state

def custom_run(state, var_list, function_name, beta, N, nr_steps=5):
    """Runs the dynamics.using the custom iterate function

    Args:
        nr_steps (float, optional): Timesteps to simulate
    """
    for i in range(nr_steps):
        # run a step
        state = custom_iterate(state, var_list, function_name, beta, N)
    return state

def custom_run_with_monitoring(state, var_list, function_name, beta, N, nr_steps=5):
    """
    Iterates at most nr_steps steps. records the network state after every
    iteration

    Args:
        nr_steps:

    Returns:
        a list of 2d network states
    """
    states = []
    states.append(state.copy())
    for i in range(nr_steps):
        # run a step
        state = custom_iterate(state, var_list, function_name, beta, N)
        states.append(state.copy())
    return states

def custom_flip_and_iterate(factory, beta, N, nr_of_flips, nr_steps, pattern_list, init_pattern=0, only_last_state=False, function_name="phi_opti", weights=None):
    noisy_init_pattern = pattern_tools.flip_n(pattern_list[init_pattern], nr_of_flips=nr_of_flips)
    noisy_init_state = noisy_init_pattern.copy().flatten()
    if only_last_state:
        if function_name == "phi_opti":
            state = custom_run(noisy_init_state, pattern_list, function_name, beta, N, nr_steps=nr_steps)
        else:
            state = custom_run(noisy_init_state, weights, function_name, beta, N, nr_steps=nr_steps)
        state_as_pattern = factory.reshape_patterns([state])
        return noisy_init_pattern, state, state_as_pattern
    else:
        if function_name == "phi_opti":
            states = custom_run_with_monitoring(noisy_init_state, pattern_list, function_name, beta, N, nr_steps=nr_steps)
        else:
            states = custom_run_with_monitoring(noisy_init_state, weights, function_name, beta, N, nr_steps=nr_steps)
        states_as_patterns = factory.reshape_patterns(states)
        return noisy_init_pattern, states, states_as_patterns

def all_same_pattern(pattern_list, M):
    for i in range(0, M):
        pattern_list[i] = pattern_list[0]
    return pattern_list