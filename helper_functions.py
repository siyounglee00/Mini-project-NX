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


def generate_random_patterns_low_activity(M, N, a, b):
    shape = squarest_pattern(N)[::-1] # [::-1] is used to reverse the tuple and take the shape as (height, width)
    hopfield_net = network.HopfieldNetwork(nr_neurons=N)
    # factory = pattern_tools.PatternFactory(width, height)
    # pattern_list = factory.create_random_pattern_list(nr_patterns=M, on_probability=a)
    pattern_list = custom_create_random_pattern_list(shape, M, on_probability=a, p_min=0, p_max=1)
    hopfield_net.weights = store_patterns_low_activity(hopfield_net, pattern_list, a, b)
    return hopfield_net, pattern_list, shape

def compute_overlap_low(pattern1, pattern2, a):
    '''Compute the overlap between two patterns
    Args:
        pattern1: numpy.ndarray
        pattern2: numpy.ndarray
        a: float
        c: float, optional'''
    c = 2 / (a * (1 - a))  # Compute the scaling factor
    shape1 = pattern1.shape
    if shape1 != pattern2.shape:
        raise ValueError("patterns are not of equal shape")
    dot_prod = np.dot((pattern1.flatten() - a), pattern2.flatten())  # Compute dot product with activity adjustment
    return float(c * dot_prod) / np.prod(shape1)  # Normalize and return the overlap    

def compute_overlap_list_low(reference_pattern, pattern_list, a):
    """
    Computes the overlap between the reference_pattern and each pattern
    in pattern_list

    Args:
        reference_pattern:
        pattern_list: list of patterns

    Returns:
        A list of the same length as pattern_list
    """
    overlap = np.zeros(len(pattern_list))
    for i in range(0, len(pattern_list)):
        overlap[i] = compute_overlap_low(reference_pattern, pattern_list[i], a)
    return overlap

def study_overlap_low_activity(states_as_patterns, pattern_list, nr_steps, a):
    overlap = compute_overlap_low(states_as_patterns[nr_steps], pattern_list[0], a)
    if overlap == 1:
        print("With {} steps, the network converged to the stored pattern.".format(nr_steps))
    elif np.round(overlap*100)/100 == 1.0:
        print("With {} steps, the network approximatively converged to the stored pattern.".format(nr_steps))
    else:
        print("With {} steps, the network did not converge to the stored pattern.".format(nr_steps))
    print("The overlap is {}".format(overlap))
    return overlap


def _plot_list(axes_list, state_sequence, reference=None, title_pattern="S({0})", color_map="brg"):
    """
    For internal use.
    Plots all states S(t) or patterns P in state_sequence.
    If a (optional) reference pattern is provided, the patters are  plotted with differences highlighted

    Args:
        state_sequence: (list(numpy.ndarray))
        reference: (numpy.ndarray)
        title_pattern (str) pattern injecting index i
    """
    for i in range(len(state_sequence)):
        normalized_state = state_sequence[i]*2-np.ones_like(state_sequence[i])
        normalized_reference = reference*2-np.ones_like(reference)
        if reference is None:
            p = normalized_state
        else:
            p = pattern_tools.get_pattern_diff(normalized_state, normalized_reference, diff_code=-0.2)
        if np.max(p) == np.min(p):
            axes_list[i].imshow(p, interpolation="nearest", cmap='RdYlBu')
        else:
            axes_list[i].imshow(p, interpolation="nearest", cmap=color_map)
        axes_list[i].set_title(title_pattern.format(i))
        axes_list[i].axis("off")

def plot_state_sequence_and_overlap_low(state_sequence, pattern_list, a, reference_idx, color_map="brg", suptitle=None):
    """
    For each time point t ( = index of state_sequence), plots the sequence of states and the overlap (barplot)
    between state(t) and each pattern.

    Args:
        state_sequence: (list(numpy.ndarray))
        pattern_list: (list(numpy.ndarray))
        reference_idx: (int) identifies the pattern in pattern_list for which wrong pixels are colored.
    """
    if reference_idx is None:
        reference_idx = 0
    reference = pattern_list[reference_idx]
    f, ax = plt.subplots(2, len(state_sequence))
    if len(state_sequence) == 1:
        ax = [ax]
    print()
    _plot_list(ax[0, :], state_sequence, reference, "S{0}", color_map) # Multiply by 2 and subtract 1 to map {0, 1} to {-1, 1}
    for i in range(len(state_sequence)):
        overlap_list = compute_overlap_list_low(state_sequence[i], pattern_list, a)
        print(overlap_list) # To delete
        ax[1, i].bar(range(len(overlap_list)), overlap_list)
        ax[1, i].set_title("m = {1}".format(i, round(overlap_list[reference_idx], 2)))
        ax[1, i].set_ylim([-4, 4]) # Set manually to min(mu) and max(mu)
        ax[1, i].get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        if i > 0:  # show lables only for the first subplot
            ax[1, i].set_xticklabels([])
            ax[1, i].set_yticklabels([])
    if suptitle is not None:
        f.suptitle(suptitle)
    plt.show()

def custom_function_low(function_name, beta, teta, a, N):
    c = 2 / (a * (1 - a))
    if function_name[:4] == "phi":
        def custom_f(sigma_s0, weights):
            # Code to see if everything works with the model from the previous exercise:
            # ---------------------------------------------------------------------------------------
            # state_old = [(2*(sigma_s0_j - 1)) for sigma_s0_j in sigma_s0]
            # h_old = np.sum(weights * state_old, axis=1)
            # state_s1_old = np.tanh(beta * h_old)
            h_respect_formula = np.sum((2*weights) * sigma_s0, axis=1) - teta
            state_s1_old = np.tanh(beta * h_respect_formula)
            # sigma_s1_old = [(0.5*(state_s1_old_j+1)) for state_s1_old_j in state_s1_old]
            # return np.array(sigma_s1_old)
            sigma_s1 = [np.random.binomial(1, 0.5*(state_s1_old_j+1)) for state_s1_old_j in state_s1_old] # Compute sigma
            return np.array(sigma_s1)
            # ---------------------------------------------------------------------------------------
            # h = (c/N) * np.sum((weights * sigma_s0) - teta, axis=1)
            # state_s1 = np.tanh(beta * h)
            # print(state_s1)
            # sigma_s1 = [np.random.binomial(1, 0.5*(state_s1_j+1)) for state_s1_j in state_s1] # Compute sigma
            # print(sigma_s1)
            # return np.array(sigma_s1)
    elif function_name == "phi_opti":
        def custom_f(sigma_s0, pattern_list):
            m_list = []
            flattened_pattern_list = np.array([pattern.flatten() for pattern in pattern_list])
            for pattern in flattened_pattern_list:
                m_list.append((c/N) * np.sum(np.dot(pattern - a, sigma_s0)))
            h = np.sum(flattened_pattern_list * np.array(m_list)[:, None], axis=0)
            state_s1 = np.tanh(beta * h)
            sigma_s1 = [np.random.binomial(1, 0.5*(state_s1_j+1)) for state_s1_j in state_s1] # Compute sigma
            return np.array(sigma_s1)
    else:
        raise ValueError("The function must be 'phi' or 'phi_opti'.")
    return custom_f

def custom_iterate_low(initial_state, var_list, function_name, beta, teta, a, N):
    """Executes one timestep of the dynamics using weights OR patterns."""
    custom_f = custom_function_low(function_name, beta, teta, a, N)
    next_state = custom_f(initial_state, var_list)
    return next_state

def custom_run_low(state, var_list, function_name, beta, teta, a, N, nr_steps=5):
    """Runs the dynamics.using the custom iterate function

    Args:
        nr_steps (float, optional): Timesteps to simulate
    """
    for i in range(nr_steps):
        # run a step
        state = custom_iterate_low(state, var_list, function_name, beta, teta, a, N)
    return state

def custom_run_with_monitoring_low(state, var_list, function_name, beta, teta, a, N, nr_steps=5):
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
    for _ in range(nr_steps):
        # run a step
        state = custom_iterate_low(state, var_list, function_name, beta, teta, a, N)
        states.append(state.copy())
    return states


def custom_flip_and_iterate_low(shape, beta, teta, a, N, nr_of_flips, nr_steps, pattern_list, init_pattern=0, only_last_state=False, function_name="phi_opti", weights=None):
    noisy_init_pattern = custom_flip_n_low(pattern_list[init_pattern], nr_of_flips, 0, 1)
    noisy_init_state = noisy_init_pattern.copy().flatten()
    if only_last_state:
        if function_name == "phi_opti":
            state = custom_run_low(noisy_init_state, pattern_list, function_name, beta, teta, a, N, nr_steps=nr_steps)
        else:
            state = custom_run_low(noisy_init_state, weights, function_name, beta, teta, a, N, nr_steps=nr_steps)
        state_as_pattern = state.reshape(shape)
        return noisy_init_pattern, state, state_as_pattern
    else:
        if function_name == "phi_opti":
            states = custom_run_with_monitoring_low(noisy_init_state, pattern_list, function_name, beta, teta, a, N, nr_steps=nr_steps)
        else:
            states = custom_run_with_monitoring_low(noisy_init_state, weights, function_name, beta, teta, a, N, nr_steps=nr_steps)
        states_as_patterns = [s.reshape(shape) for s in states]
        return noisy_init_pattern, states, states_as_patterns
    
def custom_flip_n_low(template, nr_of_flips, p_min=0, p_max=1):
    """
    makes a copy of the template pattern and flips
    exactly n randomly selected states.
    Args:
        template:
        nr_of_flips:
    Returns:
        a new pattern
    """
    n = np.prod(template.shape)
    # pick nrOfMutations indices (without replacement)
    idx_reassignment = np.random.choice(n, nr_of_flips, replace=False)
    linear_template = template.flatten()
    for id in idx_reassignment:
        linear_template[id] = p_min if (linear_template[id] == p_max) else p_max
    return linear_template.reshape(template.shape)
    

def standard_teta(weights):
    teta = np.sum(weights, axis = 1)
    return teta


def store_patterns_low_activity(hopfield_net, pattern_list, a, b):
    """
    Learns the patterns by updating the network weights with low activity adaptation.

    Args:
        hopfield_net: Hopfield network object.
        pattern_list: A nonempty list of patterns.
        a: Float, activity level.
    """
    c = 2 / (a * (1 - a))

    # Check if all patterns have the same number of states as the network neurons
    all_same_size_as_net = all(len(p.flatten()) == hopfield_net.nrOfNeurons for p in pattern_list)
    if not all_same_size_as_net:
        errMsg = "Not all patterns in pattern_list have exactly the same number of states " \
                 "as this network has neurons n = {0}.".format(hopfield_net.nrOfNeurons)
        raise ValueError(errMsg)
    
    # Update the weights using low activity adaptation
    for p in pattern_list:
        p_flat = p.flatten()
        for i in range(hopfield_net.nrOfNeurons):
            for k in range(hopfield_net.nrOfNeurons):
                hopfield_net.weights[i, k] += (c / hopfield_net.nrOfNeurons) * (p_flat[i] - b) * (p_flat[k] - a)
    
    # Normalize the weights
    # hopfield_net.weights *= c / hopfield_net.nrOfNeurons

    # No self connections
    np.fill_diagonal(hopfield_net.weights, 0)
    return hopfield_net.weights

def custom_create_random_pattern_list(shape, nr_patterns, on_probability=0.5, p_min=-1, p_max=1):
    """
    Creates a list of nr_patterns random patterns
    Args:
        nr_patterns: length of the new list
        on_probability:

    Returns:
        a list of new random patterns of size (pattern_length x pattern_width)
    """
    p_list = []
    for _ in range(nr_patterns):
        p = np.random.binomial(1, on_probability, np.prod(shape))
        p = p * (p_max-p_min) + p_min  # map {0, 1} to {p_min, p_max}
        p_list.append(p.reshape(shape))
    return p_list