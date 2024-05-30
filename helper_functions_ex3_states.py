from neurodynex3.hopfield_network import network, pattern_tools, plot_tools
from matplotlib import pyplot as plt
import numpy as np
import helper_functions as hf

def store_patterns(pattern_list, a, N, N_I, K):
    """
    Learns the patterns by updating the network weights

    Args:
        pattern_list: A nonempty list of patterns.
        a: Float, activity level.
        N: Int, total number of neurons
        N_I: Int, number of inhibitory neurons
        K: Int, number of excitatory neurons from which each inhibitory neuron receives an input
    """
    N_E = N - N_I
    c = 2 / (a * a * (1 - a))
    W_EE = np.zeros((N_E, N_E))
    W_EI = np.zeros((N_E, N_I))

    # Check if all patterns have the same number of states as the network has neurons
    all_same_size_as_net = all(len(p.flatten()) == N for p in pattern_list)
    if not all_same_size_as_net:
        errMsg = "Not all patterns in pattern_list have exactly the same number of states " \
                 "as this network has neurons n = {0}.".format(N)
        raise ValueError(errMsg)
    
    # Update the weights using low activity adaptation
    for p in pattern_list:
        p_flat = p.flatten()
        for i in range(N_I, N):
            for j in range(N):
                if j >= N_I:
                    W_EE[i-N_I, j-N_I] += (c / N) * p_flat[i] * p_flat[j]
                if j < N_I:
                    W_EI[i-N_I, j] += ((c * a) / N_I) * p_flat[i]
    
    W_IE = np.ones((N_I, N_E)) * (1/K)
    
    # No self connections in W_EE
    np.fill_diagonal(W_EE, 0)

    weights = {
        "W_EE": W_EE,
        "W_EI": W_EI,
        "W_IE": W_IE
    }

    return weights

def generate_random_patterns(cst):
    shape = hf.squarest_pattern(cst["N"])[::-1] # [::-1] is used to reverse the tuple and take the shape as (height, width)
    pattern_list = hf.custom_create_random_pattern_list(shape, cst["M"], on_probability=cst["a"], p_min=0, p_max=1)
    weights = store_patterns(pattern_list, cst["a"], cst["N"], cst["N_I"], cst["K"])
    return weights, pattern_list, shape

def custom_function(beta, theta, a, K, N, N_I, function_name="sync"):
    c = 2 / (a * a * (1 - a))
    if function_name == "sync":
        def custom_f(init_state, pattern_list, h_inhib):
            sigmas_exhit = [np.random.binomial(1, 0.5*(init_state_i+1)) for init_state_i in init_state] # Compute sigma
            if h_inhib is None:
                sigmas_inhib = np.random.choice([0, 1], size=N_I, p=[1-a, a])
            else:
                sigmas_inhib = [np.random.binomial(1, h_inhib) for _ in range(N_I)]
            K_indexes = np.random.choice(range(N_I, N), K, replace=False)
            m_list = []
            mean_inhib_a = 0
            h_inhib = 0
            flat_p_list_exhit = np.array([pattern.flatten() for pattern in pattern_list])
            for j in range(N_I):
                mean_inhib_a += sigmas_inhib[j] / N_I
            for pattern in flat_p_list_exhit:
                m_list.append((c/N) * np.sum(np.dot(pattern, sigmas_exhit)))
            for i in K_indexes:
                h_inhib += sigmas_exhit[i-N_I] / K
            h_exhit = np.sum(flat_p_list_exhit * (np.array(m_list)[:, None] - c * a * mean_inhib_a), axis=0)
            print("beta: ", beta)
            print("h_exhit: ", h_exhit)
            print("theta: ", theta)
            next_state = np.tanh(beta * (h_exhit - theta))
            return np.array(next_state), h_inhib
    elif function_name == "seq":
        def custom_f(init_state, pattern_list):
            sigmas_exhit = [np.random.binomial(1, 0.5*(init_state_i+1)) for init_state_i in init_state] # Compute sigma
            K_indexes = np.random.choice(range(N_I, N), K, replace=False)
            m_list = []
            mean_inhib_a = 0
            h_inhib = 0
            flat_p_list_exhit = np.array([pattern.flatten() for pattern in pattern_list])
            for i in K_indexes:
                h_inhib += sigmas_exhit[i-N_I] / K
            sigmas_inhib = [np.random.binomial(1, h_inhib) for _ in range(N_I)]
            for j in range(N_I):
                mean_inhib_a += sigmas_inhib[j] / N_I
            for pattern in flat_p_list_exhit:
                m_list.append((c/N) * np.sum(np.dot(pattern, sigmas_exhit)))
            h_exhit = np.sum(flat_p_list_exhit * (np.array(m_list)[:, None] - c * a * mean_inhib_a), axis=0)
            next_state = np.tanh(beta * (h_exhit - theta))
            return np.array(next_state)
    return custom_f

# beta, theta, a, b, N, nr_flips, nr_steps
def flip_and_iterate(cst, shape, pattern_list, init_pattern=0, only_last_state=False, function_name="sync"):
    noisy_init_pattern = hf.custom_flip_n_low(pattern_list[init_pattern], cst["nr_flips"], 0, 1)
    noisy_init_state = noisy_init_pattern.copy().flatten() * 2 - 1
    if only_last_state:
        state = run(noisy_init_state, pattern_list, function_name, cst["beta"], cst["theta"], cst["a"], cst["K"], cst["N"], cst["N_I"], nr_steps=cst["T"])
        state_as_pattern = state.reshape(shape)
        return noisy_init_pattern, state, state_as_pattern
    else:
        states = run_with_monitoring(noisy_init_state, pattern_list, function_name, cst["beta"], cst["theta"], cst["a"], cst["K"], cst["N"], cst["N_I"], nr_steps=cst["T"])
        states_as_patterns = [state.reshape(shape) for state in states]
        return noisy_init_pattern, states, states_as_patterns
    
def iterate(init_sigmas, pat_list, function_name, beta, theta, a, K, N, N_I, h_inhib=None):
    """Executes one timestep of the dynamics using weights OR patterns."""
    custom_f = custom_function(beta, theta, a, K, N, N_I, function_name)
    if function_name == "sync":
        next_sigmas, next_h_inhib = custom_f(init_sigmas, pat_list, h_inhib)
        return next_sigmas, next_h_inhib
    else:
        next_sigmas = custom_f(init_sigmas, pat_list)
        return next_sigmas
    
def run(state, pat_list, function_name, beta, theta, a, K, N, N_I, nr_steps=5):
    """Runs the dynamics.using the custom iterate function

    Args:
        nr_steps (float, optional): Timesteps to simulate
    """
    for i in range(nr_steps):
        # run a step
        if function_name == "sync":
            if i == 0:
                h_inhib = None
            state, h_inhib = iterate(state, pat_list, function_name, beta, theta, a, K, N, N_I, h_inhib)
        else:
            state = iterate(state, pat_list, function_name, beta, theta, a, K, N, N_I)
    return state

def run_with_monitoring(state, pat_list, function_name, beta, theta, a, K, N, N_I, nr_steps=5):
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
        if function_name == "sync":
            if i == 0:
                h_inhib = None
            state, h_inhib = iterate(state, pat_list, function_name, beta, theta, a, K, N, N_I, h_inhib)
        else:
            state = iterate(state, pat_list, function_name, beta, theta, a, K, N, N_I)
        states.append(state.copy())
    return states

def plot_state_sequence_and_overlap(state_sequence, pattern_list, reference_idx=0, color_map="brg", suptitle=None):
    """
    For each time point t ( = index of state_sequence), plots the sequence of states and the overlap (barplot)
    between state(t) and each pattern.

    Args:
        state_sequence: (list(numpy.ndarray))
        pattern_list: (list(numpy.ndarray))
        reference_idx: (int) identifies the pattern in pattern_list for which wrong pixels are colored.
    """
    reference = pattern_list[reference_idx]
    f, ax = plt.subplots(2, len(state_sequence))
    if len(state_sequence) == 1:
        ax = [ax]
    print()
    hf._plot_list(ax[0, :], state_sequence, reference, "S{0}", color_map) # Multiply by 2 and subtract 1 to map {0, 1} to {-1, 1}
    for i in range(len(state_sequence)):
        overlap_list = compute_overlap_list(state_sequence[i], pattern_list)
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

def compute_overlap_list(reference_state, pattern_list):
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
    for i in range(len(pattern_list)):
        overlap[i] = compute_overlap(reference_state, pattern_list[i])
    return overlap

def compute_overlap(state, pattern):
    '''Compute the overlap between two patterns
    Args:
        pattern1: numpy.ndarray
        pattern2: numpy.ndarray
        a: float
    '''
    if state.shape != pattern.shape:
        raise ValueError("state and pattern are not of equal shape")
    norm_pattern = pattern * 2 - np.ones_like(pattern) # normalized pattern
    print("State: ", state.copy().flatten())
    print("Pattern normalized: ", norm_pattern.copy().flatten())
    dot_prod = np.dot(state.copy().flatten(), norm_pattern.copy().flatten())  # Compute dot product with activity adjustment
    return float(dot_prod) / np.prod(pattern.shape)  # Normalize and return the overlap    