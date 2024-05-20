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
    c = 2 / (a * (1 - a))
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
    c = 2 / (a * (1 - a))
    if function_name == "sync":
        def custom_f(init_sigmas, pattern_list):
            init_sigmas_inhib = init_sigmas[:N_I]
            init_sigmas_exhib = init_sigmas[N_I:]
            K_indexes = np.random.choice(range(N_I, N), K, replace=False)
            m_list = []
            mean_inhib_a = 0
            h_inhib = 0
            flat_p_list_exhib = np.array([pattern.flatten()[N_I:] for pattern in pattern_list])
            for j in range(N_I):
                mean_inhib_a += init_sigmas_inhib[j] / N_I
            for pattern in flat_p_list_exhib:
                m_list.append((c/N) * np.sum(np.dot(pattern, init_sigmas_exhib)))
            for i in K_indexes:
                h_inhib += init_sigmas_exhib[i-N_I] / K

            h_exhib = np.sum(flat_p_list_exhib * (np.array(m_list)[:, None] - c * a * mean_inhib_a), axis=0)
            state_s1 = np.tanh(beta * (h_exhib - theta))
            next_sigma_exhib = [np.random.binomial(1, 0.5*(state_s1_j+1)) for state_s1_j in state_s1] # Compute sigma
            next_sigma_inhib = [np.random.binomial(1, h_inhib) for _ in range(N_I)]
            return np.array(next_sigma_inhib + next_sigma_exhib)
    elif function_name == "seq":
        def custom_f(init_sigmas, pattern_list):
            init_sigmas_inhib = init_sigmas[:N_I]
            init_sigmas_exhib = init_sigmas[N_I:]
            K_indexes = np.random.choice(range(N_I, N), K, replace=False)
            m_list = []
            mean_inhib_a = 0
            h_inhib = 0
            flat_p_list_exhib = np.array([pattern.flatten()[N_I:] for pattern in pattern_list])
            for i in K_indexes:
                h_inhib += init_sigmas_exhib[i-N_I] / K
            next_sigma_inhib = [np.random.binomial(1, h_inhib) for _ in range(N_I)]
            for j in range(N_I):
                mean_inhib_a += next_sigma_inhib[j] / N_I
            for pattern in flat_p_list_exhib:
                m_list.append((c/N) * np.sum(np.dot(pattern, init_sigmas_exhib)))

            h_exhib = np.sum(flat_p_list_exhib * (np.array(m_list)[:, None] - c * a * mean_inhib_a), axis=0)
            state_s1 = np.tanh(beta * (h_exhib - theta))
            next_sigma_exhib = [np.random.binomial(1, 0.5*(state_s1_j+1)) for state_s1_j in state_s1] # Compute sigma
            return np.array(next_sigma_inhib + next_sigma_exhib)
    return custom_f

# beta, theta, a, b, N, nr_flips, nr_steps
def flip_and_iterate(cst, shape, pattern_list, init_pattern=0, only_last_state=False, function_name="sync"):
    noisy_init_pattern = hf.custom_flip_n_low(pattern_list[init_pattern], cst["nr_flips"], 0, 1)
    noisy_init_sigmas = noisy_init_pattern.copy().flatten() 
    if only_last_state:
        sigmas = run(noisy_init_sigmas, pattern_list, function_name, cst["beta"], cst["theta"], cst["a"], cst["K"], cst["N"], cst["N_I"], nr_steps=cst["T"])
        sigmas_as_pattern = sigmas.reshape(shape)
        return noisy_init_pattern, sigmas, sigmas_as_pattern
    else:
        sigmas_list = run_with_monitoring(noisy_init_sigmas, pattern_list, function_name, cst["beta"], cst["theta"], cst["a"], cst["K"], cst["N"], cst["N_I"], nr_steps=cst["T"])
        sigmas_list_as_patterns = [sigmas.reshape(shape) for sigmas in sigmas_list]
        return noisy_init_pattern, sigmas_list, sigmas_list_as_patterns
    
def iterate(init_sigmas, pat_list, function_name, beta, theta, a, K, N, N_I):
    """Executes one timestep of the dynamics using weights OR patterns."""
    custom_f = custom_function(beta, theta, a, K, N, N_I, function_name)
    next_sigmas = custom_f(init_sigmas, pat_list)
    return next_sigmas
    
def run(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, nr_steps=5):
    """Runs the dynamics.using the custom iterate function

    Args:
        nr_steps (float, optional): Timesteps to simulate
    """
    for i in range(nr_steps):
        # run a step
        sigmas = iterate(sigmas, var_list, function_name, beta, theta, a, K, N, N_I)
    return sigmas

def run_with_monitoring(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, nr_steps=5):
    """
    Iterates at most nr_steps steps. records the network state after every
    iteration

    Args:
        nr_steps:

    Returns:
        a list of 2d network states
    """
    sigmas_list = []
    sigmas_list.append(sigmas.copy())
    for _ in range(nr_steps):
        # run a step
        sigmas = iterate(sigmas, var_list, function_name, beta, theta, a, K, N, N_I)
        sigmas_list.append(sigmas.copy())
    return sigmas_list

def plot_state_sequence_and_overlap(sigmas_sequence, pattern_list, reference_idx=0, color_map="brg", suptitle=None, overlap_from=0):
    """
    For each time point t ( = index of state_sequence), plots the sequence of states and the overlap (barplot)
    between state(t) and each pattern.

    Args:
        state_sequence: (list(numpy.ndarray))
        pattern_list: (list(numpy.ndarray))
        reference_idx: (int) identifies the pattern in pattern_list for which wrong pixels are colored.
    """
    reference = pattern_list[reference_idx]
    f, ax = plt.subplots(2, len(sigmas_sequence))
    if len(sigmas_sequence) == 1:
        ax = [ax]
    print()
    hf._plot_list(ax[0, :], sigmas_sequence, reference, "S{0}", color_map) # Multiply by 2 and subtract 1 to map {0, 1} to {-1, 1}
    for i in range(len(sigmas_sequence)):
        overlap_list = compute_overlap_list(sigmas_sequence[i], pattern_list, only_from=overlap_from)
        print("Sigmas : ", sigmas_sequence[i])
        print("Pattern list: ", pattern_list)
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

def compute_overlap_list(reference_sigmas, pattern_list, only_from=0):
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
        overlap[i] = compute_overlap(reference_sigmas, pattern_list[i], only_from)
    return overlap

def compute_overlap(sigmas, pattern, only_from=0):
    '''Compute the overlap between two patterns
    Args:
        pattern1: numpy.ndarray
        pattern2: numpy.ndarray
        a: float
    '''
    if sigmas.shape != pattern.shape:
        raise ValueError("state and pattern are not of equal shape")
    norm_sigmas = sigmas.flatten()[only_from:] * 2 - 1  # Normalize sigmas to {-1, 1}
    norm_pattern = pattern.flatten()[only_from:] * 2 - 1  # Normalize pattern to {-1, 1}
    # print("Sigmas: ", norm_sigmas)
    # print("Pattern normalized: ", norm_pattern)
    dot_prod = np.dot(norm_sigmas, norm_pattern)  # Compute dot product with activity adjustment
    return float(dot_prod) / (np.prod(pattern.shape) - only_from)  # Normalize and return the overlap    