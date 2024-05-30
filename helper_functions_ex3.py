from neurodynex3.hopfield_network import pattern_tools
from matplotlib import pyplot as plt
import numpy as np
import helper_functions as hf
import math

def round_half_up(n):
    """Round a number to the nearest integer. x.5 is always rounded up to x+1."""
    if n - math.floor(n) == 0.5:
        return math.ceil(n)
    else:
        return round(n)

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
        def custom_f(init_sigmas, pattern_list, ext_p=None):
            init_sigmas_inhib = init_sigmas[:N_I]
            init_sigmas_excit = init_sigmas[N_I:]
            m_list = []
            h_inhib = []
            flat_p_list_excit = np.array([pattern.flatten()[N_I:] for pattern in pattern_list])
            for pattern in flat_p_list_excit:
                m_list.append((c/(N-N_I)) * np.sum(np.dot(pattern, init_sigmas_excit)))
            for _ in range(N_I):
                K_indexes = np.random.choice(range(N_I, N), K, replace=False)
                h_inhib_k = 0
                for K_index in K_indexes:
                    h_inhib_k += init_sigmas_excit[K_index-N_I] / K
                h_inhib.append(h_inhib_k)

            h_excit = np.sum(flat_p_list_excit * (np.array(m_list)[:, None] - c * a * np.mean(init_sigmas_inhib)), axis=0)
            state_s1 = np.tanh(beta * (h_excit - theta))
            next_sigmas_excit = [np.random.binomial(1, 0.5*(state_s1_j+1)) for state_s1_j in state_s1] # Compute sigma
            next_sigmas_inhib = [np.random.binomial(1, h_inhib_k) if (0 <= h_inhib_k <= 1) and not np.isnan(h_inhib_k) else 0 for h_inhib_k in h_inhib]
            return np.array(next_sigmas_inhib + next_sigmas_excit)
    elif function_name == "seq":
        def custom_f(init_sigmas, pattern_list, ext_p=None):
            init_sigmas_inhib = init_sigmas[:N_I]
            init_sigmas_excit = init_sigmas[N_I:]
            m_list = []
            h_inhib = []
            flat_p_list_excit = np.array([pattern.flatten()[N_I:] for pattern in pattern_list])
            for _ in range(N_I):
                K_indexes = np.random.choice(range(N_I, N), K, replace=False)
                h_inhib_k = 0
                for K_index in K_indexes:
                    h_inhib_k += init_sigmas_excit[K_index-N_I] / K
                h_inhib.append(h_inhib_k)
            next_sigmas_inhib = [np.random.binomial(1, h_inhib_k) if (0 <= h_inhib_k <= 1) and not np.isnan(h_inhib_k) else 0 for h_inhib_k in h_inhib]
            for pattern in flat_p_list_excit:
                m_list.append((c/(N-N_I)) * np.sum(np.dot(pattern, init_sigmas_excit)))

            h_excit = np.sum(flat_p_list_excit * (np.array(m_list)[:, None] - c * a * np.mean(next_sigmas_inhib)), axis=0)
            state_s1 = np.tanh(beta * (h_excit - theta))
            next_sigmas_excit = [np.random.binomial(1, 0.5*(state_s1_j+1)) for state_s1_j in state_s1] # Compute sigma
            return np.array(next_sigmas_inhib + next_sigmas_excit)
    elif function_name == "sync_2inhib":
        def custom_f(init_sigmas, pattern_list, ext_p=None):
            # Separate N_I1 and N_I2 inhibitory neurons:
            init_sigmas_inhib1 = init_sigmas[:round_half_up(N_I/2)]
            init_sigmas_inhib2 = init_sigmas[round_half_up(N_I/2):N_I]
            init_sigmas_excit = init_sigmas[N_I:]
            h_inhib1, h_inhib2 = [], []
            m_list = []
            flat_p_list_excit = np.array([pattern.flatten()[N_I:] for pattern in pattern_list])
            for _ in range(int(N_I/2)):
                K_indexes1 = np.random.choice(range(N_I, N), K, replace=False)
                K_indexes2 = np.random.choice(range(N_I, N), K, replace=False)
                h_inhib1_k, h_inhib2_k = 0, 0
                for K_index in range(K):
                    h_inhib1_k += init_sigmas_excit[K_indexes1[K_index]-N_I] / K
                    h_inhib2_k += init_sigmas_excit[K_indexes2[K_index]-N_I] / K
                h_inhib1.append(h_inhib1_k)
                h_inhib2.append(h_inhib2_k)
            if int(N_I/2) != round_half_up(N_I/2):
                K_indexes1 = np.random.choice(range(N_I, N), K, replace=False)
                h_inhib1_k = 0
                for K_index1 in K_indexes1:
                    h_inhib1_k += init_sigmas_excit[K_index1-N_I] / K
                h_inhib1.append(h_inhib1_k)
            if ext_p is not None:
                h_extern = (ext_p["J"] * flat_p_list_excit[ext_p["mu"]]).astype(np.float64)
            for pattern in flat_p_list_excit:
                m_list.append((c/(N-N_I)) * np.sum(np.dot(pattern, init_sigmas_excit)))
                if ext_p is not None:
                    h_extern -= ext_p["J"] * pattern / len(flat_p_list_excit)

            h_excit = np.sum(flat_p_list_excit * (np.array(m_list)[:, None] - c * a * np.mean(init_sigmas_inhib1)), axis=0) - c * a * np.mean(init_sigmas_inhib2)
            h_excit = h_excit + h_extern if ext_p is not None else h_excit
            state_s1 = np.tanh(beta * (h_excit - theta))
            next_sigmas_excit = [np.random.binomial(1, 0.5*(state_s1_j+1)) for state_s1_j in state_s1] # Compute sigma
            next_sigmas_inhib1 = [np.random.binomial(1, h_inhib1_k) if (0 <= h_inhib1_k <= 1) and not np.isnan(h_inhib1_k) else 0 for h_inhib1_k in h_inhib1]
            if np.mean(next_sigmas_excit) > a: 
                next_sigmas_inhib2 = [np.random.binomial(1, h_inhib2_k) if (0 <= h_inhib2_k <= 1) and not np.isnan(h_inhib2_k) else 0 for h_inhib2_k in h_inhib2]
            else:
                next_sigmas_inhib2 = list(np.zeros_like(np.array(init_sigmas_inhib2)))
            return np.array(next_sigmas_inhib1 + next_sigmas_inhib2 + next_sigmas_excit)
    elif function_name == "seq_2inhib":
        def custom_f(init_sigmas, pattern_list, ext_p=None):
            init_sigmas_inhib1 = init_sigmas[:round_half_up(N_I/2)]
            init_sigmas_inhib2 = init_sigmas[round_half_up(N_I/2):N_I]
            init_sigmas_excit = init_sigmas[N_I:]
            h_inhib1, h_inhib2 = [], []
            m_list = []
            flat_p_list_excit = np.array([pattern.flatten()[N_I:] for pattern in pattern_list])
            for _ in range(int(N_I/2)):
                K_indexes1 = np.random.choice(range(N_I, N), K, replace=False)
                K_indexes2 = np.random.choice(range(N_I, N), K, replace=False)
                h_inhib1_k, h_inhib2_k = 0, 0
                for K_index in range(K):
                    h_inhib1_k += init_sigmas_excit[K_indexes1[K_index]-N_I] / K
                    h_inhib2_k += init_sigmas_excit[K_indexes2[K_index]-N_I] / K
                h_inhib1.append(h_inhib1_k)
                h_inhib2.append(h_inhib2_k)
            if int(N_I/2) != round_half_up(N_I/2):
                K_indexes1 = np.random.choice(range(N_I, N), K, replace=False)
                h_inhib1_k = 0
                for K_index1 in K_indexes1:
                    h_inhib1_k += init_sigmas_excit[K_index1-N_I] / K
                h_inhib1.append(h_inhib1_k)
            next_sigmas_inhib1 = [np.random.binomial(1, h_inhib1_k) if (0 <= h_inhib1_k <= 1) and not np.isnan(h_inhib1_k) else 0 for h_inhib1_k in h_inhib1]
            if np.mean(init_sigmas_excit) > a:
                next_sigmas_inhib2 = [np.random.binomial(1, h_inhib2_k) if (0 <= h_inhib2_k <= 1) and not np.isnan(h_inhib2_k) else 0 for h_inhib2_k in h_inhib2]
            else:
                next_sigmas_inhib2 = list(np.zeros_like(np.array(init_sigmas_inhib2)))
            if ext_p is not None:
                h_extern = (ext_p["J"] * flat_p_list_excit[ext_p["mu"]]).astype(np.float64)
            for pattern in flat_p_list_excit:
                m_list.append((c/(N-N_I)) * np.sum(np.dot(pattern, init_sigmas_excit)))
                if ext_p is not None:
                    h_extern -= ext_p["J"] * pattern / len(flat_p_list_excit)
            h_excit = np.sum(flat_p_list_excit * (np.array(m_list)[:, None] - c * a * np.mean(next_sigmas_inhib1)), axis=0) - c * a * np.mean(next_sigmas_inhib2)
            h_excit = h_excit + h_extern if ext_p is not None else h_excit
            state_s1 = np.tanh(beta * (h_excit - theta))
            next_sigmas_excit = [np.random.binomial(1, 0.5*(state_s1_j+1)) for state_s1_j in state_s1] # Compute sigma
            return np.array(next_sigmas_inhib1 + next_sigmas_inhib2 + next_sigmas_excit)
    return custom_f

def flip_and_iterate(cst, shape, pattern_list, init_pattern=0, only_last_state=False, function_name="sync", ext_p=None):
    noisy_init_pattern = hf.custom_flip_n_low(pattern_list[init_pattern], cst["nr_flips"], 0, 1)
    noisy_init_sigmas = noisy_init_pattern.copy().flatten() 
    if only_last_state:
        sigmas = run(noisy_init_sigmas, pattern_list, function_name, cst["beta"], cst["theta"], cst["a"], cst["K"], cst["N"], cst["N_I"], nr_steps=cst["T"], ext_p=ext_p)
        sigmas_as_pattern = sigmas.reshape(shape)
        return noisy_init_pattern, sigmas, sigmas_as_pattern
    else:
        if ext_p is not None:
            full_sigmas_list, sigmas_list, sigmas_list_ids, patterns_presented = run_with_monitoring(
                noisy_init_sigmas, pattern_list, function_name, cst["beta"], cst["theta"], cst["a"], cst["K"], cst["N"], cst["N_I"], nr_steps=cst["T"], ext_p=ext_p)
            sigmas_list_as_patterns = [sigmas.reshape(shape) for sigmas in sigmas_list]
            full_sigmas_list_as_patterns = [sigmas.reshape(shape) for sigmas in full_sigmas_list]
            return noisy_init_pattern, sigmas_list, sigmas_list_as_patterns, sigmas_list_ids, full_sigmas_list_as_patterns, patterns_presented
        else:
            sigmas_list = run_with_monitoring(
                noisy_init_sigmas, pattern_list, function_name, cst["beta"], cst["theta"], cst["a"], cst["K"], cst["N"], cst["N_I"], nr_steps=cst["T"])
            sigmas_list_as_patterns = [sigmas.reshape(shape) for sigmas in sigmas_list]
            return noisy_init_pattern, sigmas_list, sigmas_list_as_patterns
    
def iterate(init_sigmas, pat_list, function_name, beta, theta, a, K, N, N_I, ext_p=None):
    """Executes one timestep of the dynamics using weights OR patterns."""
    custom_f = custom_function(beta, theta, a, K, N, N_I, function_name)
    next_sigmas = custom_f(init_sigmas, pat_list, ext_p=ext_p)
    return next_sigmas
    
def run(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, nr_steps=5, ext_p=None):
    """Runs the dynamics.using the custom iterate function

    Args:
        nr_steps (float, optional): Timesteps to simulate
    """
    for i in range(nr_steps):
        # run a step
        if ext_p is not None:
            step_in_loop = (i - ext_p["init_steps"]) % (ext_p["feed_steps"] + ext_p["evolve_steps"])
            if i < ext_p["init_steps"] or (ext_p["sequence_length"] >= 0 and step_in_loop >= ext_p["feed_steps"]):
                sigmas = iterate(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, ext_p=None)
            else:
                if step_in_loop == 0:
                    ext_p["sequence_length"] -= 1
                    ext_p["mu"] = np.random.choice(range(len(var_list)))
                sigmas = iterate(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, ext_p)
        else:
            sigmas = iterate(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, ext_p)
    return sigmas

def run_with_monitoring(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, nr_steps=5, ext_p=None):
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
    if ext_p is not None:
        sigmas_list_ids = [0]
        full_sigmas_list = []
        full_sigmas_list.append(sigmas.copy())
        patterns_presented = []
        for i in range(nr_steps):
            step_in_loop = (i - ext_p["init_steps"]) % (ext_p["feed_steps"] + ext_p["evolve_steps"])
            if i < ext_p["init_steps"] or (ext_p["sequence_length"] >= 0 and step_in_loop >= ext_p["feed_steps"]):
                sigmas = iterate(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, ext_p=None)
            else:
                if step_in_loop == 0:
                    ext_p["sequence_length"] -= 1
                    ext_p["mu"] = np.random.choice(range(len(var_list)))
                    patterns_presented.append(ext_p["mu"])
                sigmas = iterate(sigmas, var_list, function_name, beta, theta, a, K, N, N_I, ext_p)
            if step_in_loop in [0, 1, ext_p["feed_steps"] - 1, ext_p["feed_steps"] + ext_p["evolve_steps"] - 1]:
                sigmas_list.append(sigmas.copy())
                sigmas_list_ids.append(i+1)
            full_sigmas_list.append(sigmas.copy())
        return full_sigmas_list, sigmas_list, sigmas_list_ids, patterns_presented
    else:
        for i in range(nr_steps):
            # run a step
            sigmas = iterate(sigmas, var_list, function_name, beta, theta, a, K, N, N_I)
            sigmas_list.append(sigmas.copy())
        return sigmas_list

def plot_state_sequence_and_overlap(sigmas_sequence, pattern_list, reference_idx=0, color_map="brg", ids=None, suptitle=None, overlap_from=0):
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
    if ids:
        _plot_list(ax[0, :], sigmas_sequence, ids, reference, "S{0}", color_map) # Multiply by 2 and subtract 1 to map {0, 1} to {-1, 1}
    else:
        hf._plot_list(ax[0, :], sigmas_sequence, reference, "S{0}", color_map) # Multiply by 2 and subtract 1 to map {0, 1} to {-1, 1}
    for i in range(len(sigmas_sequence)):
        overlap_list = compute_overlap_list(sigmas_sequence[i], pattern_list, only_from=overlap_from)
        ax[1, i].bar(range(len(overlap_list)), overlap_list)
        ax[1, i].set_title("m = {1}".format(i, round(overlap_list[reference_idx], 2)))
        ax[1, i].set_ylim([-1, 1]) # Set manually to min(mu) and max(mu)
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
    dot_prod = np.dot(norm_sigmas, norm_pattern)  # Compute dot product with activity adjustment
    return float(dot_prod) / (np.prod(pattern.shape) - only_from)  # Normalize and return the overlap    

def study_overlap(cst, sigmas_as_patterns, pattern_list, overlap_from=0, pattern_init=0):
    overlap = compute_overlap(sigmas_as_patterns[cst["T"]], pattern_list[0], only_from=overlap_from)
    if overlap == 1:
        print(f"With {cst['T']} steps, the network converged to the stored pattern {pattern_init}.")
    elif np.round(overlap*100)/100 == 1.0:
        print(f"With {cst['T']} steps, the network approximatively converged to the stored pattern.")
    else:
        print(f"With {cst['T']} steps, the network did not converge to the stored pattern.")
    print(f"The overlap is {overlap}")
    return overlap

# ----------------------- Ex 3.3 -----------------------

def hamming_distance(pattern1, pattern2, N, only_from=0):
    norm_pattern1 = pattern1.copy().flatten()[only_from:] * 2 - 1  # Normalize sigmas to {-1, 1}
    norm_pattern2 = pattern2.copy().flatten()[only_from:] * 2 - 1  # Normalize pattern to {-1, 1}
    N_real = N - only_from # Use only the excitatory neurons for example
    return (N_real-np.dot(norm_pattern1, norm_pattern2))/(2*N_real)

def study_simple_retrieval(sigmas_f_as_pattern, pattern, mu, N, c_f, only_from, silent=False):
    hamming_dist = hamming_distance(np.array(sigmas_f_as_pattern), pattern, N, only_from=only_from)
    if hamming_dist <= c_f:
        if not silent:
            print("The network correctly retrieved the pattern P{}.".format(mu))
            print("The hamming distance is {}.".format(hamming_dist))
        return hamming_dist, mu
    else:
        return hamming_dist, None

def study_simple_capacity(cst, M_values, function_name="sync"):
    mean_retrieved_patterns_list = []
    std_retrieved_patterns_list = []
    max_retrieved_patterns_list = {}

    for M in M_values:
        cst["M"] = M
        print(f">> Computing M={M} value for N={cst['N']}")

        nr_retrieved_patterns_list = []

        weights, pattern_list, shape = generate_random_patterns(cst)
        for i in range(cst["nr_iter"]):
            retrieved_patterns = []
            hamming_distances = []

            for mu in range(M):
                noisy_init_pattern, sigmas, sigmas_as_pattern = flip_and_iterate(cst, shape, pattern_list, init_pattern=mu, only_last_state=True, function_name=function_name)
                hamming_distance, mu = study_simple_retrieval(sigmas_as_pattern, pattern_list[mu], mu, cst["N"], cst["c_f"], silent=True, only_from=cst["N_I"])
                if mu is not None:
                    retrieved_patterns.append(mu)
                hamming_distances.append(hamming_distance)
            
            
            nr_retrieved_patterns_list.append(len(retrieved_patterns))

        mean_retrieved_patterns_list.append(np.mean(nr_retrieved_patterns_list))
        std_retrieved_patterns_list.append(np.std(nr_retrieved_patterns_list))
        max_retrieved_patterns_list[cst["M"]] = np.amax(nr_retrieved_patterns_list)

    capacity = np.amax(list(max_retrieved_patterns_list.values())) / cst["N"]

    plt.errorbar(M_values / cst["N"], mean_retrieved_patterns_list, yerr=std_retrieved_patterns_list, fmt='o')
    plt.xlabel(r"Loadings $L = \frac{M}{N}$ of the network")
    plt.ylabel("Number of retrieved patterns")
    plt.title(f"Number of retrieved patterns as a function of L for N = {cst['N']} neurons")
    plt.show()

    return capacity

def study_capacity(cst, N_values, N_I_values, K_values, prev_capacity, function_name="sync"):
    capacities = []

    for n, N in enumerate(N_values):
        cst["N"] = N
        cst["N_I"] = N_I_values[n]
        cst["K"] = K_values[n]
        print(f"> Computing capacity for N = {cst['N']}" + ", N_I" + f" = {cst['N_I']}" + f" and K = {cst['K']}...")
        # M values are in a range of 5 values of M such that M/N is smaller than the capacity + c_f and M/N is larger than the capacity - c_f
        M_values = np.arange(round((prev_capacity - cst["c_f"]) * N), round((prev_capacity + cst["c_f"]) * N), round((2*cst["c_f"]) * N / 6))
        M_values = [mu for mu in M_values if mu > 0] # Remove any negative total pattern amounts from the list.

        capacity = study_simple_capacity(cst, M_values, function_name=function_name)

        capacities.append(capacity)

    # A plot of the capacity as a function of N:
    plt.figure()
    plt.plot(N_values, capacities)
    plt.xlabel("Number of neurons N")
    plt.ylabel("Capacity")
    plt.title("Capacity as a function of N")
    plt.show()

    return N_values, capacities

# ----------------------- Ex 3.5 -----------------------

def _plot_list(axes_list, state_sequence, ids, reference=None, title_pattern="S({0})", color_map="brg"):
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
        axes_list[i].set_title(title_pattern.format(ids[i]))
        axes_list[i].axis("off")

def study_hamming_distances(cst, sigmas_as_patterns, pattern_list, overlap_from=0, patterns_ids=None):
    plt.figure()
    hamming_distances = []
    if patterns_ids is None:
        patterns_ids = range(len(pattern_list))
    for mu in patterns_ids:
        hamming_distance_list = []
        for t in range(len(sigmas_as_patterns)):
            hamming_distance_list.append(hamming_distance(sigmas_as_patterns[t], pattern_list[mu], cst["N"], only_from=overlap_from))
        hamming_distances.append(hamming_distance_list)
        plt.plot(np.arange(len(sigmas_as_patterns)), hamming_distances[mu], label="Pattern {}".format(mu))
    plt.xlabel("Time step")
    plt.ylabel("Hamming distance")
    plt.legend()
    plt.show()

def plot_raster(cst, full_sigmas_list_as_patterns, separations=[], colors=["#808782", "#656565", "#232323"],
                pop_description =["Inhibitory population 1", "Inhibitory population 2", "Excitatory population"]):
    if len(full_sigmas_list_as_patterns) != (cst["T"] + 1):
        print(len(full_sigmas_list_as_patterns))
        print(cst["T"])
        raise ValueError("The length of the sigmas list is not equal to the number of time steps.")
    
    sigmas_clean = []
    separations.append(cst["N"])
    for sigmas in full_sigmas_list_as_patterns:
        sigmas = sigmas.copy().flatten()
        if len(sigmas) != cst["N"]:
            raise ValueError("The length of the sigmas is not equal to the total number of neurons.")
        sep_index = 0
        population = []
        for i, sigma in enumerate(sigmas):
            if sigma >= 1:
                sigma *= i+1
                population.append(sigma)
            if i == separations[sep_index] - 1:
                sigmas_clean.append(population)
                population = []
                sep_index += 1

    plt.figure()
    pop_index = 0
    step = 0
    for sigmas_list in sigmas_clean:
        if pop_index == len(separations):
            pop_index = 0
            step += 1
        # plt.scatter(np.ones_like(sigmas_list)*step, 
        #             sigmas_list, c=colors[pop_index], marker="s", lw=0, label=pop_description[pop_index] if step == 0 else None)
        plt.scatter(np.ones_like(sigmas_list)*step, 
                    sigmas_list, c=colors[pop_index], marker="8", s=5, lw=0)
        pop_index += 1
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.title("Raster plot of the network activity")
    plt.legend()
    plt.show()