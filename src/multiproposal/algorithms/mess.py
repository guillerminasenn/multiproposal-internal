# MESS (M >= 1 proposals)

# algorithms/mess.py
import numpy as np
from .utils import solve_transition_lp

def mess_step(
    x,
    problem,
    rng,
    M=1,
    use_lp=False,
    distance_metric='angular',
    lam=0.1,
    P0=None,
    return_diagnostics=False,
):
    """Perform a MESS step.
    Parameters
    ----------
    x : np.ndarray
        Current state.
    problem : multiproposal.problems.base.Problem
        Problem instance.
    rng : np.random.Generator
        Random number generator.
    M : int
        Number of proposals to generate.
    use_lp : bool
        If True, compute the entries of the transition matrix using
        linear programming. If False, split probability evenly across
        candidate proposals.
    distance_metric: string
        Specifies the distance metric used to compute the distance
        between candidate proposals. Use 'angular' for great-circle 
        angular distance between angles and 'euclidean' for Euclidean 
        distance between the corresponding proposals.
    lam : float
        Weight parameter in the regularization term of the objective
        function in lp.
    P0 : np.ndarray or None
        Initial doubly stochastic matrix for the transition probabilities.
        If None, a uniform matrix with zero diagonal is used.
    return_diagnostics : bool
        If True, return per-iteration diagnostics including the accepted
        proposal index and distances to all candidates.

    Returns
    -------
    x_new : np.ndarray
        New state after the MESS step.
    nr_intervals : int
        Number of shrinking steps performed.
    P1 : np.ndarray
        Transition matrix used to sample the new state (only if lp=True).
    diagnostics : list
        Per-iteration diagnostic data (only if return_diagnostics=True).
    """
    # Initialize transition matrix to None
    P1 = None
    diagnostics = [] if return_diagnostics else None

    # Center the current state and the auxiliary sample from the prior
    x_centered = x - problem.prior_mean()
    nu_centered = problem.sample_prior(rng) - problem.prior_mean() 

    # Sample the likelihood threshold
    logy = problem.log_likelihood(x) + np.log(rng.uniform())

    # Sample alpha, the angle corresponding to the current state
    alpha = rng.uniform(0, 2*np.pi)

    # Initialize the angle interval
    phi_min = 0
    phi_max = 2 * np.pi

    # Shrink interval until acceptance
    nr_intervals = 0
    while True:
        # Sample M angles
        phi_vector = rng.uniform(phi_min, phi_max, size=M)

        # Compute the proposals
        x_prop_vector = (
            problem.prior_mean()[:, np.newaxis]
            + np.cos(phi_vector - alpha) * x_centered[:, np.newaxis]
            + np.sin(phi_vector - alpha) * nu_centered[:, np.newaxis]
        )

        # Evaluate the likelihood for all proposals
        log_likelihoods = np.array([
            problem.log_likelihood(x_prop_vector[:, i])
            for i in range(M)
        ])

        # Compute A_i, the set of indexes of the candidate proposals
        A = np.where(log_likelihoods > logy)[0]

        diag_entry = None
        if return_diagnostics:
            abs_angular_dist = np.abs(phi_vector - alpha)
            angular_distances = np.minimum(abs_angular_dist, 2 * np.pi - abs_angular_dist)
            diff = x_prop_vector - x[:, np.newaxis]
            euclidean_distances = np.linalg.norm(diff, axis=0)
            diag_entry = {
                'phi_min': float(phi_min),
                'phi_max': float(phi_max),
                'alpha': float(alpha),
                'phi_vector': phi_vector.copy(),
                'log_likelihoods': log_likelihoods.copy(),
                'valid_indices': A.copy(),
                'angular_distances': angular_distances,
                'euclidean_distances': euclidean_distances,
                'accepted_index': None,
            }
            diagnostics.append(diag_entry)

        # If there are valid proposals, select one and return
        if len(A) > 0:

            # Sample the proposal using a transition matrix computed with lp
            if use_lp:
                psi = np.concatenate([phi_vector[A], np.array([alpha])])
                psi_sorted = np.sort(psi)
                
                # Compute the distance matrix
                if distance_metric== 'angular':
                    abs_angular_dist = np.abs(psi_sorted[:, None] - psi_sorted[None, :])
                    D = np.minimum(abs_angular_dist, 2 * np.pi - abs_angular_dist)

                elif distance_metric== 'euclidean' or distance_metric == 'euclidean_squared':
                    # Compute the proposals corresponding to the sorted angles (NOTE: inefficient, because I already have them above)
                    x_psi_sorted = (
                        problem.prior_mean()[:, np.newaxis]
                        + np.cos(psi_sorted - alpha) * x_centered[:, np.newaxis]
                        + np.sin(psi_sorted - alpha) * nu_centered[:, np.newaxis]
                    )

                    # Compute the Euclidean distance matrix between these proposals
                    diff = x_psi_sorted[:, :, np.newaxis] - x_psi_sorted[:, np.newaxis, :]  # (d, n, n)

                    if distance_metric == 'euclidean':
                        D = np.linalg.norm(diff, axis=0)
                    elif distance_metric == 'euclidean_squared':
                        D = np.sum(diff**2, axis=0)

                # Compute the transition matrix

                # If not specified, use the initial doubly stochastic matrix (uniform, zero diagonal)
                if P0 is None:
                    P0 = np.ones((len(A) + 1 , len(A) + 1)) / (len(A))
                np.fill_diagonal(P0, 0)

                # Solve
                P1 = solve_transition_lp(D, P0, lam=lam, verbose=False)
                
                # Sample i according to the row of P1 corresponding to the current state
                current_index = np.where(psi_sorted == alpha)[0][0]
                row_P1 = P1[current_index, :]

                # Remove index corresponding to the current state
                row_P1 = np.delete(row_P1, current_index)
                i = rng.choice(A, p=row_P1)

            # Sample uniformly among the valid proposals
            else:
                i = rng.choice(A)
            if return_diagnostics and diag_entry is not None:
                diag_entry['accepted_index'] = int(i)

            if return_diagnostics:
                return x_prop_vector[:, i], nr_intervals, P1, diagnostics
            return x_prop_vector[:, i], nr_intervals, P1

        # Otherwise, shrink the angle interval
        phi_min = np.max(np.concatenate([np.array([phi_min]), phi_vector[np.where(phi_vector < alpha)]]))
        phi_max = np.min(np.concatenate([np.array([phi_max]), phi_vector[np.where(phi_vector >= alpha)]]))

        # Count the number of shrinking steps
        nr_intervals += 1