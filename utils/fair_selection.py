import numpy as np


def make_candidate_seeds(outer_seed, n_candidates, stride):
    """Deterministic candidate seeds shared by every compared method.

    Candidate numbering is one-based so candidate 1 matches the first DDQN
    episode used in the original best-of-five protocol.
    """
    outer_seed = int(outer_seed)
    n_candidates = int(n_candidates)
    stride = int(stride)

    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")
    if stride < 1:
        raise ValueError("stride must be positive")

    return [
        outer_seed + stride * candidate_index
        for candidate_index in range(1, n_candidates + 1)
    ]


def select_best_candidate(validation_scores):
    """Return the zero-based index with the smallest validation score."""
    scores = np.asarray(validation_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("validation_scores cannot be empty")
    if not np.all(np.isfinite(scores)):
        raise ValueError("validation_scores contains a non-finite value")
    return int(np.argmin(scores))
