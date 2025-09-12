import numpy as np


def feature_vector(state):
    """Extract features from state - consistent across all components"""
    if isinstance(state, tuple):
        state = state[0]
    state = np.array(state, dtype=np.float32)

    # Normalize state values to reasonable ranges to help with learning
    # state[0] is position, state[1] is velocity, state[2] is angle, state[3] is angular velocity
    if len(state) == 4:  # CartPole
        # Normalize: position, velocity, angle, angular_velocity
        normalized_state = np.array(
            [
                state[0] / 2.4,  # position normalized by boundary
                np.tanh(state[1] / 3.0),  # velocity normalized and bounded
                state[2] / 0.2095,  # angle normalized by boundary
                np.tanh(state[3] / 3.0),  # angular velocity normalized and bounded
            ]
        )
    elif len(state) == 2:  # MountainCar
        # Normalize: position, velocity
        normalized_state = np.array(
            [
                (state[0] + 1.2) / 1.8,  # position normalized to [0,1]
                state[1] / 0.07,  # velocity normalized by max
            ]
        )
    else:
        normalized_state = state

    # Build feature vector
    feats = list(normalized_state)  # Linear features
    feats += list(normalized_state**2)  # Quadratic features
    # sin because it helps with periodic features
    feats += list(np.sin(normalized_state * np.pi))  # Sine features
    # cos because it helps with periodic features
    feats += list(np.cos(normalized_state * np.pi))  # Cosine features

    # Add pairwise products for interaction terms
    for i in range(len(normalized_state)):
        for j in range(i + 1, len(normalized_state)):
            # we multiply because it captures interaction between features
            feats.append(normalized_state[i] * normalized_state[j])

    return np.array(feats, dtype=np.float32)


def feature_expectations(trajs, gamma=0.99):
    mu = np.zeros_like(feature_vector(trajs[0][0]), dtype=np.float64)
    for traj in trajs:
        # traj is a list of (state, action, reward) tuples
        for t, (s, _, _) in enumerate(traj):
            # mu is the feature expectation vector
            mu += (gamma**t) * feature_vector(s)
    mu /= len(trajs)
    return mu
