def step_activation(x : float) -> int:
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0