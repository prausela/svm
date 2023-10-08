def step_activation(x: float) -> int:
    if x < 0:
        return -1
    return 1


activation_function_types = {
    step_activation: "step"
}
