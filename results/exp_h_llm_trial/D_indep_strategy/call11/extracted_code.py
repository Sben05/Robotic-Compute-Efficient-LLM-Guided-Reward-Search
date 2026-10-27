def compute_reward(root_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward for making the ant run forward as fast as possible.

    root_states: [N, 13] =
