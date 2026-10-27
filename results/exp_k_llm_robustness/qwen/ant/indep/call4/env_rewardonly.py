@torch.jit.script
def compute_reward(root_states: torch.Tensor, obs_buf: torch.Tensor, heading_vec: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity from root_states (indices 7:10)
    velocity = root_states[:, 7:10]
    
    # Compute forward velocity in heading direction
    forward_velocity = torch.sum(velocity * heading_vec, dim=1)
    
    # Extract up_proj from obs_buf (index 10)
    up_proj = obs_buf[:, 10]
    
    # Compute action penalty (L2 norm of actions)
    action_penalty = torch.sum(actions ** 2, dim=1)
    
    # Device of the input tensors
    device = root_states.device
    
    # Reward coefficients
    forward_coeff = torch.tensor(1.0, device=device)
    up_coeff = torch.tensor(1.0, device=device)
    action_coeff = torch.tensor(0.005, device=device)
    
    # Calculate individual reward components
    forward_reward = forward_velocity * forward_coeff
    up_reward = up_proj * up_coeff
    action_penalty = -action_penalty * action_coeff
    
    # Total reward
    total_reward = forward_reward + up_reward + action_penalty
    
    # Dictionary of reward components
    reward_components = {
        "forward_reward": forward_reward,
        "up_reward": up_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
