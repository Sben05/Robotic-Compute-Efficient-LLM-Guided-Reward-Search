@torch.jit.script
def compute_reward(root_states: torch.Tensor, heading_vec: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity from root states
    velocity = root_states[:, 7:10]  # (x, y, z) velocity in world frame
    
    # Compute forward velocity as projection onto heading direction
    forward_velocity = torch.sum(velocity * heading_vec, dim=1)
    
    # Normalize forward velocity using tanh with temperature scaling
    temperature_forward = 0.1
    reward_forward = torch.tanh(forward_velocity * temperature_forward)
    
    # Compute upright projection (alignment with global up direction)
    global_up = torch.tensor([0.0, 0.0, 1.0], device=root_states.device)
    up_proj = torch.sum(up_vec * global_up.unsqueeze(0), dim=1)
    
    # Normalize upright projection using exponential with temperature
    temperature_upright = 2.0
    reward_upright = torch.exp(up_proj * temperature_upright)
    
    # Total reward is weighted sum of components
    total_reward = reward_forward + 0.5 * reward_upright
    
    return total_reward, {
        "forward_velocity_reward": reward_forward,
        "upright_reward": reward_upright
    }
