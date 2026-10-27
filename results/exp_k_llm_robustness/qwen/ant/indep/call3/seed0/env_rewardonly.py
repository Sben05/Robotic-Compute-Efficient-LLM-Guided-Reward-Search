@torch.jit.script
def compute_reward(obs_buf: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant components from obs_buf
    forward_vel = obs_buf[:, 1]  # Forward velocity in local frame
    lateral_vel = obs_buf[:, 2]  # Lateral velocity in local frame
    spin_vel = obs_buf[:, 6]     # Angular velocity around z-axis (spin)
    up_proj = obs_buf[:, 10]     # Upright projection
    
    # Temperature parameters for transformations
    temp_forward = 0.1
    temp_lateral = 1.0
    temp_spin = 1.0
    temp_upright = 1.0
    energy_scale = 0.001
    
    # Compute individual reward components
    reward_forward = torch.tanh(forward_vel * temp_forward)
    reward_lateral = - (lateral_vel ** 2) * temp_lateral
    reward_spin = - (spin_vel ** 2) * temp_spin
    reward_upright = up_proj * temp_upright
    reward_energy = - energy_scale * torch.sum(actions ** 2, dim=1)
    
    # Total reward
    total_reward = reward_forward + reward_lateral + reward_spin + reward_upright + reward_energy
    
    # Dictionary of individual components
    reward_components = {
        "forward": reward_forward,
        "lateral": reward_lateral,
        "spin": reward_spin,
        "upright": reward_upright,
        "energy": reward_energy
    }
    
    return total_reward, reward_components
