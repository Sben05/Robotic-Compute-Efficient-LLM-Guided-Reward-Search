@torch.jit.script
def compute_reward(root_states: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity from root states (indices 7:10)
    velocity = root_states[:, 7:10]
    
    # Compute forward velocity: dot product of velocity and heading direction
    forward_velocity = torch.sum(velocity * heading_vec, dim=1)
    
    # Reward is the forward velocity (encourages maximum forward speed)
    reward = forward_velocity
    
    # Component-wise breakdown (for analysis)
    reward_components = {
        "forward_velocity": reward
    }
    
    return reward, reward_components
