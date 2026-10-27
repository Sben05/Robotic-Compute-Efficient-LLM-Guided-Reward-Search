@torch.jit.script
def compute_reward(root_states: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity (3D) from root states
    linear_velocity = root_states[:, 7:10]
    
    # Calculate forward velocity component in heading direction
    forward_velocity = torch.sum(linear_velocity * heading_vec, dim=-1)
    
    # Reward is the forward velocity component
    reward = forward_velocity
    
    return reward, {"forward_velocity": reward}
