@torch.jit.script
def compute_reward(root_states: torch.Tensor, heading_vec: torch.Tensor, obs_buf: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    forward_speed = torch.sum(velocity * heading_vec, dim=1) / dt  # Forward velocity in heading direction
    upright_proj = obs_buf[:, 10]  # up_proj from obs_buf
    upright_weight = 0.1
    total_reward = forward_speed + upright_weight * upright_proj
    reward_components = {
        "forward_speed": forward_speed,
        "upright_reward": upright_proj
    }
    return total_reward, reward_components
