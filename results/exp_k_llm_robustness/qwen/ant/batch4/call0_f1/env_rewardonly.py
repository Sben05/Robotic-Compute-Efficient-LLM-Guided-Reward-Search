@torch.jit.script
def compute_reward(root_states: torch.Tensor, heading_vec: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    root_velocity = root_states[:, 7:10]  # (num_envs, 3)
    forward_velocity = torch.sum(root_velocity * heading_vec, dim=1)  # dot product (forward speed)
    reward = forward_velocity * dt  # scale by time step
    return reward, {"forward_velocity": reward}
