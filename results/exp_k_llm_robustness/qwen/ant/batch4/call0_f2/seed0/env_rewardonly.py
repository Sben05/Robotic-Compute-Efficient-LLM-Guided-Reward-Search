@torch.jit.script
def compute_reward(root_states: torch.Tensor, heading_vec: torch.Tensor, up_vec: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    root_velocity = root_states[:, 7:10]
    # Forward component
    forward_vel = torch.sum(root_velocity * heading_vec, dim=1)
    # Lateral component (cross product of heading and up gives right direction)
    right_dir = torch.nn.functional.normalize(torch.cross(heading_vec, up_vec), dim=1)
    lateral_vel = torch.sum(root_velocity * right_dir, dim=1)
    reward = forward_vel * dt - 0.5 * torch.abs(lateral_vel) * dt
    return reward, {"forward_vel": forward_vel * dt, "lateral_penalty": 0.5 * torch.abs(lateral_vel) * dt}
