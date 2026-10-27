@torch.jit.script
def compute_reward(root_states: torch.Tensor, heading_vec: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    root_velocity = root_states[:, 7:10]
    forward_vel = torch.sum(root_velocity * heading_vec, dim=1)
    action_cost = 0.01 * torch.norm(actions, dim=1)
    reward = forward_vel * dt - action_cost
    return reward, {"forward_vel": forward_vel * dt, "action_cost": action_cost}
