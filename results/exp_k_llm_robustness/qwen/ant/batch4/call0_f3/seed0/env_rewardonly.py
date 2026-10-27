@torch.jit.script
def compute_reward(prev_potentials: torch.Tensor, potentials: torch.Tensor, root_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pot_diff = prev_potentials - potentials  # Progress toward goal
    ang_vel = root_states[:, 10:13]  # Angular velocity
    ang_vel_cost = 0.1 * torch.sum(ang_vel ** 2, dim=1)
    reward = pot_diff - ang_vel_cost
    return reward, {"potential_diff": pot_diff, "angular_vel_penalty": ang_vel_cost}
