@torch.jit.script
def compute_reward(root_states: torch.Tensor, sensor_force_torques: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    heading_vec = root_states[:, 3:6]  # Assuming heading is aligned with base orientation
    heading_vec = heading_vec / (torch.norm(heading_vec, dim=1, keepdim=True) + 1e-6)
    forward_speed = torch.sum(velocity * heading_vec, dim=1) / dt
    energy_temp = 0.01
    energy_cost = energy_temp * torch.sum(torch.abs(sensor_force_torques), dim=1)
    total_reward = forward_speed - energy_cost
    reward_components = {
        "forward_speed": forward_speed,
        "energy_cost": energy_cost
    }
    return total_reward, reward_components
