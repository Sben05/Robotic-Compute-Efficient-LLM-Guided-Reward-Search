@torch.jit.script
def compute_reward(root_states: torch.Tensor, obs_buf: torch.Tensor, dof_vel: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]
    heading_vec = root_states[:, 3:6]
    heading_vec = heading_vec / (torch.norm(heading_vec, dim=1, keepdim=True) + 1e-6)
    forward_speed = torch.sum(velocity * heading_vec, dim=1) / dt
    upright_proj = obs_buf[:, 10]
    joint_vel_penalty = torch.sum(dof_vel ** 2, dim=1)
    joint_vel_weight = 0.001
    upright_weight = 0.1
    total_reward = forward_speed + upright_weight * upright_proj - joint_vel_weight * joint_vel_penalty
    reward_components = {
        "forward_speed": forward_speed,
        "upright_reward": upright_proj,
        "joint_vel_penalty": joint_vel_penalty
    }
    return total_reward, reward_components
