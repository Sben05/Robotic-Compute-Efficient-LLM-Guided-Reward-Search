@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear and angular velocity from root states
    velocity = root_states[:, 7:10]  # Linear velocity in world space
    angular_vel = root_states[:, 10:13]  # Angular velocity in world space

    # Calculate forward velocity in the heading direction
    forward_vel = torch.sum(velocity * heading_vec, dim=-1)

    # Penalize angular velocity around the heading axis (yaw)
    angular_vel_yaw = torch.sum(angular_vel * heading_vec, dim=-1)
    angular_penalty = 0.005 * torch.abs(angular_vel_yaw)

    # Penalize not being upright by checking the z-component of up vector
    # Assumes up axis is z (index 2)
    up_proj = up_vec[:, 2]
    upright_penalty = 0.5 * (1.0 - up_proj)

    # Total reward: forward velocity - penalties
    total_reward = forward_vel - angular_penalty - upright_penalty

    reward_components = {
        "forward_velocity": forward_vel,
        "angular_penalty": angular_penalty,
        "upright_penalty": upright_penalty
    }

    return total_reward, reward_components
