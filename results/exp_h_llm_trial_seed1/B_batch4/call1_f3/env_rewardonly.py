@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                     heading_vec: torch.Tensor,
                     dof_pos: torch.Tensor,
                     dof_limits_lower: torch.Tensor,
                     dof_limits_upper: torch.Tensor,
                     dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward 3 - saturating (exponential) speed reward with posture / joint regularisation.

      * speed term 1 - exp(-v / T) : monotone in speed but bounded, so that very fast
        running does not dominate all other terms (T = speed_temperature)
      * uprightness computed directly from the torso quaternion (xyzw):
        up_z = 1 - 2 (qx^2 + qy^2) = cos(tilt)
      * penalty on joint limit violations and on joint velocities (smooth, cheap gait)
    """
    speed_temperature = 2.0
    limit_penalty_weight = 0.05
    joint_vel_penalty_weight = 0.0005

    # forward speed along the heading direction
    vel_xy = root_states[:, 7:9]
    heading_xy = heading_vec[:, 0:2]
    heading_xy = heading_xy / (torch.norm(heading_xy, p=2, dim=-1, keepdim=True) + 1.0e-6)
    forward_speed = torch.sum(vel_xy * heading_xy, dim=-1)

    # saturating speed reward (monotonically increasing, upper bounded by 1)
    speed_reward = 1.0 - torch.exp(-forward_speed / speed_temperature)

    # uprightness from the root quaternion (Isaac Gym stores xyzw)
    q = root_states[:, 3:7]
    qw = q[:, 3]
    qx = q[:, 0]
    qy = q[:, 1]
    qz = q[:, 2]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright_reward = torch.clamp(up_z, min=0.0, max=1.0)

    # joint limit violations
    upper_violation = torch.clamp(dof_pos - dof_limits_upper, min=0.0)
    lower_violation = torch.clamp(dof_limits_lower - dof_pos, min=0.0)
    limit_penalty = -limit_penalty_weight * torch.sum(
        torch.square(upper_violation) + torch.square(lower_violation), dim=-1)

    # joint velocity smoothness
    joint_vel_penalty = -joint_vel_penalty_weight * torch.sum(torch.square(dof_vel), dim=-1)

    total = 3.0 * speed_reward + 0.5 * upright_reward + limit_penalty + joint_vel_penalty

    components: Dict[str, torch.Tensor] = {
        "forward_speed": 3.0 * speed_reward,
        "upright": 0.5 * upright_reward,
        "joint_limits": limit_penalty,
        "joint_velocity": joint_vel_penalty,
    }
    return total, components
