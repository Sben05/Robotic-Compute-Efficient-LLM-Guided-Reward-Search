@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, potentials: torch.Tensor,
                     prev_potentials: torch.Tensor, dof_pos: torch.Tensor,
                     dof_limits_lower: torch.Tensor, dof_limits_upper: torch.Tensor,
                     dof_vel: torch.Tensor,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Potential-based progress shaping (closing speed) + joint-limit / smoothness penalties."""
    vel = root_states[:, 7:10]
    quat = root_states[:, 3:7]
    qw = quat[:, 0:1]
    qxyz = quat[:, 1:4]

    zeros = torch.zeros_like(qw)
    e_x = torch.cat((torch.ones_like(zeros), zeros, zeros), dim=-1)
    t1 = torch.cross(qxyz, e_x, dim=-1)
    t2 = torch.cross(qxyz, t1, dim=-1)
    fwd = e_x + 2.0 * (qw * t1 + t2)

    fwd_speed = torch.sum(vel * fwd, dim=-1)
    up_proj = up_vec[:, 2]
    upright = torch.clamp(up_proj, min=0.0, max=1.0)

    # potentials = -||target - pos|| / dt  ->  difference = closing speed on the target
    progress_reward = prev_potentials - potentials

    # joint limit violations (keeps the gait anatomically valid)
    lower_viol = torch.clamp(dof_limits_lower - dof_pos, min=0.0)
    upper_viol = torch.clamp(dof_pos - dof_limits_upper, min=0.0)
    limit_penalty = -0.1 * torch.sum(lower_viol + upper_viol, dim=-1)

    # gait smoothness: penalise fast / large joint motion and large control effort
    joint_speed = torch.sum(dof_vel * dof_vel, dim=-1)
    smoothness_penalty = -0.003 * joint_speed - 0.01 * torch.sum(actions * actions, dim=-1)

    fall_penalty = -3.0 * torch.clamp(-up_proj, min=0.0) \
        - 2.0 * torch.clamp(0.8 - root_states[:, 2], min=0.0)
    upright_reward = 0.5 * upright

    reward = 1.0 * fwd_speed + 2.0 * progress_reward + upright_reward \
        + limit_penalty + smoothness_penalty + fall_penalty

    components: Dict[str, torch.Tensor] = {}
    components["forward_speed"] = fwd_speed
    components["progress_shaping"] = 2.0 * progress_reward
    components["upright_bonus"] = upright_reward
    components["joint_limit_penalty"] = limit_penalty
    components["smoothness_penalty"] = smoothness_penalty
    components["fall_penalty"] = fall_penalty
    return reward, components
