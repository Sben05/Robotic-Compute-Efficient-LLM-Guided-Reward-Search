@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   actions: torch.Tensor,
                   dof_vel: torch.Tensor,
                   dof_pos: torch.Tensor,
                   dof_limits_lower: torch.Tensor,
                   dof_limits_upper: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- torso orientation (quaternion -> world axes) ----------------
    quat = root_states[:, 3:7]          # (w, x, y, z)
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]

    # body +x axis expressed in world frame (the ant's heading / forward direction)
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qw * qz + qx * qy)
    fwd_z = 2.0 * (qx * qz - qw * qy)
    forward_dir = torch.stack((fwd_x, fwd_y, fwd_z), dim=-1)

    # body +z axis expressed in world frame (the ant's up direction)
    up_x = 2.0 * (qw * qy + qx * qz)
    up_y = 2.0 * (qy * qz - qw * qx)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    up_vec = torch.stack((up_x, up_y, up_z), dim=-1)

    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    # ---------------- task specific signals ----------------
    forward_vel = torch.sum(lin_vel * forward_dir, dim=-1)     # speed along own heading
    upright = up_vec[:, 2]                                     # 1.0 perfectly upright
    yaw_rate = torch.sum(ang_vel * up_vec, dim=-1)             # spinning about vertical
    lateral_vel = lin_vel[:, 0] * forward_dir[:, 1] - lin_vel[:, 1] * forward_dir[:, 0]

    # ---------------- penalties ----------------
    action_pen = torch.sum(torch.square(actions), dim=-1)
    dof_vel_pen = torch.sum(torch.square(dof_vel), dim=-1)
    ang_vel_pen = torch.sum(torch.square(ang_vel), dim=-1)
    lower_viol = torch.clamp(dof_pos - dof_limits_lower, min=0.0)
    upper_viol = torch.clamp(dof_pos - dof_limits_upper, min=0.0)
    limit_pen = torch.sum(torch.square(lower_viol) + torch.square(upper_viol), dim=-1)

    # ---------------- weights ----------------
    w_speed = 1.0
    w_upright = 1.0
    w_alive = 0.1
    w_action = 0.005
    w_dof_vel = 0.00025
    w_ang_vel = 0.005
    w_limit = 0.01
    w_lateral = 0.02
    w_yaw = 0.01

    # ---------------- reward components ----------------
    reward_speed = w_speed * forward_vel
    reward_upright = w_upright * upright
    reward_alive = w_alive * torch.ones_like(forward_vel)
    penalty_action = -w_action * action_pen
    penalty_dof_vel = -w_dof_vel * dof_vel_pen
    penalty_ang_vel = -w_ang_vel * ang_vel_pen
    penalty_limit = -w_limit * limit_pen
    penalty_lateral = -w_lateral * torch.square(lateral_vel)
    penalty_yaw = -w_yaw * torch.square(yaw_rate)

    total_reward = (reward_speed + reward_upright + reward_alive
                    + penalty_action + penalty_dof_vel + penalty_ang_vel
                    + penalty_limit + penalty_lateral + penalty_yaw)

    components: Dict[str, torch.Tensor] = {
        "forward_speed": reward_speed,
        "upright": reward_upright,
        "alive": reward_alive,
        "action_penalty": penalty_action,
        "dof_vel_penalty": penalty_dof_vel,
        "ang_vel_penalty": penalty_ang_vel,
        "joint_limit_penalty": penalty_limit,
        "lateral_penalty": penalty_lateral,
        "yaw_penalty": penalty_yaw,
    }

    return total_reward, components
