@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ant: run forward (world +x) as fast as possible.

    root_states : [N, 13] -> pos[0:3], quat(x,y,z,w)[3:7], lin_vel[7:10], ang_vel[10:13]
    dof_vel     : [N, num_dof]
    actions     : [N, num_actions]
    """

    # ------------------------- hyper parameters -------------------------
    speed_weight: float = 1.5
    speed_temperature: float = 6.0        # [m/s] tanh scale of the forward-speed reward
    surge_weight: float = 1.0
    surge_start: float = 2.5              # [m/s] where the high-speed ramp starts
    surge_range: float = 7.0              # [m/s] width of the high-speed ramp
    body_speed_weight: float = 0.5
    body_speed_temperature: float = 6.0
    upright_weight: float = 0.6
    upright_temperature: float = 0.3
    heading_weight: float = 0.4
    alive_weight: float = 0.3
    alive_height_offset: float = 0.25     # [m]
    alive_height_range: float = 0.25      # [m]
    action_cost_coeff: float = -0.01
    dof_vel_cost_coeff: float = -5.0e-05
    lateral_cost_coeff: float = -0.02
    vertical_cost_coeff: float = -0.01
    ang_vel_cost_coeff: float = -5.0e-04

    # ------------------------- state extraction -------------------------
    # root quaternion is stored as (x, y, z, w)
    qx = root_states[:, 3]
    qy = root_states[:, 4]
    qz = root_states[:, 5]
    qw = root_states[:, 6]

    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]
    torso_z = root_states[:, 2]

    # body x-axis (forward) expressed in world frame
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qz * qw)
    fwd_z = 2.0 * (qx * qz - qy * qw)

    # body z-axis (up) expressed in world frame
    up_x = 2.0 * (qx * qz + qy * qw)
    up_y = 2.0 * (qy * qz - qx * qw)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # ------------------------- reward components -------------------------
    # 1) main term: world-frame forward running speed (soft, wide temperature)
    forward_speed = lin_vel[:, 0]
    speed_reward = speed_weight * torch.tanh(forward_speed / speed_temperature)

    # 2) high-speed ramp: constant gradient well beyond the current speed
    surge_reward = surge_weight * torch.clamp((forward_speed - surge_start) / surge_range,
                                              min=0.0, max=1.0)

    # 3) speed along the body's own forward axis (running, not sliding sideways)
    body_forward_speed = (lin_vel[:, 0] * fwd_x +
                          lin_vel[:, 1] * fwd_y +
                          lin_vel[:, 2] * fwd_z)
    body_speed_reward = body_speed_weight * torch.tanh(body_forward_speed / body_speed_temperature)

    # 4) stay upright (true body up-axis vs world vertical)
    upright_reward = upright_weight * torch.exp((up_z - 1.0) / upright_temperature)

    # 5) keep the body facing the running direction (world +x)
    heading_reward = heading_weight * fwd_x

    # 6) alive: torso must stay at a sensible height
    alive_reward = alive_weight * torch.clamp((torso_z - alive_height_offset) / alive_height_range,
                                               min=0.0, max=1.0)

    # 7) mild regularizers (kept small so they do not cap the speed)
    action_cost = action_cost_coeff * torch.sum(actions * actions, dim=1)
    dof_vel_cost = dof_vel_cost_coeff * torch.sum(dof_vel * dof_vel, dim=1)
    lateral_cost = lateral_cost_coeff * lin_vel[:, 1] * lin_vel[:, 1]
    vertical_cost = vertical_cost_coeff * lin_vel[:, 2] * lin_vel[:, 2]
    ang_vel_cost = ang_vel_cost_coeff * torch.sum(ang_vel * ang_vel, dim=1)

    total = (speed_reward + surge_reward + body_speed_reward + upright_reward +
             heading_reward + alive_reward + action_cost + dof_vel_cost +
             lateral_cost + vertical_cost + ang_vel_cost)

    reward_dict = {"forward_speed": speed_reward}
    reward_dict["surge_speed"] = surge_reward
    reward_dict["body_forward_speed"] = body_speed_reward
    reward_dict["upright"] = upright_reward
    reward_dict["heading_align"] = heading_reward
    reward_dict["alive"] = alive_reward
    reward_dict["action_cost"] = action_cost
    reward_dict["dof_vel_cost"] = dof_vel_cost
    reward_dict["lateral_vel_cost"] = lateral_cost
    reward_dict["vertical_vel_cost"] = vertical_cost
    reward_dict["ang_vel_cost"] = ang_vel_cost

    return total, reward_dict
