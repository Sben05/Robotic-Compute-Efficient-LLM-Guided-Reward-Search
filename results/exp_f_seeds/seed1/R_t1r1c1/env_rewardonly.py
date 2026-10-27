@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ant "run forward as fast as possible" reward.

    root_states : [N, 13] -> pos[0:3], quat (x,y,z,w)[3:7], lin_vel[7:10], ang_vel[10:13]
    dof_vel     : [N, num_dof] joint velocities
    actions     : [N, num_actions] last applied actions
    """
    # ------------------------- hyper parameters -------------------------
    speed_weight: float = 1.0
    vel_temperature: float = 3.0          # [m/s] scale of the forward-speed reward
    surge_weight: float = 0.5             # extra push beyond 2 m/s
    surge_start: float = 2.0              # [m/s]
    surge_range: float = 2.0              # [m/s]
    upright_weight: float = 1.0
    upright_temperature: float = 0.5
    alive_weight: float = 0.5
    heading_weight: float = 0.2
    action_cost_coeff: float = -0.05
    dof_vel_cost_coeff: float = -0.0002
    lateral_cost_coeff: float = -0.05
    vertical_cost_coeff: float = -0.05
    ang_vel_cost_coeff: float = -0.002

    # ------------------------- state extraction -------------------------
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    # root quaternion is (x, y, z, w) in Isaac Gym
    qw = root_states[:, 3]
    qx = root_states[:, 4]
    qy = root_states[:, 5]
    qz = root_states[:, 6]

    # body forward axis (x) expressed in world frame
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qw * qz + qx * qy)
    # body up axis (z) expressed in world frame (world z component)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # ------------------------- reward components -------------------------
    # 1) forward (world +x) running speed
    forward_speed = lin_vel[:, 0]
    speed_reward = speed_weight * torch.tanh(forward_speed / vel_temperature)

    # 2) extra incentive for genuinely fast running
    surge_reward = surge_weight * torch.clamp((forward_speed - surge_start) / surge_range,
                                              min=0.0, max=1.0)

    # 3) stay upright
    upright_reward = upright_weight * torch.exp((up_z - 1.0) / upright_temperature)

    # 4) alive bonus (only while the torso is not flipped over)
    alive = torch.clamp((up_z - 0.2) / 0.5, min=0.0, max=1.0)
    alive_reward = alive_weight * alive

    # 5) keep the body facing the direction of travel
    heading_align = fwd_x / torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + 1.0e-6)
    heading_reward = heading_weight * heading_align

    # 6) regularizers
    action_cost = action_cost_coeff * torch.sum(actions * actions, dim=1)
    dof_vel_cost = dof_vel_cost_coeff * torch.sum(dof_vel * dof_vel, dim=1)
    lateral_cost = lateral_cost_coeff * lin_vel[:, 1] * lin_vel[:, 1]
    vertical_cost = vertical_cost_coeff * lin_vel[:, 2] * lin_vel[:, 2]
    ang_vel_cost = ang_vel_cost_coeff * torch.sum(ang_vel * ang_vel, dim=1)

    total = (speed_reward + surge_reward + upright_reward + alive_reward +
             heading_reward + action_cost + dof_vel_cost +
             lateral_cost + vertical_cost + ang_vel_cost)

    reward_dict = {"forward_speed": speed_reward}
    reward_dict["surge_speed"] = surge_reward
    reward_dict["upright"] = upright_reward
    reward_dict["alive"] = alive_reward
    reward_dict["heading_align"] = heading_reward
    reward_dict["action_cost"] = action_cost
    reward_dict["dof_vel_cost"] = dof_vel_cost
    reward_dict["lateral_vel_cost"] = lateral_cost
    reward_dict["vertical_vel_cost"] = vertical_cost
    reward_dict["ang_vel_cost"] = ang_vel_cost

    return total, reward_dict
