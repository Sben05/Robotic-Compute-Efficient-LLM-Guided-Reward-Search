@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ant: run forward (+X world) as fast as possible.

    root_states : [N, 13] -> pos[0:3], quat (x,y,z,w)[3:7], lin_vel[7:10], ang_vel[10:13]
    dof_vel     : [N, num_dof] joint velocities
    actions     : [N, num_actions] last applied actions
    """

    # ------------------------- hyper parameters -------------------------
    speed_weight: float = 1.0            # soft (tanh) forward speed
    speed_temperature: float = 6.0       # [m/s] scale -> keeps gradient alive at high speed
    linear_speed_weight: float = 0.8     # linear forward speed (no early saturation)
    linear_speed_scale: float = 12.0     # [m/s] speed mapped to +-1
    surge_weight: float = 0.4            # tier 1 bonus
    surge_start: float = 4.0             # [m/s]
    surge_range: float = 4.0             # [m/s]
    super_surge_weight: float = 0.3      # tier 2 bonus (very fast running)
    super_surge_start: float = 8.0       # [m/s]
    super_surge_range: float = 4.0       # [m/s]
    body_speed_weight: float = 0.4       # speed along the body forward axis
    body_speed_temperature: float = 6.0  # [m/s]
    upright_weight: float = 1.2
    upright_temperature: float = 0.4     # exp() scale on the body up-axis
    heading_weight: float = 0.15         # only paid when actually facing forward
    action_cost_coeff: float = -0.02
    dof_vel_cost_coeff: float = -0.00002
    lateral_cost_coeff: float = -0.03
    vertical_cost_coeff: float = -0.03
    ang_vel_cost_coeff: float = -0.0015

    # ------------------------- state extraction -------------------------
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    # Isaac Gym root quaternion is stored as (x, y, z, w)
    qx = root_states[:, 3]
    qy = root_states[:, 4]
    qz = root_states[:, 5]
    qw = root_states[:, 6]

    # body forward axis (local +x) expressed in world frame
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    # body up axis (local +z) world component -> 1 when perfectly upright
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    v_world = lin_vel[:, 0]                                   # world +X speed
    v_body = lin_vel[:, 0] * fwd_x + lin_vel[:, 1] * fwd_y      # speed along body forward

    # ------------------------- reward components -------------------------
    # 1) soft saturating forward speed (never fully saturates at realistic speeds)
    speed_reward = speed_weight * torch.tanh(v_world / speed_temperature)

    # 2) linear forward speed -> constant marginal gain for going faster
    linear_speed = linear_speed_weight * torch.clamp(v_world / linear_speed_scale,
                                                     min=-1.0, max=1.0)

    # 3) speed tier bonuses (reward for genuinely fast running)
    surge = surge_weight * torch.clamp((v_world - surge_start) / surge_range,
                                       min=0.0, max=1.0)
    super_surge = super_surge_weight * torch.clamp(
        (v_world - super_surge_start) / super_surge_range, min=0.0, max=1.0)

    # 4) run along the body's own forward axis (no side-slipping / backwards)
    body_speed = body_speed_weight * torch.tanh(v_body / body_speed_temperature)

    # 5) stay upright (replaces the redundant constant 'alive' term)
    upright = upright_weight * torch.exp((up_z - 1.0) / upright_temperature)

    # 6) only reward heading when it is actually pointing forward
    heading = heading_weight * torch.clamp(fwd_x, min=0.0, max=1.0)

    # 7) regularizers
    action_cost = action_cost_coeff * torch.sum(actions * actions, dim=1)
    dof_vel_cost = dof_vel_cost_coeff * torch.sum(dof_vel * dof_vel, dim=1)
    lateral_cost = lateral_cost_coeff * lin_vel[:, 1] * lin_vel[:, 1]
    vertical_cost = vertical_cost_coeff * lin_vel[:, 2] * lin_vel[:, 2]
    ang_vel_cost = ang_vel_cost_coeff * torch.sum(ang_vel * ang_vel, dim=1)

    total = (speed_reward + linear_speed + surge + super_surge + body_speed +
             upright + heading + action_cost + dof_vel_cost +
             lateral_cost + vertical_cost + ang_vel_cost)

    reward_dict = {"forward_speed": speed_reward}
    reward_dict["linear_speed"] = linear_speed
    reward_dict["surge_speed"] = surge
    reward_dict["super_surge_speed"] = super_surge
    reward_dict["body_forward_speed"] = body_speed
    reward_dict["upright"] = upright
    reward_dict["heading"] = heading
    reward_dict["action_cost"] = action_cost
    reward_dict["dof_vel_cost"] = dof_vel_cost
    reward_dict["lateral_vel_cost"] = lateral_cost
    reward_dict["vertical_vel_cost"] = vertical_cost
    reward_dict["ang_vel_cost"] = ang_vel_cost

    return total, reward_dict
