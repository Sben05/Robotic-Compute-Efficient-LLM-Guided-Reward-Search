@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ant: run forward (+x world) as fast as possible.

    root_states : [N, 13]  pos[0:3], quat(x,y,z,w)[3:7], lin_vel[7:10], ang_vel[10:13]
    dof_vel     : [N, num_dof]
    actions     : [N, num_actions]
    """
    # ------------------------- hyper parameters -------------------------
    speed_weight: float = 1.5               # weight of world-frame forward speed
    speed_temperature: float = 4.0          # [m/s] soft saturation scale
    body_speed_weight: float = 0.3           # weight of body-frame forward speed
    body_speed_temperature: float = 4.0      # [m/s]
    surge_weight: float = 0.4                # extra push to run really fast
    surge_threshold: float = 4.0            # [m/s]
    surge_range: float = 2.0                # [m/s]
    upright_weight: float = 0.5
    upright_temperature: float = 0.5
    fall_weight: float = 0.5                 # penalty when the torso tips over
    fall_threshold: float = 0.3              # up_z below this == falling
    fall_range: float = 0.4
    heading_weight: float = 0.2              # keep facing the travel direction
    action_cost_coeff: float = -0.03
    dof_vel_cost_coeff: float = -0.0005
    lateral_cost_coeff: float = -0.05
    vertical_cost_coeff: float = -0.05
    ang_vel_cost_coeff: float = -0.001

    # ------------------------- state extraction -------------------------
    # NOTE: Isaac Gym stores the root quaternion as (x, y, z, w)
    qx = root_states[:, 3]
    qy = root_states[:, 4]
    qz = root_states[:, 5]
    qw = root_states[:, 6]

    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    # body forward axis (body x) expressed in world frame
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qz * qw)
    # body up axis (body z) world-z component -> 1 when perfectly upright
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # ------------------------- reward components -------------------------
    # 1) world-frame forward running speed (main term, soft-saturating)
    forward_speed = lin_vel[:, 0]
    speed_reward = speed_weight * (1.0 - torch.exp(-forward_speed / speed_temperature))

    # 2) body-frame forward speed (rewards running along the facing direction)
    body_forward_speed = lin_vel[:, 0] * fwd_x + lin_vel[:, 1] * fwd_y
    body_speed_reward = body_speed_weight * (1.0 - torch.exp(-body_forward_speed /
                                                             body_speed_temperature))

    # 3) surge bonus once the ant is already fast -> push for even more speed
    surge_reward = surge_weight * torch.clamp((forward_speed - surge_threshold) / surge_range,
                                              min=0.0, max=1.0)

    # 4) stay upright (torso up axis aligned with world z)
    upright_reward = upright_weight * torch.exp((up_z - 1.0) / upright_temperature)

    # 5) fall penalty (only active when the torso really tips over)
    fall_penalty = -fall_weight * torch.clamp((fall_threshold - up_z) / fall_range,
                                              min=0.0, max=1.0)

    # 6) keep the body facing the direction of travel
    heading_align = torch.clamp(fwd_x, min=-1.0, max=1.0)
    heading_reward = heading_weight * heading_align

    # 7) regularizers
    action_cost = action_cost_coeff * torch.sum(actions * actions, dim=1)
    dof_vel_cost = dof_vel_cost_coeff * torch.sum(dof_vel * dof_vel, dim=1)
    lateral_cost = lateral_cost_coeff * lin_vel[:, 1] * lin_vel[:, 1]
    vertical_cost = vertical_cost_coeff * lin_vel[:, 2] * lin_vel[:, 2]
    ang_vel_cost = ang_vel_cost_coeff * torch.sum(ang_vel * ang_vel, dim=1)

    total = (speed_reward + body_speed_reward + surge_reward + upright_reward +
             fall_penalty + heading_reward + action_cost + dof_vel_cost +
             lateral_cost + vertical_cost + ang_vel_cost)

    reward_dict = {"forward_speed": speed_reward}
    reward_dict["body_forward_speed"] = body_speed_reward
    reward_dict["surge_speed"] = surge_reward
    reward_dict["upright"] = upright_reward
    reward_dict["fall_penalty"] = fall_penalty
    reward_dict["heading_align"] = heading_reward
    reward_dict["action_cost"] = action_cost
    reward_dict["dof_vel_cost"] = dof_vel_cost
    reward_dict["lateral_vel_cost"] = lateral_cost
    reward_dict["vertical_vel_cost"] = vertical_cost
    reward_dict["ang_vel_cost"] = ang_vel_cost

    return total, reward_dict
