@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ant: run forward (+x world) as fast as possible.

    root_states : [N, 13]  pos[0:3], quat (x,y,z,w)[3:7], lin_vel[7:10], ang_vel[10:13]
    dof_vel     : [N, D]   joint velocities
    actions     : [N, A]   last applied actions
    """

    # ------------------------- hyper parameters -------------------------
    speed_weight: float = 2.0           # main forward-speed reward
    speed_temperature: float = 4.0      # [m/s] tanh scale (keeps gradient at high speed)
    surge_weight: float = 1.0           # extra push for really fast running
    surge_threshold: float = 3.0        # [m/s] where the surge bonus starts
    surge_range: float = 4.0            # [m/s] linear ramp length of the surge bonus
    upright_weight: float = 1.0
    upright_temperature: float = 0.3    # sharper posture control
    height_weight: float = 0.5
    height_min: float = 0.30            # [m] below this the torso is considered collapsed
    height_range: float = 0.35          # [m] ramp up to full credit
    lateral_cost_coeff: float = -0.05   # straight running
    yaw_cost_coeff: float = -0.01       # no turning / spinning
    vertical_cost_coeff: float = -0.02
    ang_vel_cost_coeff: float = -0.005  # smooth torso motion
    dof_vel_cost_coeff: float = -0.0003
    action_cost_coeff: float = -0.02    # light: do not choke the gait

    # ------------------------- state extraction -------------------------
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]
    torso_z = root_states[:, 2]

    qw = root_states[:, 3]
    qx = root_states[:, 4]
    qy = root_states[:, 5]
    qz = root_states[:, 6]

    # world-z component of the body up axis (1.0 == perfectly upright)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    forward_speed = lin_vel[:, 0]

    # ------------------------- reward components -------------------------
    # 1) main forward speed (softly saturated, still ~0.14 grad/(m/s) at 5 m/s)
    speed_reward = speed_weight * torch.tanh(forward_speed / speed_temperature)

    # 2) linear "sprint" bonus: constant slope between 3 and 7 m/s
    excess = torch.clamp(forward_speed - surge_threshold, min=0.0)
    surge_reward = surge_weight * torch.clamp(excess / surge_range, min=0.0, max=1.0)

    # 3) posture: keep the torso upright (replaces the saturated 'alive' term)
    upright_reward = upright_weight * torch.exp((up_z - 1.0) / upright_temperature)

    # 4) keep the torso off the ground (prevents crawling / diving forward)
    height_reward = height_weight * torch.clamp((torso_z - height_min) / height_range,
                                                min=0.0, max=1.0)

    # 5) regularizers
    lateral_cost = lateral_cost_coeff * lin_vel[:, 1] * lin_vel[:, 1]
    yaw_cost = yaw_cost_coeff * ang_vel[:, 2] * ang_vel[:, 2]
    vertical_cost = vertical_cost_coeff * lin_vel[:, 2] * lin_vel[:, 2]
    ang_vel_cost = ang_vel_cost_coeff * torch.sum(ang_vel * ang_vel, dim=1)
    dof_vel_cost = dof_vel_cost_coeff * torch.sum(dof_vel * dof_vel, dim=1)
    action_cost = action_cost_coeff * torch.sum(actions * actions, dim=1)

    total = (speed_reward + surge_reward + upright_reward + height_reward +
             lateral_cost + yaw_cost + vertical_cost + ang_vel_cost +
             dof_vel_cost + action_cost)

    reward_dict = {"forward_speed": speed_reward}
    reward_dict["surge_speed"] = surge_reward
    reward_dict["upright"] = upright_reward
    reward_dict["torso_height"] = height_reward
    reward_dict["lateral_vel_cost"] = lateral_cost
    reward_dict["yaw_rate_cost"] = yaw_cost
    reward_dict["vertical_vel_cost"] = vertical_cost
    reward_dict["ang_vel_cost"] = ang_vel_cost
    reward_dict["dof_vel_cost"] = dof_vel_cost
    reward_dict["action_cost"] = action_cost

    return total, reward_dict
