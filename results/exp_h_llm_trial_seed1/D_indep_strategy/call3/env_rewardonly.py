@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float,
                   potentials: torch.Tensor, prev_potentials: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # ------------------------- hyper-parameters (named variables) -------------------------
    target_speed: float = 4.0        # desired forward speed [m/s]
    speed_temp: float = 1.0          # temperature of the speed shaping
    heading_temp: float = 0.25       # temperature of the heading shaping
    upright_temp: float = 0.5        # temperature of the upright shaping
    progress_temp: float = 1.0       # temperature (speed scale) of the progress shaping
    bonus_speed_cap: float = 8.0     # speed at which the linear speed bonus saturates

    w_speed: float = 1.0             # weight of the exponential speed shaping
    w_speed_bonus: float = 0.5       # weight of the linear "go faster" bonus
    w_heading: float = 0.5           # weight of the heading alignment
    w_upright: float = 0.25          # weight of the upright (not flipped) term
    w_progress: float = 0.5          # weight of the potential-based progress term
    w_alive: float = 0.25            # alive bonus
    w_action: float = 0.01           # small action-magnitude penalty

    eps: float = 1e-6

    # ------------------------------- state extraction -------------------------------
    pos = root_states[:, 0:3]
    quat = root_states[:, 3:7]
    lin_vel = root_states[:, 7:10]

    # torso forward axis (body +x) expressed in the world frame  (quaternion is w,x,y,z)
    q_w = quat[:, 0]
    q_x = quat[:, 1]
    q_y = quat[:, 2]
    q_z = quat[:, 3]

    fwd_x = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    fwd_y = 2.0 * (q_x * q_y + q_w * q_z)
    fwd_z = 2.0 * (q_x * q_z - q_w * q_y)

    fwd_xy = torch.stack((fwd_x, fwd_y), dim=-1)
    fwd_xy = fwd_xy / torch.clamp(torch.norm(fwd_xy, dim=-1, keepdim=True), min=eps)

    # torso up axis (body +z) z-component in the world frame (1 = upright, -1 = upside down)
    up_z = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)

    # horizontal direction that defines "forward" (towards the goal)
    to_goal = targets[:, 0:2] - pos[:, 0:2]
    goal_xy = to_goal / torch.clamp(torch.norm(to_goal, dim=-1, keepdim=True), min=eps)

    # forward speed along the goal direction
    speed = lin_vel[:, 0] * goal_xy[:, 0] + lin_vel[:, 1] * goal_xy[:, 1]

    # --------------------------------- components ---------------------------------
    # 1) exponential shaping of the gap w.r.t. the target forward speed
    speed_err = torch.abs(speed - target_speed)
    r_speed = torch.exp(-speed_err / speed_temp)

    # 2) linear bonus so that the ant keeps pushing beyond the target speed
    r_speed_bonus = torch.clamp(speed, min=0.0, max=bonus_speed_cap) / bonus_speed_cap

    # 3) heading alignment between the torso forward axis and the goal direction
    cos_align = fwd_xy[:, 0] * goal_xy[:, 0] + fwd_xy[:, 1] * goal_xy[:, 1]
    r_heading = torch.exp((cos_align - 1.0) / heading_temp)

    # 4) stay on the feet (do not flip over while sprinting)
    r_upright = torch.exp((up_z - 1.0) / upright_temp)

    # 5) potential based progress: potentials = -||to_goal||/dt  ->  difference is speed gain
    progress_m = (potentials - prev_potentials) * dt          # metres closed to the goal
    r_progress = 1.0 - torch.exp(-torch.clamp(progress_m, min=0.0) / (progress_temp * dt))

    # 6) alive bonus
    r_alive = torch.ones_like(speed)

    # 7) small action magnitude penalty
    r_action = -torch.clamp(torch.mean(torch.abs(actions), dim=-1), max=1.0)

    total = (w_speed * r_speed
             + w_speed_bonus * r_speed_bonus
             + w_heading * r_heading
             + w_upright * r_upright
             + w_progress * r_progress
             + w_alive * r_alive
             + w_action * r_action)

    components: Dict[str, torch.Tensor] = {
        "speed": r_speed,
        "speed_bonus": r_speed_bonus,
        "heading": r_heading,
        "upright": r_upright,
        "progress": r_progress,
        "alive": r_alive,
        "action_penalty": r_action,
    }

    return total, components
