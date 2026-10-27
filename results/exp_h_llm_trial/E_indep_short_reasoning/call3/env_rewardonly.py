@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    actions: torch.Tensor,
    dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- temperatures (for the exponential transforms) ----------------
    speed_temp: float = 4.0      # temperature of the (forward) speed reward
    upright_temp: float = 0.5   # temperature of the uprightness reward
    speed_mag_temp: float = 4.0 # temperature of the overall horizontal speed reward

    # ---------------- reward weights ----------------
    w_speed: float = 1.0        # run forward as fast as possible
    w_speed_mag: float = 0.2    # keep moving fast in the horizontal plane
    w_upright: float = 0.5      # stay on its legs
    w_lateral: float = 0.1      # run straight (no sideways slipping)
    w_yaw_rate: float = 0.02    # do not spin around
    w_action: float = 0.005     # energy / smoothness
    w_dof_vel: float = 1.0e-5   # joint velocity regularization
    w_alive: float = 0.1        # stay alive

    # root state decomposition (world frame)
    vel = root_states[:, 7:10]
    vel_xy = vel[:, 0:2]
    ang_vel = root_states[:, 10:13]

    # heading (forward) direction of the ant projected on the ground plane
    heading_xy = heading_vec[:, 0:2]
    heading_xy = heading_xy / torch.norm(heading_xy, dim=-1, keepdim=True).clamp(min=1.0e-6)

    # decompose the horizontal velocity into forward / lateral components
    forward_speed = torch.sum(vel_xy * heading_xy, dim=-1)
    lateral_speed = vel_xy[:, 0] * (-heading_xy[:, 1]) + vel_xy[:, 1] * heading_xy[:, 0]
    speed_mag = torch.norm(vel_xy, dim=-1)

    # uprightness: cosine of the tilt of the torso (up axis is z)
    up_norm = up_vec / torch.norm(up_vec, dim=-1, keepdim=True).clamp(min=1.0e-6)
    uprightness = up_norm[:, 2]

    # ---------------- individual reward components ----------------
    speed_reward = torch.exp(forward_speed / speed_temp) - 1.0
    speed_mag_reward = torch.exp(speed_mag / speed_mag_temp) - 1.0
    upright_reward = torch.exp((uprightness - 1.0) / upright_temp)
    lateral_penalty = lateral_speed * lateral_speed
    yaw_rate_penalty = ang_vel[:, 2] * ang_vel[:, 2]
    action_penalty = torch.sum(actions * actions, dim=-1)
    dof_vel_penalty = torch.sum(dof_vel * dof_vel, dim=-1)
    alive_reward = torch.ones_like(forward_speed)

    reward = (
        w_speed * speed_reward
        + w_speed_mag * speed_mag_reward
        + w_upright * upright_reward
        - w_lateral * lateral_penalty
        - w_yaw_rate * yaw_rate_penalty
        - w_action * action_penalty
        - w_dof_vel * dof_vel_penalty
        + w_alive * alive_reward
    )

    components = {
        "forward_speed": speed_reward,
        "speed_magnitude": speed_mag_reward,
        "upright": upright_reward,
        "lateral_penalty": lateral_penalty,
        "yaw_rate_penalty": yaw_rate_penalty,
        "action_penalty": action_penalty,
        "dof_vel_penalty": dof_vel_penalty,
        "alive": alive_reward,
    }

    return reward, components
