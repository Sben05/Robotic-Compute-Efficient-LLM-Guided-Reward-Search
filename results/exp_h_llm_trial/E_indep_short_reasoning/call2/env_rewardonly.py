@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    actions: torch.Tensor,
    dof_vel: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- named parameters (not inputs) ----------------
    forward_speed_scale: float = 3.0    # m/s at which the normalized speed reward saturates
    world_speed_scale: float = 3.0      # m/s at which the normalized world speed reward saturates
    speed_temperature: float = 2.0      # temperature of the exponential speed shaping
    upright_temperature: float = 0.3    # temperature of the exponential upright shaping
    progress_scale: float = 0.02       # meters of forward progress at 1 m/s (per step)
    max_speed: float = 8.0             # clip of the exponential speed argument

    w_alive: float = 0.5
    w_body_forward: float = 1.5
    w_world_forward: float = 0.8
    w_speed_exp: float = 0.05
    w_progress: float = 0.5
    w_upright: float = 0.3
    w_lateral: float = 0.05
    w_action: float = 0.01
    w_joint_vel: float = 1.0e-4

    num_envs = root_states.shape[0]
    device = root_states.device

    # ---------------- body frame from the root quaternion (w, x, y, z) ----
    quat = root_states[:, 3:7]
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]

    # body x-axis (forward) expressed in world frame
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    # body z-axis (up) expressed in world frame
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # world frame linear velocity of the torso
    lin_vel = root_states[:, 7:10]
    vel_x = lin_vel[:, 0]
    vel_y = lin_vel[:, 1]

    # horizontal unit forward direction
    fwd_norm = torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + 1.0e-8)
    fwd_x_h = fwd_x / fwd_norm
    fwd_y_h = fwd_y / fwd_norm

    # speed along the direction the ant is facing
    body_forward_speed = vel_x * fwd_x_h + vel_y * fwd_y_h
    # speed along the world forward (+x) axis
    world_forward_speed = vel_x
    # sideways slipping speed
    lateral_speed = -vel_x * fwd_y_h + vel_y * fwd_x_h

    # ---------------- reward components ----------------
    body_forward_reward = torch.clamp(body_forward_speed / forward_speed_scale, min=-1.0, max=1.0)
    world_forward_reward = torch.clamp(world_forward_speed / world_speed_scale, min=-1.0, max=1.0)

    speed_exp_reward = torch.exp(
        torch.clamp(body_forward_speed, min=-2.0, max=max_speed) / speed_temperature
    )

    forward_progress = body_forward_speed * dt
    progress_reward = torch.clamp(forward_progress / progress_scale, min=-1.0, max=1.0)

    tilt = torch.clamp(1.0 - up_z, min=0.0, max=2.0)
    upright_reward = torch.exp(-tilt / upright_temperature)

    alive_reward = torch.full((num_envs,), w_alive, dtype=torch.float32, device=device)

    lateral_penalty = torch.square(lateral_speed)
    action_penalty = torch.sum(torch.square(actions), dim=-1)
    joint_vel_penalty = torch.sum(torch.square(dof_vel), dim=-1)

    total_reward = (
        alive_reward
        + w_body_forward * body_forward_reward
        + w_world_forward * world_forward_reward
        + w_speed_exp * speed_exp_reward
        + w_progress * progress_reward
        + w_upright * upright_reward
        - w_lateral * lateral_penalty
        - w_action * action_penalty
        - w_joint_vel * joint_vel_penalty
    )

    components: Dict[str, torch.Tensor] = {
        "alive": alive_reward,
        "body_forward_speed": body_forward_reward,
        "world_forward_speed": world_forward_reward,
        "speed_exp": speed_exp_reward,
        "forward_progress": progress_reward,
        "upright": upright_reward,
        "lateral_penalty": lateral_penalty,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty,
    }

    return total_reward, components
