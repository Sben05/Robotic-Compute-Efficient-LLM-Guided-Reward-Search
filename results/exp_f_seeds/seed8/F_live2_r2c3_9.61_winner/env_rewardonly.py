@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dof_vel: torch.Tensor
                   ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # ----------------------------- hyper-parameters -----------------------------
    speed_temperature: float = 3.0        # m/s, tanh scale of the forward-speed bonus
    speed_weight: float = 2.0
    linear_speed_weight: float = 0.2      # small constant push for ever faster running
    tracking_target: float = 9.0          # m/s, speed the ant is asked to reach
    tracking_sigma: float = 3.5           # m/s, width of the tracking bonus
    tracking_weight: float = 2.0
    facing_temperature: float = 3.0       # m/s, tanh scale of the facing-direction bonus
    facing_weight: float = 0.8
    align_temperature: float = 1.0
    align_weight: float = 0.3
    tilt_threshold: float = 0.6           # cos(torso tilt); below -> too tilted
    tilt_weight: float = 0.6
    lateral_threshold: float = 0.5        # m/s, |vy| above this is penalised
    lateral_weight: float = 0.3
    alive_weight: float = 0.2
    action_cost_scale: float = 0.01
    joint_vel_cost_scale: float = 0.001
    ang_vel_cost_scale: float = 0.005
    fall_height: float = 0.35             # m, torso height below this = fallen
    fall_weight: float = 1.0

    n_envs = root_states.shape[0]
    device = root_states.device
    dtype = root_states.dtype

    # ------------------------------ state extraction -----------------------------
    quat = root_states[:, 3:7]            # (x, y, z, w)
    vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]
    pos_z = root_states[:, 2]

    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    # torso forward axis (+x of the body) expressed in the world frame
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    # torso up axis (+z of the body) component along world z
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # horizontal (xy) projection of the torso forward axis
    horiz = torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + 1.0e-8)
    fwd_hx = fwd_x / horiz
    fwd_hy = fwd_y / horiz

    vx = vel[:, 0]
    vy = vel[:, 1]
    facing_speed = vx * fwd_hx + vy * fwd_hy

    # ------------------------------ reward components ---------------------------
    # 1) saturating forward-speed bonus (gradient survives up to ~8 m/s)
    forward_speed = speed_weight * torch.tanh(vx / speed_temperature)
    # 2) small linear nudge so the policy keeps pushing for a higher top speed
    speed_linear = linear_speed_weight * vx
    # 3) tracking bonus towards a high target speed (persistent gradient)
    speed_err = (vx - tracking_target) / tracking_sigma
    speed_tracking = tracking_weight * torch.exp(-speed_err * speed_err)
    # 4) speed along the direction the body is actually facing
    facing = facing_weight * torch.tanh(facing_speed / facing_temperature)
    # 5) small bonus for heading roughly along +x
    heading_align = align_weight * torch.tanh(fwd_hx / align_temperature)
    # 6) hinge penalty: only when the torso is really tilted
    tilt_penalty = -tilt_weight * torch.clamp(tilt_threshold - up_z, min=0.0)
    # 7) hinge penalty: only when the ant drifts sideways noticeably
    lateral_drift = -lateral_weight * torch.clamp(torch.abs(vy) - lateral_threshold, min=0.0)
    # 8) small survival bonus
    alive = alive_weight * torch.ones(n_envs, device=device, dtype=dtype)
    # 9) regularisers
    action_cost = -action_cost_scale * torch.sum(actions * actions, dim=-1)
    joint_vel_cost = -joint_vel_cost_scale * torch.mean(dof_vel * dof_vel, dim=-1)
    ang_vel_cost = -ang_vel_cost_scale * torch.sum(ang_vel[:, 0:2] * ang_vel[:, 0:2], dim=-1)
    # 10) fall / collapse penalty
    fall_penalty = -fall_weight * torch.clamp(fall_height - pos_z, min=0.0)

    reward = (forward_speed + speed_linear + speed_tracking + facing + heading_align
              + tilt_penalty + lateral_drift + alive + action_cost + joint_vel_cost
              + ang_vel_cost + fall_penalty)

    components: Dict[str, torch.Tensor] = {
        "forward_speed": forward_speed,
        "speed_linear": speed_linear,
        "speed_tracking": speed_tracking,
        "facing_speed": facing,
        "heading_align": heading_align,
        "tilt_penalty": tilt_penalty,
        "lateral_drift": lateral_drift,
        "alive": alive,
        "action_cost": action_cost,
        "joint_vel_cost": joint_vel_cost,
        "ang_vel_cost": ang_vel_cost,
        "fall_penalty": fall_penalty,
    }

    return reward, components
