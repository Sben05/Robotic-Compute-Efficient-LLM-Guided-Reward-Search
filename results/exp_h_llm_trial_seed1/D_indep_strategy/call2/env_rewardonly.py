@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   targets: torch.Tensor,
                   potentials: torch.Tensor,
                   prev_potentials: torch.Tensor,
                   actions: torch.Tensor,
                   dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward for making the Ant sprint forward as fast as possible.

    root_states : [N, 13] -> pos(0:3), quat xyzw(3:7), lin_vel(7:10), ang_vel(10:13)
    targets     : [N, 3]  world-frame target position
    potentials  : [N]     potential of the current state  (-||to_target||/dt)
    prev_potentials : [N] potential of the previous state
    actions     : [N, A]  last applied actions
    dt          : control time step

    Main term  : potential difference (distance closed towards the target this step)
    Extra terms: forward speed (towards target / along the torso heading),
                 survival (alive + upright), posture (heading alignment),
                 small action-magnitude penalty for smooth gaits.
    """

    # ------------------------------------------------------------------ weights
    w_progress = 15.0   # per meter of distance closed towards the target
    w_speed = 1.0       # per m/s of speed along the target direction
    w_facing_speed = 0.5  # per m/s of speed along the torso heading (real "running")
    w_alive = 1.0       # survival bonus
    w_upright = 0.5     # keeping the torso vertical
    w_align = 0.3       # facing the direction of travel
    w_xspeed = 0.2      # world-frame forward speed
    w_action = 0.005    # action magnitude penalty

    # ------------------------------------------------------- state extraction
    pos = root_states[:, 0:3]
    quat = root_states[:, 3:7]
    lin_vel = root_states[:, 7:10]

    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    # torso local +x axis expressed in the world frame (z-up) -> heading
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_xy = torch.stack([fwd_x, fwd_y], dim=-1)
    fwd_len = torch.norm(fwd_xy, dim=-1, keepdim=True)
    fwd_unit = fwd_xy / torch.clamp(fwd_len, 1e-6, 1e6)

    # z-component of the torso up axis -> uprightness in [0, 1]
    upright = torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), 0.0, 1.0)

    vel_xy = lin_vel[:, 0:2]

    # -------------------------------------------------- target direction (xy)
    to_target = targets - pos
    to_target[:, 2] = 0.0
    dist = torch.norm(to_target, dim=-1, keepdim=True)
    unit_target = to_target / torch.clamp(dist, 1e-6, 1e6)

    # speed components
    speed_to_target = torch.sum(vel_xy * unit_target[:, 0:2], dim=-1)
    speed_facing = torch.sum(vel_xy * fwd_unit, dim=-1)
    x_speed = lin_vel[:, 0]

    # heading / travel-direction alignment (cosine)
    alignment = torch.clamp(torch.sum(fwd_unit * unit_target[:, 0:2], dim=-1), -1.0, 1.0)

    # --------------------------------------- potential difference (progress)
    # potentials = -||to_target||/dt  =>  (potentials - prev_potentials)*dt
    # is the distance (in meters) closed towards the target during this step.
    progress = torch.clamp((potentials - prev_potentials) * dt, -0.5, 0.5)

    alive = torch.ones_like(progress)
    action_cost = torch.mean(actions * actions, dim=-1)

    # ------------------------------------------------------- reward components
    r_progress = w_progress * progress
    r_speed = w_speed * speed_to_target
    r_facing_speed = w_facing_speed * speed_facing
    r_alive = w_alive * alive
    r_upright = w_upright * upright
    r_align = w_align * alignment
    r_xspeed = w_xspeed * x_speed
    r_action = -w_action * action_cost

    reward = (r_progress + r_speed + r_facing_speed + r_alive +
              r_upright + r_align + r_xspeed + r_action)

    components = {
        "progress": r_progress,
        "forward_speed": r_speed,
        "facing_speed": r_facing_speed,
        "alive": r_alive,
        "upright": r_upright,
        "alignment": r_align,
        "x_speed": r_xspeed,
        "action_penalty": r_action,
    }

    return reward, components
