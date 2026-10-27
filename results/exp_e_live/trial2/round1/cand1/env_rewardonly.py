@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ------------------- hyper-parameters -------------------
    speed_weight: float = 1.0          # weight on forward running speed [m/s]
    speed_bonus_weight: float = 0.5    # bounded extra reward for high speeds
    speed_temperature: float = 3.0     # temperature of the tanh speed bonus
    alive_weight: float = 0.1          # small bonus for staying alive
    upright_weight: float = 0.2        # reward for keeping the torso up
    ctrl_weight: float = 0.005         # penalty on action magnitude
    lateral_weight: float = 0.02       # penalty on sideways drift
    vertical_weight: float = 0.01      # penalty on vertical bouncing
    eps: float = 1e-6

    # ------------------- torso state -------------------------
    quat = root_states[:, 3:7]
    lin_vel = root_states[:, 7:10]

    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + eps)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    # body -> world rotation matrix
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    forward = torch.stack([r00, r10, r20], dim=-1)   # body +x axis in world frame
    up = torch.stack([r02, r12, r22], dim=-1)        # body +z axis in world frame

    # ------------- forward running speed ---------------------
    fwd_h = forward[:, 0:2]
    fwd_dir = fwd_h / (torch.norm(fwd_h, dim=-1, keepdim=True) + eps)

    vel_h = lin_vel[:, 0:2]
    fwd_vel_scalar = torch.sum(vel_h * fwd_dir, dim=-1)
    forward_speed = fwd_vel_scalar                       # m/s along body forward
    horiz_speed = torch.norm(vel_h, dim=-1)               # total planar speed

    # sideways (lateral) component of the velocity
    lat_vel = vel_h - fwd_dir * torch.sum(vel_h * fwd_dir, dim=-1, keepdim=True)
    lateral_speed_sq = torch.sum(lat_vel * lat_vel, dim=-1)

    up_z = up[:, 2]                                      # 1 = perfectly upright
    vert_speed = lin_vel[:, 2]
    ctrl_cost = torch.sum(actions * actions, dim=-1)

    # ---------------- reward components -----------------------
    rew_speed = speed_weight * forward_speed
    rew_speed_bonus = speed_bonus_weight * torch.tanh(forward_speed / speed_temperature)
    rew_alive = alive_weight * torch.ones_like(forward_speed)
    rew_upright = upright_weight * up_z

    pen_ctrl = ctrl_weight * ctrl_cost
    pen_lateral = lateral_weight * lateral_speed_sq
    pen_vertical = vertical_weight * torch.abs(vert_speed)

    total = (rew_speed + rew_speed_bonus + rew_alive + rew_upright
             - pen_ctrl - pen_lateral - pen_vertical)

    components: Dict[str, torch.Tensor] = {
        "forward_speed": rew_speed,
        "forward_speed_bonus": rew_speed_bonus,
        "alive_bonus": rew_alive,
        "upright": rew_upright,
        "control_cost": -pen_ctrl,
        "lateral_drift": -pen_lateral,
        "vertical_bounce": -pen_vertical,
        "forward_vel_raw": forward_speed,
        "horizontal_speed": horiz_speed,
    }

    return total, components
