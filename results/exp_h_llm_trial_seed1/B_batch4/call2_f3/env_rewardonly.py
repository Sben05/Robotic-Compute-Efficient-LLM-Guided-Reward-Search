@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Variant 3 - tanh-shaped speed with body-frame velocity decomposition.
    The velocity is rotated into the torso frame so that the ant is rewarded for
    running in the direction it is facing (and penalised for strafing /
    bouncing).  The world-frame forward speed is shaped with tanh so that very
    fast speeds do not dominate the gradient.
    """
    speed_temp: float = 3.0      # temperature of the tanh speed shaping
    body_coeff: float = 0.5      # weight of the body-frame forward speed
    lat_coeff: float = 0.3       # lateral (sideways) velocity penalty
    vert_coeff: float = 0.1      # vertical bouncing penalty
    upright_coeff: float = 0.5
    ctrl_coeff: float = 0.005

    qw = root_states[:, 3]
    qx = root_states[:, 4]
    qy = root_states[:, 5]
    qz = root_states[:, 6]
    v = root_states[:, 7:10]
    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]

    # world velocity -> body frame velocity (transpose of the rotation matrix)
    b_x = ((1.0 - 2.0 * (qy * qy + qz * qz)) * vx
           + 2.0 * (qx * qy + qw * qz) * vy
           + 2.0 * (qx * qz - qw * qy) * vz)
    b_y = (2.0 * (qx * qy - qw * qz) * vx
           + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy
           + 2.0 * (qy * qz + qw * qx) * vz)
    b_z = (2.0 * (qx * qz + qw * qy) * vx
           + 2.0 * (qy * qz - qw * qx) * vy
           + (1.0 - 2.0 * (qx * qx + qy * qy)) * vz)

    world_speed = torch.tanh(torch.clamp(vx, min=0.0, max=10.0) / speed_temp)
    body_speed = torch.tanh(torch.clamp(b_x, min=0.0, max=10.0) / speed_temp)

    lat_pen = torch.square(b_y)
    vert_pen = torch.square(b_z)

    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright = torch.clamp(up_z, min=0.0, max=1.0)

    ctrl_pen = torch.sum(torch.square(actions), dim=-1)

    total = (world_speed
             + body_coeff * body_speed
             + upright_coeff * upright
             - lat_coeff * lat_pen
             - vert_coeff * vert_pen
             - ctrl_coeff * ctrl_pen)
    return total, {"world_speed": world_speed, "body_speed": body_speed, "upright": upright,
                   "lat_pen": lat_pen, "vert_pen": vert_pen, "ctrl_pen": ctrl_pen}
