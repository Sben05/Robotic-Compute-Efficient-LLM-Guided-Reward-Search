@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- temperature / weight parameters ----------------
    speed_temp: float = 4.0          # m/s scale for the forward-speed tanh
    align_temp: float = 1.5          # sharpness of the heading-alignment term
    upright_temp: float = 1.5        # sharpness of the upright term
    lateral_temp: float = 1.5        # sharpness of the lateral-drift penalty

    w_speed: float = 2.0
    w_align: float = 0.5
    w_upright: float = 0.5
    w_alive: float = 0.2
    w_lateral: float = 0.2
    ctrl_weight: float = 0.02
    vel_weight: float = 5.0e-4

    # ---------------- torso orientation (quaternion w,x,y,z) ----------------
    q = root_states[:, 3:7]
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # body forward (x) axis expressed in world frame
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    # body up (z) axis expressed in world frame (z component)
    up_z = 1.0 - 2.0 * (x * x + y * y)

    # ---------------- velocities ----------------
    vel = root_states[:, 7:10]
    vx = vel[:, 0]      # world forward (+x) speed
    vy = vel[:, 1]      # lateral speed

    # ---------------- reward components ----------------
    forward_speed = torch.tanh(vx / speed_temp)              # run forward, fast
    heading_align = torch.tanh(fwd_x / align_temp)           # face the running direction
    upright = torch.tanh(up_z / upright_temp)                # do not flip over
    lateral = torch.tanh(torch.abs(vy) / lateral_temp)       # keep running straight
    alive = torch.ones_like(vx)                              # stay alive

    ctrl_cost = torch.sum(torch.square(actions), dim=-1)     # smooth / small torques
    vel_cost = torch.sum(torch.square(dof_vel), dim=-1)       # avoid crazy joint speeds

    reward = (w_speed * forward_speed
              + w_align * heading_align
              + w_upright * upright
              + w_alive * alive
              - w_lateral * lateral
              - ctrl_weight * ctrl_cost
              - vel_weight * vel_cost)

    components = {
        "forward_speed": forward_speed,
        "heading_align": heading_align,
        "upright": upright,
        "lateral": lateral,
        "alive": alive,
        "ctrl_cost": ctrl_cost,
        "vel_cost": vel_cost,
    }
    return reward, components
