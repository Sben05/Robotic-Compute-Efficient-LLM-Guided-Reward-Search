@torch.jit.script
def compute_reward(ball_positions: torch.Tensor,
                     ball_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Classic quadratic-penalty reward (Isaac-Gym style) with an alive bonus.

    Only very little height information is used (a single low "floor" threshold), so the
    reward is robust to an inaccurate knowledge of the table height: as long as the ball
    is on the table the agent gets its alive bonus and only has to suppress the motion
    of the ball; if the ball rolls off and hits the ground a large penalty is paid.
    """
    # ------------------- tunables (constants, not env inputs) -------------------
    table_center_x: float = 0.0
    table_center_y: float = 0.0
    target_ball_z: float = 0.85       # nominal resting height of the ball
    floor_z: float = 0.30             # ball below this -> it has left the table
    w_alive: float = 1.0
    w_speed: float = 1.0              # horizontal kinetic energy
    w_vz: float = 0.5                 # vertical kinetic energy
    w_center: float = 0.5             # distance to the middle of the table
    w_height: float = 0.2             # deviation from the nominal table height
    w_fall: float = 10.0
    v_sq_max: float = 25.0            # clipping keeps the penalty bounded
    # -----------------------------------------------------------------------------

    px = ball_positions[:, 0]
    py = ball_positions[:, 1]
    pz = ball_positions[:, 2]
    vx = ball_linvels[:, 0]
    vy = ball_linvels[:, 1]
    vz = ball_linvels[:, 2]

    dx = px - table_center_x
    dy = py - table_center_y

    v_sq = torch.clamp(vx * vx + vy * vy, min=0.0, max=v_sq_max)
    r_speed = -w_speed * v_sq

    vz_sq = torch.clamp(vz * vz, min=0.0, max=v_sq_max)
    r_vz = -w_vz * vz_sq

    r_center = -w_center * (dx * dx + dy * dy)

    dz = pz - target_ball_z
    r_height = -w_height * dz * dz

    fall = torch.clamp((floor_z - pz) / floor_z, min=0.0, max=1.0)
    r_fall = -w_fall * fall

    reward = w_alive + r_speed + r_vz + r_center + r_height + r_fall

    components: Dict[str, torch.Tensor] = {}
    components["alive"] = torch.full_like(pz, w_alive)
    components["speed"] = r_speed
    components["vz"] = r_vz
    components["center"] = r_center
    components["height"] = r_height
    components["fall"] = r_fall
    return reward, components
