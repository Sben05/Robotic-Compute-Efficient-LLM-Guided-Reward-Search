@torch.jit.script
def compute_reward(ball_positions: torch.Tensor,
                     ball_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Shaped (exponential) reward -- "keep the ball quiet and on the table".

    Three dense, bounded components:
      * centring : the ball stays above the middle of the table,
      * height   : the ball stays at (or above) table-top height,
      * stillness: the ball is not rolling / bouncing,
    plus a hard penalty as soon as the ball has clearly left the table (near the floor).

    The height term is one sided (only penalised when the ball is *below* the target
    height) so that a slightly mis-calibrated table height only adds a constant offset
    and never rewards throwing the ball into the air.
    """
    # ------------------- tunables (constants, not env inputs) -------------------
    table_center_x: float = 0.0      # the table is spawned at the world origin
    table_center_y: float = 0.0
    target_ball_z: float = 0.85       # ball centre height when it rests on the table
    floor_z: float = 0.30             # below this the ball is certainly off the table
    sigma_xy: float = 0.40            # temperature of the centring term
    sigma_z: float = 0.15             # temperature of the height term
    sigma_v: float = 0.50             # temperature of the velocity term
    w_center: float = 1.0
    w_height: float = 1.0
    w_still: float = 1.0
    w_fall: float = 3.0
    # -----------------------------------------------------------------------------

    px = ball_positions[:, 0]
    py = ball_positions[:, 1]
    pz = ball_positions[:, 2]
    vx = ball_linvels[:, 0]
    vy = ball_linvels[:, 1]
    vz = ball_linvels[:, 2]

    dx = px - table_center_x
    dy = py - table_center_y

    r_center = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma_xy * sigma_xy))

    dz = torch.clamp(target_ball_z - pz, min=0.0)
    r_height = torch.exp(-(dz * dz) / (2.0 * sigma_z * sigma_z))

    r_still = torch.exp(-(vx * vx + vy * vy + vz * vz) / (2.0 * sigma_v * sigma_v))

    fall = torch.clamp((floor_z - pz) / floor_z, min=0.0, max=1.0)
    r_fall = -w_fall * fall

    reward = w_center * r_center + w_height * r_height + w_still * r_still + r_fall

    components: Dict[str, torch.Tensor] = {}
    components["center"] = r_center
    components["height"] = r_height
    components["still"] = r_still
    components["fall"] = r_fall
    return reward, components
