@torch.jit.script
def compute_reward(ball_positions: torch.Tensor,
                     ball_linvels: torch.Tensor,
                     sensor_forces: torch.Tensor,
                     sensor_torques: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Contact / sensor based reward.

    The F/T sensor mounted on the table measures the load that is transmitted by the
    ball.  While the ball is supported by the table this load is clearly non-zero,
    when the ball leaves the table the load disappears.  The agent is therefore paid
    for keeping the ball in contact with the table, for keeping it at rest and over
    the middle of the table, and for moving the table gently (small joint torques).
    """
    # ------------------- tunables (constants, not env inputs) -------------------
    sigma_force: float = 1.0           # temperature of the contact (load) term [N]
    sigma_v: float = 0.50              # temperature of the ball velocity term
    sigma_xy: float = 0.40             # temperature of the centring term
    sigma_torque: float = 2.0          # temperature of the table torque term
    w_contact: float = 1.5
    w_still: float = 1.0
    w_center: float = 0.5
    w_smooth: float = 0.3
    floor_z: float = 0.30
    w_fall: float = 3.0
    eps: float = 1e-8
    # -----------------------------------------------------------------------------

    px = ball_positions[:, 0]
    py = ball_positions[:, 1]
    pz = ball_positions[:, 2]
    vx = ball_linvels[:, 0]
    vy = ball_linvels[:, 1]
    vz = ball_linvels[:, 2]

    # --- contact load measured by the force sensor (first sensor, 3 components) ---
    f = sensor_forces[:, 0, :]
    fx = f[:, 0]
    fy = f[:, 1]
    fz = f[:, 2]
    f_mag = torch.sqrt(fx * fx + fy * fy + fz * fz + eps)
    r_contact = 1.0 - torch.exp(-f_mag / sigma_force)

    # --- joint torques: penalise aggressive tilting of the table ---
    t = sensor_torques[:, 0, :]
    tx = t[:, 0]
    ty = t[:, 1]
    tz = t[:, 2]
    t_mag = torch.sqrt(tx * tx + ty * ty + tz * tz + eps)
    r_smooth = torch.exp(-t_mag / sigma_torque)

    # --- ball dynamics ---
    r_still = torch.exp(-(vx * vx + vy * vy + vz * vz) / (2.0 * sigma_v * sigma_v))
    r_center = torch.exp(-(px * px + py * py) / (2.0 * sigma_xy * sigma_xy))

    fall = torch.clamp((floor_z - pz) / floor_z, min=0.0, max=1.0)
    r_fall = -w_fall * fall

    reward = (w_contact * r_contact + w_still * r_still + w_center * r_center
              + w_smooth * r_smooth + r_fall)

    components: Dict[str, torch.Tensor] = {}
    components["contact"] = r_contact
    components["still"] = r_still
    components["center"] = r_center
    components["smooth"] = r_smooth
    components["fall"] = r_fall
    return reward, components
