@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,
    ball_linvels: torch.Tensor,
    sensor_forces: torch.Tensor,
    sensor_torques: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Keep the ball on the table top (contact-sensor aware alive bonus)."""
    force_scale: float = 5.0        # [N]   contact force scale
    torque_scale: float = 0.5       # [Nm]  contact torque scale
    vel_scale: float = 0.40         # [m/s] ball speed scale
    centre_scale: float = 0.30      # [m]   radial distance scale
    fall_scale: float = 0.10        # [m]   height-drop scale

    w_alive: float = 3.0
    w_still: float = 1.5
    w_centre: float = 0.5
    w_fall: float = 3.0

    ball_xy = ball_positions[:, 0:2]
    ball_z = ball_positions[:, 2]
    vel = ball_linvels
    n_env = ball_z.shape[0]

    # ---- contact wrench produced by the ball resting on the table -------
    f_norm = torch.norm(sensor_forces, dim=-1)
    t_norm = torch.norm(sensor_torques, dim=-1)
    f_sum = torch.sum(f_norm, dim=-1)
    t_sum = torch.sum(t_norm, dim=-1)

    contact = torch.tanh(f_sum / force_scale) + 0.25 * torch.tanh(t_sum / torque_scale)
    contact = torch.clamp(contact, min=0.0, max=1.0)

    # ---- relative height (table level ~ median ball height) -------------
    z_sorted = torch.sort(ball_z).values
    z_ref = z_sorted[n_env // 2]
    drop = torch.clamp(z_ref - ball_z, min=0.0)
    a_drop = drop / fall_scale
    level = 1.0 - torch.clamp(a_drop, min=0.0, max=1.0)

    # ball is "alive" on the table if it is in contact AND at table level
    alive = 0.5 * contact + 0.5 * level

    # ---- the ball should be (almost) motionless -------------------------
    speed = torch.norm(vel, dim=-1)
    a_v = speed / vel_scale
    r_still = torch.exp(-(a_v * a_v))

    # ---- and close to the centre of the table --------------------------
    dist_centre = torch.norm(ball_xy, dim=-1)
    a_c = dist_centre / centre_scale
    r_centre = torch.exp(-(a_c * a_c))

    # ---- penalise downward motion (leaving the table) ------------------
    fall_speed = torch.clamp(-vel[:, 2], min=0.0)
    a_f = fall_speed / vel_scale
    r_fallv = -torch.clamp(a_f, min=0.0, max=1.0)

    total = (
        w_alive * alive
        + w_still * r_still
        + w_centre * r_centre
        + w_fall * r_fallv
    )

    components: Dict[str, torch.Tensor] = {
        "alive_on_table": w_alive * alive,
        "still": w_still * r_still,
        "centre": w_centre * r_centre,
        "fall_velocity": w_fall * r_fallv,
    }
    return total, components
