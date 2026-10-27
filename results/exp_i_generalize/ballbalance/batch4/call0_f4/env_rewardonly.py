@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,
    ball_linvels: torch.Tensor,
    sensor_forces: torch.Tensor,
    dof_velocities: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Keep the ball on the table top (contact x height alive bonus + shaping)."""
    force_scale: float = 5.0        # [N]   contact force scale
    level_scale: float = 0.08       # [m]   ball height deviation scale
    vel_scale: float = 0.35         # [m/s] ball speed scale
    joint_scale: float = 4.0        # [rad/s] joint speed scale

    w_alive: float = 2.0
    w_level: float = 1.0
    w_calm: float = 1.5
    w_fall: float = 2.0
    w_effort: float = 0.02

    ball_z = ball_positions[:, 2]
    vel = ball_linvels
    n_env = ball_z.shape[0]

    # ---- contact between ball and table --------------------------------
    f_norm = torch.norm(sensor_forces, dim=-1)
    f_sum = torch.sum(f_norm, dim=-1)
    contact = torch.clamp(torch.tanh(f_sum / force_scale), min=0.0, max=1.0)

    # ---- height of the ball relative to the table level ----------------
    z_sorted = torch.sort(ball_z).values
    z_ref = z_sorted[n_env // 2]
    drop = torch.clamp(z_ref - ball_z, min=0.0)
    a_drop = drop / level_scale
    level = 1.0 - torch.clamp(a_drop, min=0.0, max=1.0)

    # ---- alive: touching the table AND still at table level -------------
    alive = contact * level

    # ---- the ball should be calm (small kinetic energy) ----------------
    speed = torch.norm(vel, dim=-1)
    a_v = speed / vel_scale
    r_calm = torch.exp(-(a_v * a_v))

    # ---- penalise downward motion --------------------------------------
    fall_speed = torch.clamp(-vel[:, 2], min=0.0)
    a_f = fall_speed / vel_scale
    r_fallv = -torch.clamp(a_f, min=0.0, max=1.0)

    # ---- control effort -------------------------------------------------
    v_sq = torch.sum(dof_velocities * dof_velocities, dim=-1)
    a_j = v_sq / (joint_scale * joint_scale)
    r_effort = -a_j

    total = (
        w_alive * alive
        + w_level * level
        + w_calm * r_calm
        + w_fall * r_fallv
        + w_effort * r_effort
    )

    components: Dict[str, torch.Tensor] = {
        "alive": w_alive * alive,
        "table_level": w_level * level,
        "calm": w_calm * r_calm,
        "fall_velocity": w_fall * r_fallv,
        "effort": w_effort * r_effort,
    }
    return total, components
