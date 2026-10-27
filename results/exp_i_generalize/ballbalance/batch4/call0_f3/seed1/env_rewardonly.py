@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,
    ball_linvels: torch.Tensor,
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Keep the ball on the table top (position centric + posture regularization)."""
    level_scale: float = 0.06       # [m]    ball height deviation scale
    centre_scale: float = 0.30      # [m]    radial distance scale
    vel_scale: float = 0.40         # [m/s]  ball speed scale
    joint_scale: float = 5.0        # [rad/s] joint speed scale
    posture_scale: float = 1.0      # [rad]  nominal joint pose scale

    w_level: float = 2.0
    w_centre: float = 1.0
    w_still: float = 1.5
    w_posture: float = 0.05
    w_effort: float = 0.02

    ball_xy = ball_positions[:, 0:2]
    ball_z = ball_positions[:, 2]
    vel = ball_linvels
    n_env = ball_z.shape[0]

    # ---- ball must stay at the table level (not sunk, not tossed) ------
    z_sorted = torch.sort(ball_z).values
    z_ref = z_sorted[n_env // 2]
    dz = (ball_z - z_ref) / level_scale
    r_level = torch.exp(-(dz * dz))

    # ---- ball must stay close to the middle of the table ---------------
    dist_centre = torch.norm(ball_xy, dim=-1)
    a_c = dist_centre / centre_scale
    r_centre = torch.exp(-(a_c * a_c))

    # ---- ball must be calm ---------------------------------------------
    speed = torch.norm(vel, dim=-1)
    a_v = speed / vel_scale
    r_still = torch.exp(-(a_v * a_v))

    # ---- keep the manipulator close to its nominal pose ----------------
    p_sq = torch.sum(dof_positions * dof_positions, dim=-1)
    a_p = p_sq / (posture_scale * posture_scale)
    r_posture = -a_p

    # ---- control effort / smoothness -----------------------------------
    v_sq = torch.sum(dof_velocities * dof_velocities, dim=-1)
    a_j = v_sq / (joint_scale * joint_scale)
    r_effort = -a_j

    total = (
        w_level * r_level
        + w_centre * r_centre
        + w_still * r_still
        + w_posture * r_posture
        + w_effort * r_effort
    )

    components: Dict[str, torch.Tensor] = {
        "table_level": w_level * r_level,
        "centre": w_centre * r_centre,
        "still": w_still * r_still,
        "posture": w_posture * r_posture,
        "effort": w_effort * r_effort,
    }
    return total, components
