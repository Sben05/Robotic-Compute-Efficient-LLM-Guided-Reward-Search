@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,
    ball_linvels: torch.Tensor,
    dof_velocities: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Keep the ball on the table top (geometry free, velocity based)."""
    # ------------------------- hyper parameters -------------------------
    vel_xy_scale: float = 0.30      # [m/s]   horizontal speed scale
    vel_z_scale: float = 0.40       # [m/s]   vertical speed scale
    centre_scale: float = 0.30      # [m]     radial distance scale
    fall_scale: float = 0.10        # [m]     height-drop scale
    joint_scale: float = 5.0        # [rad/s] joint speed scale

    w_still: float = 2.0
    w_centre: float = 1.0
    w_up: float = 1.0
    w_fall: float = 4.0
    w_effort: float = 0.02

    # ------------------------- ball state -------------------------------
    ball_xy = ball_positions[:, 0:2]
    ball_z = ball_positions[:, 2]
    v_xy = ball_linvels[:, 0:2]
    v_z = ball_linvels[:, 2]
    n_env = ball_z.shape[0]

    # (1) the ball must hardly move horizontally -> it cannot roll off
    speed_xy = torch.norm(v_xy, dim=-1)
    a_xy = speed_xy / vel_xy_scale
    r_still = torch.exp(-(a_xy * a_xy))

    # (2) the ball should stay close to the middle of the table
    dist_centre = torch.norm(ball_xy, dim=-1)
    a_c = dist_centre / centre_scale
    r_centre = torch.exp(-(a_c * a_c))

    # (3) the ball must not be sinking downwards
    fall_speed = torch.clamp(-v_z, min=0.0)
    a_vz = fall_speed / vel_z_scale
    r_up = torch.exp(-(a_vz * a_vz))

    # (4) fall detection: table level ~ median ball height of the batch
    z_sorted = torch.sort(ball_z).values
    z_ref = z_sorted[n_env // 2]
    drop = torch.clamp(z_ref - ball_z, min=0.0)
    a_drop = drop / fall_scale
    r_fall = -torch.clamp(a_drop, min=0.0, max=1.0)

    # (5) control effort / smoothness
    joint_speed = torch.norm(dof_velocities, dim=-1)
    a_j = joint_speed / joint_scale
    r_effort = -(a_j * a_j)

    total = (
        w_still * r_still
        + w_centre * r_centre
        + w_up * r_up
        + w_fall * r_fall
        + w_effort * r_effort
    )

    components: Dict[str, torch.Tensor] = {
        "still": w_still * r_still,
        "centre": w_centre * r_centre,
        "no_fall_velocity": w_up * r_up,
        "fall_penalty": w_fall * r_fall,
        "effort": w_effort * r_effort,
    }
    return total, components
