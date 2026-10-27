@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,   # self.ball_positions  (N, 3)
    ball_linvels: torch.Tensor,     # self.ball_linvels    (N, 3)
    dof_velocities: torch.Tensor,   # self.dof_velocities  (N, num_dofs)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward for "keep the ball on the table top without letting it fall".

    The reward is built from four ideas:
      1. the ball must sit at the height of the table surface (rest height),
      2. the ball must stay close to the middle of the table (so it cannot
         slowly roll over the edge),
      3. the ball must stay almost at rest (a fast rolling ball leaves the
         table much faster),
      4. a large penalty is paid as soon as the ball drops below the table
         top (it fell off), and a smaller penalty if it is lifted off the
         table (tossed in the air).

    NOTE: `table_surface_z` must match the height of the table top in the
    frame used by `self.ball_positions` (0.0 if the observation frame has the
    table top at its origin), and `table_center_xy` / `table_half_extent`
    must match the table geometry.  They are hard-coded constants (not
    environment inputs) and are meant to be tuned.
    """

    # ------------------------------------------------------------------
    # Tunable constants (defined inside the function -> not extra inputs)
    # ------------------------------------------------------------------
    table_surface_z: float = 0.0      # z of the table top in the observation frame
    ball_radius: float = 0.05         # nominal radius of the ball
    table_center_xy: float = 0.0      # xy position of the table centre
    table_half_extent: float = 0.30   # half width of the (square) table top
    edge_margin: float = 0.05         # width of the soft transition bands

    # One temperature per exponentially shaped component
    temp_height: float = 0.05
    temp_center: float = 0.15
    temp_velocity: float = 0.25

    # Reward weights
    w_height: float = 1.0
    w_center: float = 0.5
    w_velocity: float = 0.5
    w_on_table: float = 1.0
    w_fall: float = 5.0
    w_air: float = 1.0
    w_dof: float = 0.005

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    ball_z = ball_positions[..., 2]
    ball_xy = ball_positions[..., 0:2]
    rest_z = table_surface_z + ball_radius

    dist_xy = torch.sqrt(torch.sum((ball_xy - table_center_xy) ** 2, dim=-1) + 1e-8)
    speed_sq = torch.sum(ball_linvels * ball_linvels, dim=-1)

    # (1) Height: reward 1 when the ball is at (or slightly above) the
    #     resting height, decaying smoothly when it sinks below the table.
    height_err = torch.clamp(rest_z - ball_z, min=0.0)
    r_height = torch.exp(-(height_err ** 2) / (2.0 * temp_height ** 2))

    # (2) Centre: keep the ball close to the middle of the table top.
    r_center = torch.exp(-dist_xy / temp_center)

    # (3) Velocity: a still ball stays on the table, a moving one rolls off.
    r_velocity = torch.exp(-speed_sq / (2.0 * temp_velocity ** 2))

    # (4) Smooth "ball is on the table" indicator:
    #     inside the table footprint AND at/above the table surface.
    inside_xy = torch.clamp((table_half_extent - dist_xy) / edge_margin, min=0.0, max=1.0)
    above_surface = torch.clamp(
        (ball_z - (table_surface_z - 0.5 * ball_radius)) / (ball_radius + edge_margin),
        min=0.0, max=1.0,
    )
    r_on_table = inside_xy * above_surface

    # (5) Fall penalty: the ball has dropped below the top of the table.
    fall = torch.clamp(
        (table_surface_z - 0.5 * ball_radius - ball_z) / edge_margin, min=0.0, max=1.0
    )
    r_fall = -fall

    # (6) Airborne penalty: the ball must not be tossed up in the air.
    airborne = torch.clamp((ball_z - (rest_z + edge_margin)) / edge_margin, min=0.0, max=1.0)
    r_air = -airborne

    # (7) Small joint-velocity regularisation -> gentle, stable corrections.
    r_dof = -torch.sum(
        dof_velocities[:, 1] ** 2 + dof_velocities[:, 3] ** 2 + dof_velocities[:, 5] ** 2,
        dim=-1,
    )

    # ------------------------------------------------------------------
    # Total reward and components
    # ------------------------------------------------------------------
    total_reward = (
        w_height * r_height
        + w_center * r_center
        + w_velocity * r_velocity
        + w_on_table * r_on_table
        + w_fall * r_fall
        + w_air * r_air
        + w_dof * r_dof
    )

    components: Dict[str, torch.Tensor] = {
        "ball_height": r_height,
        "ball_center": r_center,
        "ball_velocity": r_velocity,
        "on_table": r_on_table,
        "fall_penalty": r_fall,
        "airborne_penalty": r_air,
        "dof_regularization": r_dof,
    }

    return total_reward, components
