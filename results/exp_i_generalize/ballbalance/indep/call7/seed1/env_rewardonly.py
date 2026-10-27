@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,
    ball_linvels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Rewards keeping the ball stable and supported by the tabletop.

    The constants below assume a table centered at the world origin.
    They should be adjusted to match the dimensions of the table asset.
    """

    # Table geometry in world coordinates.
    table_half_x = 0.50
    table_half_y = 0.50
    table_surface_z = 0.75
    ball_radius = 0.025
    ball_target_z = table_surface_z + ball_radius

    # Position error relative to the center of the table.
    dx = ball_positions[..., 0]
    dy = ball_positions[..., 1]
    horizontal_distance_sq = dx * dx + dy * dy
    vertical_error = ball_positions[..., 2] - ball_target_z

    # Separate temperatures allow each shaping term to be tuned independently.
    center_temperature = 0.08
    height_temperature = 0.0025
    velocity_temperature = 0.04

    # Dense shaping: prefer the center, the correct height, and low speed.
    center_component = torch.exp(
        -horizontal_distance_sq / center_temperature
    )
    height_component = torch.exp(
        -(vertical_error * vertical_error) / height_temperature
    )

    speed_sq = torch.sum(ball_linvels * ball_linvels, dim=-1)
    stability_component = torch.exp(
        -speed_sq / velocity_temperature
    )

    # A valid ball pose must lie over the table and close to its surface.
    inside_table_xy = (
        (torch.abs(dx) <= table_half_x)
        & (torch.abs(dy) <= table_half_y)
    )
    at_table_height = torch.abs(vertical_error) <= 0.08
    on_table = inside_table_xy & at_table_height

    inside_table_component = inside_table_xy.to(dtype=ball_positions.dtype)
    on_table_component = on_table.to(dtype=ball_positions.dtype)
    off_table_component = 1.0 - inside_table_component

    # Increasing penalty as the ball falls below the tabletop.
    fall_distance = 0.25
    fall_component = torch.clamp(
        (ball_target_z - ball_positions[..., 2]) / fall_distance,
        min=0.0,
        max=1.0,
    )

    total_reward = (
        3.0 * center_component
        + 2.0 * height_component
        + 0.5 * stability_component
        + 2.0 * on_table_component
        - 3.0 * off_table_component
        - 4.0 * fall_component
    )

    components: Dict[str, torch.Tensor] = {
        "center": center_component,
        "height": height_component,
        "stability": stability_component,
        "on_table": on_table_component,
        "off_table": off_table_component,
        "falling": fall_component,
    }

    return total_reward, components
