@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,
    ball_linvels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward a stable ball resting near the center of the table.

    ball_positions and ball_linvels must be expressed in the table's
    coordinate frame, with the table center at x = y = 0.

    The geometry constants below must match the environment. If ball_positions
    are already relative to the table top, set table_top_height to 0.0.
    """
    table_top_height: float = 0.50   # Table surface height in the observation frame.
    table_half_width: float = 0.50   # Distance from the table center to an edge.
    ball_radius: float = 0.03

    rest_height: float = table_top_height + ball_radius

    # Reward the ball being at its resting height above the table.
    height_temperature: float = 0.12
    height_error = torch.abs(ball_positions[:, 2] - rest_height)
    height_score = torch.exp(-height_error / height_temperature)

    # Reward keeping the ball away from the table edges.
    center_temperature: float = 0.30
    horizontal_distance = torch.sqrt(
        ball_positions[:, 0] * ball_positions[:, 0]
        + ball_positions[:, 1] * ball_positions[:, 1]
        + 1.0e-8
    )
    normalized_distance = horizontal_distance / table_half_width
    center_score = torch.exp(
        -(normalized_distance * normalized_distance) / center_temperature
    )

    # Penalize fast motion: a balanced ball should move slowly.
    velocity_temperature: float = 0.40
    ball_speed = torch.sqrt(
        torch.sum(ball_linvels * ball_linvels, dim=1) + 1.0e-8
    )
    stability_score = torch.exp(-ball_speed / velocity_temperature)

    # Detect a ball that has dropped below the table. The grace distance
    # allows small bounces without immediately terminating the reward.
    drop_grace_distance: float = 0.15
    fall_distance = torch.clamp(
        rest_height - drop_grace_distance - ball_positions[:, 2],
        min=0.0,
    )
    fall_indicator = torch.clamp(
        fall_distance / drop_grace_distance,
        min=0.0,
        max=1.0,
    )
    alive_score = 1.0 - fall_indicator

    # Downward motion gives an early signal that the ball is leaving the table.
    downward_speed = torch.clamp(-ball_linvels[:, 2], min=0.0)

    height_component = 2.0 * height_score
    centering_component = 1.5 * center_score
    stability_component = 0.5 * stability_score
    alive_component = 1.0 * alive_score
    fall_component = -4.0 * fall_indicator
    downward_velocity_component = -0.5 * downward_speed

    total_reward = (
        height_component
        + centering_component
        + stability_component
        + alive_component
        + fall_component
        + downward_velocity_component
    )

    components: Dict[str, torch.Tensor] = {
        "height_alignment": height_component,
        "table_centering": centering_component,
        "velocity_stability": stability_component,
        "alive_bonus": alive_component,
        "fall_penalty": fall_component,
        "downward_velocity_penalty": downward_velocity_component,
    }

    return total_reward, components
