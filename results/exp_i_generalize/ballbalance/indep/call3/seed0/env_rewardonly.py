@torch.jit.script
def compute_reward(
    ball_linvels: torch.Tensor,
    sensor_forces: torch.Tensor,
    sensor_torques: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = ball_linvels.device
    dtype = ball_linvels.dtype
    num_envs = ball_linvels.shape[0]

    # A load on the table sensor indicates that the table is supporting the ball.
    contact_force_threshold = 0.05
    contact_temperature = 0.25
    velocity_temperature = 0.05
    torque_temperature = 0.02

    force_vector = sensor_forces[:, 0, :]
    torque_vector = sensor_torques[:, 0, :]

    force_magnitude_sq = torch.sum(force_vector * force_vector, dim=1)
    force_magnitude = torch.sqrt(force_magnitude_sq + 1.0e-8)
    has_contact = (force_magnitude > contact_force_threshold).to(dtype)

    # Saturating contact reward avoids rewarding excessive force on the table.
    contact_score = has_contact * torch.sigmoid(
        force_magnitude / contact_temperature
    )

    # A supported ball should move slowly; this also detects accelerating falls.
    speed_sq = torch.sum(ball_linvels * ball_linvels, dim=1)
    stability_score = torch.exp(-speed_sq / velocity_temperature)

    # A small reaction torque generally means the ball is close to the table's
    # supported center rather than close to an edge.
    torque_magnitude_sq = torch.sum(torque_vector * torque_vector, dim=1)
    centering_score = has_contact * torch.exp(
        -torque_magnitude_sq / torque_temperature
    )

    alive_component = 0.25 * torch.ones(
        num_envs, device=device, dtype=dtype
    )
    contact_component = 2.0 * contact_score
    stability_component = 1.0 * stability_score
    centering_component = 0.5 * centering_score
    off_table_component = -3.0 * (1.0 - has_contact)

    total_reward = (
        alive_component
        + contact_component
        + stability_component
        + centering_component
        + off_table_component
    )

    components = {
        "alive_bonus": alive_component,
        "table_contact": contact_component,
        "ball_stability": stability_component,
        "centered_ball": centering_component,
        "off_table_penalty": off_table_component,
    }

    return total_reward, components
