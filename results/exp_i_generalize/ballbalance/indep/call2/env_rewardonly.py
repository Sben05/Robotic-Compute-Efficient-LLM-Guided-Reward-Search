@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,
    ball_linvels: torch.Tensor,
    sensor_forces: torch.Tensor,
    sensor_torques: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    num_envs = ball_positions.shape[0]

    ball_pos = ball_positions.reshape(num_envs, 3)
    ball_vel = ball_linvels.reshape(num_envs, 3)

    # Use the whole force/torque reading so the reward does not depend on the
    # sensor's sign convention or which force axis is reported.
    force_values = sensor_forces.reshape(num_envs, -1)
    torque_values = sensor_torques.reshape(num_envs, -1)[:, 0:3]

    force_magnitude = torch.sqrt(
        torch.sum(force_values * force_values, dim=1) + 1.0e-8
    )
    torque_squared = torch.sum(torque_values * torque_values, dim=1)
    velocity_squared = torch.sum(ball_vel * ball_vel, dim=1)

    contact_temperature = 10.0
    stability_temperature = 0.10
    centering_temperature = 0.20
    downward_velocity_temperature = 0.10
    height_temperature = 0.25

    # The force sensor is used as a proxy for ball-table contact. Saturating
    # this term rewards support without rewarding excessive impact force.
    contact_quality = 1.0 - torch.exp(-force_magnitude / contact_temperature)
    contact_component = 4.0 * contact_quality

    # A low ball speed indicates that the ball is settled rather than rolling,
    # bouncing, or being thrown.
    stability_component = torch.exp(-velocity_squared / stability_temperature)

    # Contact away from the sensor center produces a larger reaction torque.
    # This softly encourages the ball to remain in the safer central region.
    centering_quality = torch.exp(-torque_squared / centering_temperature)
    centering_component = 0.5 * contact_quality * centering_quality

    # Downward motion is a strong early indicator of slipping or falling.
    downward_speed = torch.clamp(-ball_vel[:, 2], min=0.0)
    downward_quality = 1.0 - torch.exp(
        -(downward_speed * downward_speed) / downward_velocity_temperature
    )
    downward_velocity_component = -2.0 * downward_quality

    # A weak height preference helps distinguish the tabletop from the floor,
    # but is gated by contact so the agent cannot gain height by throwing the ball.
    height_quality = torch.sigmoid(ball_pos[:, 2] / height_temperature)
    height_component = 0.25 * contact_quality * height_quality

    reward = (
        contact_component
        + stability_component
        + centering_component
        + downward_velocity_component
        + height_component
    )

    components: Dict[str, torch.Tensor] = {
        "table_contact": contact_component,
        "ball_stability": stability_component,
        "centered_contact": centering_component,
        "downward_velocity": downward_velocity_component,
        "table_height": height_component,
    }

    return reward, components
