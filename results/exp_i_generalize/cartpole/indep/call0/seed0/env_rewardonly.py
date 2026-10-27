@torch.jit.script
def compute_reward(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Observations: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    pole_angle = obs_buf[:, 2]
    pole_angular_velocity = obs_buf[:, 3]

    # Separate temperatures control the sharpness of each transformed reward.
    pole_angle_temperature = 0.20
    pole_velocity_temperature = 0.50

    # Maximum reward when the pole is vertical and its angular velocity is zero.
    upright_reward = torch.exp(-torch.abs(pole_angle) / pole_angle_temperature)
    stability_reward = torch.exp(
        -torch.abs(pole_angular_velocity) / pole_velocity_temperature
    )

    total_reward = upright_reward + 0.25 * stability_reward

    components = {
        "upright_reward": upright_reward,
        "pole_stability_reward": 0.25 * stability_reward,
    }

    return total_reward, components
