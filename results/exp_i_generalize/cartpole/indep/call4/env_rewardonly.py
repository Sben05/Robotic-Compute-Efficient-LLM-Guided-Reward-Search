@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward an agent for keeping the pole upright and the cart controlled.

    Inputs:
        dof_pos: Joint positions [cart position, pole angle], shape (num_envs, 2).
        dof_vel: Joint velocities [cart velocity, pole angular velocity],
                 shape (num_envs, 2).
    """
    cart_position = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_velocity = dof_vel[:, 0]
    pole_angular_velocity = dof_vel[:, 1]

    # Separate temperatures allow each reward component to have its own scale.
    upright_temperature = 0.20       # Radians.
    position_temperature = 0.50      # Cart displacement.
    velocity_temperature = 1.00      # Joint velocities.

    # Maximum reward when the pole is exactly vertical.
    upright_reward = torch.exp(
        -pole_angle.pow(2) / (upright_temperature * upright_temperature)
    )

    # Discourage the cart from drifting far from its starting position.
    cart_position_reward = torch.exp(
        -cart_position.pow(2) / (position_temperature * position_temperature)
    )

    # Favor smooth motion rather than corrective, high-speed movements.
    velocity_reward = torch.exp(
        -(cart_velocity.pow(2) + pole_angular_velocity.pow(2))
        / (velocity_temperature * velocity_temperature)
    )

    reward = (
        2.0 * upright_reward
        + 0.25 * cart_position_reward
        + 0.25 * velocity_reward
    )

    components = {
        "upright": upright_reward,
        "cart_position": cart_position_reward,
        "velocity": velocity_reward,
    }

    return reward, components
