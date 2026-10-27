@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Cart and pole state from the environment's DOF buffers.
    cart_position = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_velocity = dof_vel[:, 0]
    pole_velocity = dof_vel[:, 1]

    # Separate temperatures provide smooth shaping for each component.
    upright_temperature: float = 0.25
    pole_velocity_temperature: float = 0.5
    cart_position_temperature: float = 1.0
    cart_velocity_temperature: float = 1.0

    # Smooth rewards: each is largest when the corresponding state is ideal.
    upright_reward = torch.exp(
        -pole_angle.pow(2) / upright_temperature
    )
    pole_velocity_reward = torch.exp(
        -pole_velocity.pow(2) / pole_velocity_temperature
    )
    cart_position_reward = torch.exp(
        -cart_position.pow(2) / cart_position_temperature
    )
    cart_velocity_reward = torch.exp(
        -cart_velocity.pow(2) / cart_velocity_temperature
    )

    # Uprightness is the primary objective; the other terms discourage
    # excessive pole motion, cart drift, and excessive cart movement.
    upright_component = 2.0 * upright_reward
    pole_velocity_component = 0.25 * pole_velocity_reward
    cart_position_component = 0.1 * cart_position_reward
    cart_velocity_component = 0.1 * cart_velocity_reward
    alive_component = torch.ones_like(upright_reward)

    reward = (
        alive_component
        + upright_component
        + pole_velocity_component
        + cart_position_component
        + cart_velocity_component
    )

    components: Dict[str, torch.Tensor] = {
        "alive": alive_component,
        "upright": upright_component,
        "pole_velocity": pole_velocity_component,
        "cart_position": cart_position_component,
        "cart_velocity": cart_velocity_component,
    }

    return reward, components
