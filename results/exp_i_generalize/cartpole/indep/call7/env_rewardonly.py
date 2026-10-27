@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Cart DOF: 0; pole hinge DOF: 1.
    cart_pos = dof_pos[:, 0]
    cart_vel = dof_vel[:, 0]
    pole_angle = dof_pos[:, 1]
    pole_vel = dof_vel[:, 1]

    # Separate temperatures allow each shaped component to be tuned independently.
    upright_temperature: float = 0.5
    angular_velocity_temperature: float = 0.5
    cart_position_temperature: float = 2.0
    cart_velocity_temperature: float = 2.0

    # Strongly reward keeping the pole near the vertical (zero-angle) position.
    upright_reward = torch.exp(-pole_angle.square() / upright_temperature)

    # Damp excessive pole motion and discourage unnecessary cart motion.
    angular_damping_reward = torch.exp(
        -pole_vel.square() / angular_velocity_temperature
    )
    cart_center_reward = torch.exp(
        -cart_pos.square() / cart_position_temperature
    )
    cart_stillness_reward = torch.exp(
        -cart_vel.square() / cart_velocity_temperature
    )

    # A small constant bonus favors remaining balanced for longer.
    alive_bonus = torch.ones_like(upright_reward)

    total_reward = (
        2.0 * upright_reward
        + 0.25 * angular_damping_reward
        + 0.10 * cart_center_reward
        + 0.05 * cart_stillness_reward
        + 0.10 * alive_bonus
    )

    components = {
        "upright_reward": upright_reward,
        "angular_damping_reward": angular_damping_reward,
        "cart_center_reward": cart_center_reward,
        "cart_stillness_reward": cart_stillness_reward,
        "alive_bonus": alive_bonus,
    }

    return total_reward, components
