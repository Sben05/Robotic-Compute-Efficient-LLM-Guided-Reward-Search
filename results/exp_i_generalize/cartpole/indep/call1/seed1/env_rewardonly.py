@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Joint 0: cart translation; Joint 1: pole rotation.
    cart_pos = dof_pos[:, 0]
    pole_ang = dof_pos[:, 1]
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]

    # --- Temperatures (one per transformed component) ---
    upright_temperature: float = 8.0
    pole_velocity_temperature: float = 0.5
    cart_velocity_temperature: float = 0.25
    cart_position_temperature: float = 0.5

    # --- Component weights ---
    upright_weight: float = 2.0
    pole_velocity_weight: float = 0.5
    cart_velocity_weight: float = 0.25
    cart_position_weight: float = 0.25

    # Main task: pole is vertical (angle == 0) and stationary.
    upright_reward = torch.exp(-upright_temperature * torch.abs(pole_ang))

    # Secondary: damp pole oscillation and cart jitter, keep cart near center.
    pole_velocity_reward = torch.exp(-pole_velocity_temperature * pole_vel.square())
    cart_velocity_reward = torch.exp(-cart_velocity_temperature * cart_vel.square())
    cart_position_reward = torch.exp(-cart_position_temperature * cart_pos.square())

    components: Dict[str, torch.Tensor] = {
        "upright": upright_weight * upright_reward,
        "pole_velocity": pole_velocity_weight * pole_velocity_reward,
        "cart_velocity": cart_velocity_weight * cart_velocity_reward,
        "cart_position": cart_position_weight * cart_position_reward,
    }

    total_reward = (
        upright_weight * upright_reward
        + pole_velocity_weight * pole_velocity_reward
        + cart_velocity_weight * cart_velocity_reward
        + cart_position_weight * cart_position_reward
    )

    return total_reward, components
