@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Multiplicative (product-of-experts) reward.

    All sub-objectives must be satisfied simultaneously for the reward to be
    large, so the agent cannot trade pole stability for cart convenience.
    """
    cart_pos = dof_pos[:, 0].reshape(-1)
    pole_angle = dof_pos[:, 1].reshape(-1)
    cart_vel = dof_vel[:, 0].reshape(-1)
    pole_vel = dof_vel[:, 1].reshape(-1)

    # --- temperature parameters ---
    upright_temp: float = 0.5
    cart_temp: float = 4.0
    pole_vel_temp: float = 1.0
    cart_vel_temp: float = 1.0

    upright = torch.exp(-torch.square(pole_angle) / upright_temp)
    cart_centered = torch.exp(-torch.square(cart_pos) / cart_temp)
    pole_damped = torch.exp(-torch.square(pole_vel) / pole_vel_temp)
    cart_damped = torch.exp(-torch.square(cart_vel) / cart_vel_temp)

    # multiplicative gating: keep a floor so the gradient signal never vanishes
    upright_term = 0.2 + 0.8 * upright
    center_term = 0.5 + 0.5 * cart_centered
    pole_term = 0.5 + 0.5 * pole_damped
    cart_term = 0.5 + 0.5 * cart_damped

    scale: float = 2.0
    alive_bonus: float = 0.25
    reward = scale * upright_term * center_term * pole_term * cart_term + alive_bonus

    components = {
        "upright_gated": upright_term,
        "center_gated": center_term,
        "pole_damped_gated": pole_term,
        "cart_damped_gated": cart_term,
        "alive": torch.ones_like(upright) * alive_bonus,
    }
    return reward, components
