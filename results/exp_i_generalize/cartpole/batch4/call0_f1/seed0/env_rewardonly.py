@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Dense exponential shaping reward.

    The pole should stay upright (small |pole angle|), move slowly, and the cart
    should stay near the centre of the track while moving smoothly.
    """
    cart_pos = dof_pos[:, 0].reshape(-1)
    pole_angle = dof_pos[:, 1].reshape(-1)
    cart_vel = dof_vel[:, 0].reshape(-1)
    pole_vel = dof_vel[:, 1].reshape(-1)

    # --- temperature parameters (not inputs) ---
    upright_temp: float = 0.25      # rad^2, sharpness of the upright bonus
    vel_temp: float = 0.5           # (rad/s or m/s)^2, sharpness of velocity bonuses
    pos_temp: float = 2.0           # m^2, sharpness of the cart-centring bonus

    # --- reward components ---
    upright = torch.exp(-torch.square(pole_angle) / upright_temp)
    pole_quiet = torch.exp(-torch.square(pole_vel) / vel_temp)
    cart_centered = torch.exp(-torch.square(cart_pos) / pos_temp)
    cart_smooth = torch.exp(-torch.square(cart_vel) / vel_temp)

    # --- weights ---
    w_upright: float = 2.0
    w_pole_vel: float = 0.5
    w_cart_pos: float = 0.5
    w_cart_vel: float = 0.25
    alive_bonus: float = 0.5

    reward = (w_upright * upright
              + w_pole_vel * pole_quiet
              + w_cart_pos * cart_centered
              + w_cart_vel * cart_smooth
              + alive_bonus)

    components = {
        "upright": upright,
        "pole_quiet": pole_quiet,
        "cart_centered": cart_centered,
        "cart_smooth": cart_smooth,
        "alive": torch.ones_like(upright) * alive_bonus,
    }
    return reward, components
