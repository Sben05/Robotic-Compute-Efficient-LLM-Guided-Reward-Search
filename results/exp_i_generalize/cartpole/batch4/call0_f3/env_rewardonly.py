@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sparse upright bonus with hinge-style failure penalties.

    A large step reward is given only while the pole is genuinely upright
    (inside a small angle tolerance). Additional terms penalise falling over,
    excessive pole/cart speed and drifting away from the track centre.
    """
    cart_pos = dof_pos[:, 0].reshape(-1)
    pole_angle = dof_pos[:, 1].reshape(-1)
    cart_vel = dof_vel[:, 0].reshape(-1)
    pole_vel = dof_vel[:, 1].reshape(-1)

    # --- thresholds / temperatures ---
    upright_tol: float = 0.087        # ~5 degrees
    fail_angle: float = 0.35          # pole is falling
    vel_tol: float = 1.0
    cart_limit: float = 1.0
    vel_temp: float = 1.0
    pos_temp: float = 2.0

    abs_angle = torch.abs(pole_angle)
    abs_pole_vel = torch.abs(pole_vel)
    abs_cart_vel = torch.abs(cart_vel)
    abs_cart_pos = torch.abs(cart_pos)

    ones = torch.ones_like(abs_angle)
    zeros = torch.zeros_like(abs_angle)

    upright_hold = torch.where(abs_angle < upright_tol, ones, zeros)
    falling = torch.where(abs_angle > fail_angle, ones, zeros)

    vel_penalty = torch.exp(-torch.square(abs_pole_vel - vel_tol) / vel_temp) * 0.0 \
        + torch.clamp(abs_pole_vel - vel_tol, min=0.0) \
        + torch.clamp(abs_cart_vel - vel_tol, min=0.0)
    pos_penalty = torch.clamp(abs_cart_pos - cart_limit, min=0.0)

    upright_bonus: float = 1.5
    fall_penalty: float = 1.0
    vel_penalty_weight: float = 0.1
    pos_penalty_weight: float = 0.2
    alive_bonus: float = 0.1

    reward = (upright_bonus * upright_hold
              + alive_bonus
              - fall_penalty * falling
              - vel_penalty_weight * vel_penalty
              - pos_penalty_weight * pos_penalty)

    components = {
        "upright_hold": upright_bonus * upright_hold,
        "falling": -fall_penalty * falling,
        "velocity_penalty": -vel_penalty_weight * vel_penalty,
        "position_penalty": -pos_penalty_weight * pos_penalty,
        "alive": torch.ones_like(abs_angle) * alive_bonus,
    }
    return reward, components
