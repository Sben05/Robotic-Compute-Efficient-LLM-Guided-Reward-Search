@torch.jit.script
def compute_reward(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Tolerance based dense reward with alive bonus and fall penalty.
    obs_buf columns: [cart_pos, cart_vel, pole_angle, pole_vel]
    """
    cart_pos = obs_buf[:, 0]
    cart_vel = obs_buf[:, 1]
    pole_angle = obs_buf[:, 2]
    pole_vel = obs_buf[:, 3]

    # --- tolerances / limits ---
    max_angle = 0.25          # rad, pole considered upright inside this cone
    max_pole_vel = 1.0        # rad/s
    max_cart_pos = 1.0        # m, cart travel limit
    max_cart_vel = 2.0        # m/s

    # --- weights ---
    w_upright = 1.0
    w_alive = 0.1
    w_pole_vel = 0.5
    w_cart_pos = 0.2
    w_cart_vel = 0.2
    w_fall = 1.0

    # linear uprightness: 1 when perfectly upright, 0 at |angle| = max_angle
    upright_reward = 1.0 - torch.clamp(torch.abs(pole_angle) / max_angle, 0.0, 1.0)

    # velocity damping terms (linear decay inside their limits)
    pole_vel_reward = 1.0 - torch.clamp(torch.abs(pole_vel) / max_pole_vel, 0.0, 1.0)
    cart_pos_reward = 1.0 - torch.clamp(torch.abs(cart_pos) / max_cart_pos, 0.0, 1.0)
    cart_vel_reward = 1.0 - torch.clamp(torch.abs(cart_vel) / max_cart_vel, 0.0, 1.0)

    # alive bonus: small constant reward for not terminating
    alive_reward = torch.full_like(upright_reward, w_alive)

    # penalty that grows once the pole leaves the upright cone
    fall_penalty = -w_fall * torch.clamp((torch.abs(pole_angle) - max_angle) / max_angle, 0.0, 1.0)

    total_reward = (w_upright * upright_reward
                    + alive_reward
                    + w_pole_vel * pole_vel_reward
                    + w_cart_pos * cart_pos_reward
                    + w_cart_vel * cart_vel_reward
                    + fall_penalty)

    components: Dict[str, torch.Tensor] = {
        "upright_reward": upright_reward,
        "alive_reward": alive_reward,
        "pole_vel_reward": pole_vel_reward,
        "cart_pos_reward": cart_pos_reward,
        "cart_vel_reward": cart_vel_reward,
        "fall_penalty": fall_penalty,
    }
    return total_reward, components
