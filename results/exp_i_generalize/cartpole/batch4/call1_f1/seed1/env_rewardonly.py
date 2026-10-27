@torch.jit.script
def compute_reward(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Weighted sum of exponential shaping rewards.
    obs_buf columns: [cart_pos, cart_vel, pole_angle, pole_vel]
    """
    cart_pos = obs_buf[:, 0]
    cart_vel = obs_buf[:, 1]
    pole_angle = obs_buf[:, 2]
    pole_vel = obs_buf[:, 3]

    # --- temperatures (per component) ---
    angle_temp = 5.0
    pole_vel_temp = 0.5
    cart_pos_temp = 1.0
    cart_vel_temp = 0.5

    # --- weights ---
    w_upright = 2.0
    w_pole_vel = 0.5
    w_cart_pos = 0.25
    w_cart_vel = 0.25

    upright_reward = torch.exp(-angle_temp * torch.abs(pole_angle))
    pole_vel_reward = torch.exp(-pole_vel_temp * torch.abs(pole_vel))
    cart_pos_reward = torch.exp(-cart_pos_temp * torch.abs(cart_pos))
    cart_vel_reward = torch.exp(-cart_vel_temp * torch.abs(cart_vel))

    total_reward = (w_upright * upright_reward
                    + w_pole_vel * pole_vel_reward
                    + w_cart_pos * cart_pos_reward
                    + w_cart_vel * cart_vel_reward)

    components: Dict[str, torch.Tensor] = {
        "upright_reward": upright_reward,
        "pole_vel_reward": pole_vel_reward,
        "cart_pos_reward": cart_pos_reward,
        "cart_vel_reward": cart_vel_reward,
    }
    return total_reward, components
