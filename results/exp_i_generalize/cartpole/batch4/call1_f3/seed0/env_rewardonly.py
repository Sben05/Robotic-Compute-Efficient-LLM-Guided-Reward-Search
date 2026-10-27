@torch.jit.script
def compute_reward(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Multiplicative (gated) reward: uprightness is only rewarded when the pole
    is also calm; strong penalties for a fallen pole and for wild cart motion.
    obs_buf columns: [cart_pos, cart_vel, pole_angle, pole_vel]
    """
    cart_pos = obs_buf[:, 0]
    cart_vel = obs_buf[:, 1]
    pole_angle = obs_buf[:, 2]
    pole_vel = obs_buf[:, 3]

    # --- temperatures / gates ---
    angle_temp = 8.0
    gate_temp = 1.0
    centering_temp = 2.0
    fall_angle_limit = 0.5

    # --- weights ---
    w_upright = 3.0
    w_gate = 1.0
    w_centering = 0.5
    w_fall = 2.0
    w_cart_vel = 0.5

    # exponential uprightness
    upright_reward = torch.exp(-angle_temp * torch.abs(pole_angle))

    # stability gate: near-zero angular velocity required to collect full reward
    stability_gate = torch.exp(-gate_temp * torch.abs(pole_vel))

    # gated upright term (upright AND steady)
    gated_reward = upright_reward * stability_gate

    # keeping the cart near the middle of the track
    centering_reward = torch.exp(-centering_temp * torch.abs(cart_pos))

    # penalty for excessive cart velocity
    cart_vel_penalty = -w_cart_vel * torch.clamp(torch.abs(cart_vel) - 1.0, 0.0, 1.0)

    # penalty for the pole tipping over
    fall_penalty = -w_fall * torch.clamp((torch.abs(pole_angle) - fall_angle_limit) / fall_angle_limit, 0.0, 1.0)

    total_reward = (w_upright * gated_reward
                    + w_gate * stability_gate
                    + w_centering * centering_reward
                    + cart_vel_penalty
                    + fall_penalty)

    components: Dict[str, torch.Tensor] = {
        "upright_reward": upright_reward,
        "stability_gate": stability_gate,
        "gated_reward": gated_reward,
        "centering_reward": centering_reward,
        "cart_vel_penalty": cart_vel_penalty,
        "fall_penalty": fall_penalty,
    }
    return total_reward, components
