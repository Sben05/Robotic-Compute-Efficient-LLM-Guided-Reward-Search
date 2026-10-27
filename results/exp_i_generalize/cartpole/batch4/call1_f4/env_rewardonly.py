@torch.jit.script
def compute_reward(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Quadratic-error (potential-like) reward mapped through exponentials.
    Emphasizes velocity damping and keeping the cart centered, with a
    separate hinge penalty when the pole is beyond the upright tolerance.
    obs_buf columns: [cart_pos, cart_vel, pole_angle, pole_vel]
    """
    cart_pos = obs_buf[:, 0]
    cart_vel = obs_buf[:, 1]
    pole_angle = obs_buf[:, 2]
    pole_vel = obs_buf[:, 3]

    # --- temperatures ---
    angle_temp = 6.0
    pole_vel_temp = 0.5
    cart_pos_temp = 1.0
    cart_vel_temp = 0.5

    # --- scales of the quadratic errors ---
    angle_scale = 0.2
    pole_vel_scale = 0.5
    cart_pos_scale = 0.5
    cart_vel_scale = 1.0

    # --- weights ---
    w_upright = 2.0
    w_damping = 1.0
    w_centering = 0.5
    w_hinge = 1.0
    hinge_limit = 0.3

    # squared errors
    angle_err = (pole_angle / angle_scale) ** 2
    pole_vel_err = (pole_vel / pole_vel_scale) ** 2
    cart_pos_err = (cart_pos / cart_pos_scale) ** 2
    cart_vel_err = (cart_vel / cart_vel_scale) ** 2

    # exponential mappings -> bounded components in (0, 1]
    upright_reward = torch.exp(-angle_temp * angle_err)
    damping_reward = torch.exp(-pole_vel_temp * pole_vel_err)
    centering_reward = torch.exp(-cart_pos_temp * cart_pos_err)
    smoothness_reward = torch.exp(-cart_vel_temp * cart_vel_err)

    # hinge penalty beyond the upright tolerance
    hinge_penalty = -w_hinge * torch.clamp((torch.abs(pole_angle) - hinge_limit) / hinge_limit, 0.0, 1.0)

    total_reward = (w_upright * upright_reward
                    + w_damping * damping_reward
                    + w_centering * centering_reward
                    + 0.5 * smoothness_reward
                    + hinge_penalty)

    components: Dict[str, torch.Tensor] = {
        "upright_reward": upright_reward,
        "damping_reward": damping_reward,
        "centering_reward": centering_reward,
        "smoothness_reward": smoothness_reward,
        "hinge_penalty": hinge_penalty,
    }
    return total_reward, components
