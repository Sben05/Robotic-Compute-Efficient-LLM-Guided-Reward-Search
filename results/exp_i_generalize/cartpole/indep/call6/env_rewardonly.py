@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- state ----------------
    cart_pos = dof_pos[:, 0]      # cart position on the rail
    cart_vel = dof_vel[:, 0]      # cart linear velocity
    pole_angle = dof_pos[:, 1]    # pole angle (0 == upright)
    pole_vel = dof_vel[:, 1]      # pole angular velocity

    # ---------------- temperatures (named params, not inputs) ----------------
    upright_temperature = 0.25      # rad   : sharpness of the "vertical" bonus
    pole_vel_temperature = 1.0      # rad/s : sharpness of the "not swinging" bonus
    cart_pos_temperature = 1.0      # m     : sharpness of the "centered cart" bonus
    cart_vel_temperature = 2.0      # m/s   : sharpness of the "still cart" bonus

    # ---------------- reward components ----------------
    upright_reward = torch.exp(-torch.abs(pole_angle) / upright_temperature)
    pole_vel_reward = torch.exp(-torch.abs(pole_vel) / pole_vel_temperature)
    cart_pos_reward = torch.exp(-torch.abs(cart_pos) / cart_pos_temperature)
    cart_vel_reward = torch.exp(-torch.abs(cart_vel) / cart_vel_temperature)

    action_penalty = torch.sum(torch.square(actions), dim=1)

    # ---------------- total reward ----------------
    reward = (
        upright_reward
        + 0.25 * pole_vel_reward
        + 0.10 * cart_pos_reward
        + 0.05 * cart_vel_reward
        - 0.005 * action_penalty
    )

    reward_components: Dict[str, torch.Tensor] = {
        "upright_reward": upright_reward,
        "pole_vel_reward": pole_vel_reward,
        "cart_pos_reward": cart_pos_reward,
        "cart_vel_reward": cart_vel_reward,
        "action_penalty": action_penalty,
    }

    return reward, reward_components
