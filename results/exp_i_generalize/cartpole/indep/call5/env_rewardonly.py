@torch.jit.script
def compute_reward(dof_pos: torch.Tensor,
                   dof_vel: torch.Tensor,
                   actions: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ----------------- weights -----------------
    upright_weight: float = 3.0
    angle_penalty_weight: float = 0.25
    pole_vel_weight: float = 0.5
    cart_vel_weight: float = 0.25
    center_weight: float = 0.25
    action_weight: float = 0.02
    alive_bonus: float = 0.25

    # ---------------- temperatures --------------
    upright_temperature: float = 0.20     # rad
    pole_vel_temperature: float = 1.0     # rad/s
    cart_vel_temperature: float = 1.0     # m/s
    center_temperature: float = 1.0       # m

    # ------------- state extraction -----------
    cart_pos = dof_pos[:, 0]
    cart_vel = dof_vel[:, 0]
    pole_angle = dof_pos[:, 1]
    pole_vel = dof_vel[:, 1]

    abs_angle = torch.abs(pole_angle)

    # ------------- reward terms -----------------
    upright_reward = torch.exp(-abs_angle / upright_temperature)
    angle_penalty = -angle_penalty_weight * abs_angle
    pole_vel_reward = torch.exp(-torch.abs(pole_vel) / pole_vel_temperature)
    cart_vel_reward = torch.exp(-torch.abs(cart_vel) / cart_vel_temperature)
    center_reward = torch.exp(-torch.abs(cart_pos) / center_temperature)
    alive = torch.ones_like(upright_reward) * alive_bonus

    if actions is None:
        action_penalty = torch.zeros_like(upright_reward)
    else:
        action_penalty = -action_weight * torch.sum(torch.square(actions), dim=-1)

    reward = (upright_weight * upright_reward
              + angle_penalty
              + pole_vel_weight * pole_vel_reward
              + cart_vel_weight * cart_vel_reward
              + center_weight * center_reward
              + action_penalty
              + alive)

    components: Dict[str, torch.Tensor] = {
        "upright": upright_weight * upright_reward,
        "angle_penalty": angle_penalty,
        "pole_velocity": pole_vel_weight * pole_vel_reward,
        "cart_velocity": cart_vel_weight * cart_vel_reward,
        "cart_center": center_weight * center_reward,
        "action_penalty": action_penalty,
        "alive": alive,
    }

    return reward, components
