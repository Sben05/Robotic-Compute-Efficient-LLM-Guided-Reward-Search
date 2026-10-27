@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Quadratic-cost (energy style) reward normalised with an exponential.

    The total cost sums weighted squared deviations of pole angle, pole angular
    velocity, cart position and cart velocity; the reward is exp(-cost / temp),
    which is smoothly normalised into (0, 1].
    """
    cart_pos = dof_pos[:, 0].reshape(-1)
    pole_angle = dof_pos[:, 1].reshape(-1)
    cart_vel = dof_vel[:, 0].reshape(-1)
    pole_vel = dof_vel[:, 1].reshape(-1)

    # --- temperature / weights ---
    cost_temp: float = 0.5
    w_angle: float = 4.0
    w_pole_vel: float = 0.5
    w_cart_pos: float = 0.5
    w_cart_vel: float = 0.1

    angle_cost = w_angle * torch.square(pole_angle)
    pole_vel_cost = w_pole_vel * torch.square(pole_vel)
    cart_pos_cost = w_cart_pos * torch.square(cart_pos)
    cart_vel_cost = w_cart_vel * torch.square(cart_vel)

    total_cost = angle_cost + pole_vel_cost + cart_pos_cost + cart_vel_cost

    upright_temp: float = 0.5
    upright = torch.exp(-torch.square(pole_angle) / upright_temp)

    reward = torch.exp(-total_cost / cost_temp)

    components = {
        "angle_cost": -angle_cost,
        "pole_vel_cost": -pole_vel_cost,
        "cart_pos_cost": -cart_pos_cost,
        "cart_vel_cost": -cart_vel_cost,
        "upright": upright,
        "total_reward": reward,
    }
    return reward, components
