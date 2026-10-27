@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---- temperature / scaling constants (not environment inputs) ----
    forward_vel_temp: float = 1.5
    lateral_vel_temp: float = 1.0
    vertical_vel_temp: float = 1.0
    ang_vel_temp: float = 1.0
    action_temp: float = 1.0
    dof_vel_temp: float = 1.0
    upright_temp: float = 1.0

    linear_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]
    quat = root_states[:, 3:7]

    forward_vel = linear_vel[:, 0]
    lateral_vel = linear_vel[:, 1]
    vertical_vel = linear_vel[:, 2]

    # main task term: go as fast as possible along +x
    reward_forward = torch.exp(forward_vel / forward_vel_temp) - 1.0

    # stay straight and on the ground
    reward_lateral = -torch.abs(lateral_vel) / lateral_vel_temp
    reward_vertical = -torch.abs(vertical_vel) / vertical_vel_temp
    reward_ang_vel = -torch.sum(torch.square(ang_vel[:, :2]), dim=1) / ang_vel_temp

    # keep the torso upright (z component of the body up-axis)
    up_proj = 1.0 - 2.0 * (quat[:, 1] * quat[:, 1] + quat[:, 2] * quat[:, 2])
    reward_upright = torch.clamp(up_proj, min=0.0) / upright_temp

    # smoothness / energy regularizers
    reward_action = -torch.sum(torch.square(actions), dim=1) / action_temp
    reward_dof_vel = -torch.sum(torch.square(dof_vel), dim=1) / dof_vel_temp

    reward_alive = torch.ones_like(forward_vel)

    total = (reward_forward
             + 0.5 * reward_lateral
             + 0.5 * reward_vertical
             + 0.5 * reward_ang_vel
             + 0.5 * reward_upright
             + 0.05 * reward_action
             + 0.05 * reward_dof_vel
             + 0.5 * reward_alive)

    components = {
        "reward_forward": reward_forward,
        "reward_lateral": reward_lateral,
        "reward_vertical": reward_vertical,
        "reward_ang_vel": reward_ang_vel,
        "reward_upright": reward_upright,
        "reward_action": reward_action,
        "reward_dof_vel": reward_dof_vel,
        "reward_alive": reward_alive,
    }
    return total, components
