@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # -------- horizontal running speed of the torso --------
    vel = root_states[:, 7:10]
    speed = torch.sqrt(vel[:, 0] * vel[:, 0] + vel[:, 1] * vel[:, 1] + 1e-6)
    speed_reward = 0.2 * speed

    # -------- stay upright / alive (torso height) --------
    height = root_states[:, 2]
    alive_reward = torch.clamp(height - 0.5, min=0.0, max=1.0) * 0.5

    # -------- control effort / smoothness --------
    action_penalty = 0.03 * torch.sum(actions * actions, dim=-1)
    dof_vel_penalty = 1.0e-3 * torch.sum(dof_vel * dof_vel, dim=-1)

    reward = speed_reward + alive_reward - action_penalty - dof_vel_penalty

    components = {
        "speed": speed_reward,
        "alive": alive_reward,
        "action_penalty": -action_penalty,
        "dof_vel_penalty": -dof_vel_penalty,
    }
    return reward, components
