@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor,
                     prev_potentials: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # -------- metres of distance closed toward the target in the last step --------
    progress = (potentials - prev_potentials) * dt
    progress_reward = 2.0 * progress

    # -------- absolute running speed --------
    vel = root_states[:, 7:10]
    speed = torch.sqrt(vel[:, 0] * vel[:, 0] + vel[:, 1] * vel[:, 1] + 1e-6)
    temp_speed = 2.0
    speed_reward = 1.0 - torch.exp(-speed / temp_speed)

    # -------- alive bonus --------
    alive_reward = torch.clamp(root_states[:, 2] - 0.6, min=0.0, max=1.0)

    # -------- penalties --------
    action_penalty = 0.03 * torch.sum(actions * actions, dim=-1)
    dof_vel_penalty = 1.0e-3 * torch.sum(dof_vel * dof_vel, dim=-1)

    reward = progress_reward + 1.0 * speed_reward + 0.5 * alive_reward \
        - action_penalty - dof_vel_penalty

    components = {
        "progress": progress_reward,
        "speed": 1.0 * speed_reward,
        "alive": 0.5 * alive_reward,
        "action_penalty": -action_penalty,
        "dof_vel_penalty": -dof_vel_penalty,
    }
    return reward, components
