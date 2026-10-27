@torch.jit.script
def compute_reward(potentials: torch.Tensor, prev_potentials: torch.Tensor, obs_buf: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    progress = (prev_potentials - potentials) * dt  # Target progress per step
    upright_proj = obs_buf[:, 10]
    upright_weight = 0.1
    total_reward = progress + upright_weight * upright_proj
    reward_components = {
        "progress_reward": progress,
        "upright_reward": upright_proj
    }
    return total_reward, reward_components
