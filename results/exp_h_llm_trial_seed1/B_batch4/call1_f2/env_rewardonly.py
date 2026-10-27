@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                     heading_vec: torch.Tensor,
                     potentials: torch.Tensor,
                     prev_potentials: torch.Tensor,
                     up_vec: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward 2 - potential based shaping + heading projected speed.

      * potentials are stored as -dist/dt, therefore
        (potentials - prev_potentials) * dt = distance gained towards the goal
        this step -> dense progress (shaping) signal
      * velocity projected on the heading (forward) direction gives the running speed
      * uprightness bonus so the agent learns to run instead of falling forward
    """
    # progress towards the goal this step (metres)
    progress = (potentials - prev_potentials) * dt
    progress_reward = torch.clamp(progress, min=-0.5, max=0.5)

    # velocity projected on the heading direction
    vel_xy = root_states[:, 7:9]
    heading_xy = heading_vec[:, 0:2]
    heading_xy = heading_xy / (torch.norm(heading_xy, p=2, dim=-1, keepdim=True) + 1.0e-6)
    forward_speed = torch.sum(vel_xy * heading_xy, dim=-1)
    speed_reward = torch.clamp(forward_speed, min=-1.0, max=10.0)

    # upright bonus
    upright_reward = torch.clamp(up_vec[:, 2], min=0.0, max=1.0)

    total = 2.0 * progress_reward + speed_reward + 0.5 * upright_reward

    components: Dict[str, torch.Tensor] = {
        "progress": 2.0 * progress_reward,
        "forward_speed": speed_reward,
        "upright": upright_reward,
    }
    return total, components
