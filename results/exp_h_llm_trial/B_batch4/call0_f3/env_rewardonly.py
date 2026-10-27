@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor,
                     prev_potentials: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Exponentially shaped forward speed combined with potential-based progress shaping."""

    quat = root_states[:, 3:7]
    q_vec = quat[:, 0:3]
    q_w = quat[:, 3]

    fwd = torch.zeros_like(q_vec)
    fwd[:, 0] = 1.0
    cross_qf = torch.cross(q_vec, fwd, dim=-1)
    fwd_world = fwd + 2.0 * q_w.unsqueeze(-1) * cross_qf + 2.0 * torch.cross(q_vec, cross_qf, dim=-1)

    up = torch.zeros_like(q_vec)
    up[:, 2] = 1.0
    cross_qu = torch.cross(q_vec, up, dim=-1)
    up_world = up + 2.0 * q_w.unsqueeze(-1) * cross_qu + 2.0 * torch.cross(q_vec, cross_qu, dim=-1)

    fwd_h = fwd_world.clone()
    fwd_h[:, 2] = 0.0
    fwd_h = fwd_h / torch.clamp(torch.norm(fwd_h, p=2, dim=-1, keepdim=True), min=1.0e-6)

    vel = root_states[:, 7:10]
    vel_h = vel.clone()
    vel_h[:, 2] = 0.0
    forward_speed = torch.sum(vel_h * fwd_h, dim=-1)
    upright = torch.clamp(up_world[:, 2], min=-1.0, max=1.0)

    # potentials = -dist/dt  ->  (potentials - prev_potentials) * dt = dist_old - dist_new
    # (positive when the ant got closer to its target during this step)
    progress = (potentials - prev_potentials) * dt

    temp_speed: float = 4.0
    temp_progress: float = 0.25

    speed_term = torch.exp(torch.clamp(forward_speed, min=-3.0, max=8.0) / temp_speed)
    progress_term = torch.exp(torch.clamp(progress, min=-1.0, max=1.0) / temp_progress)

    w_speed: float = 1.0
    w_progress: float = 0.3
    w_upright: float = 0.3

    reward = w_speed * speed_term + w_progress * progress_term + w_upright * upright

    info: Dict[str, torch.Tensor] = {"forward_speed": forward_speed,
                                     "progress": progress,
                                     "speed_term": speed_term,
                                     "progress_term": progress_term,
                                     "upright": upright}
    return reward, info
