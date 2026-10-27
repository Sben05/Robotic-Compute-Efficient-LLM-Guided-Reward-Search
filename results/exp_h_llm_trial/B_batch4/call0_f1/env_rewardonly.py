@torch.jit.script
def compute_reward(root_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Pure 'run forward' reward: horizontal speed along the ant's own body-forward axis."""

    # ---- body axes expressed in world frame (Isaac Gym quaternions are xyzw) ----
    quat = root_states[:, 3:7]
    q_vec = quat[:, 0:3]
    q_w = quat[:, 3]

    fwd = torch.zeros_like(q_vec)
    fwd[:, 0] = 1.0
    cross_qf = torch.cross(q_vec, fwd, dim=-1)
    fwd_world = fwd + 2.0 * q_w.unsqueeze(-1) * cross_qf + 2.0 * torch.cross(q_vec, cross_qf, dim=-1)

    # ---- horizontal (world x-y) parts of forward direction and velocity ----
    fwd_h = fwd_world.clone()
    fwd_h[:, 2] = 0.0
    fwd_h = fwd_h / torch.clamp(torch.norm(fwd_h, p=2, dim=-1, keepdim=True), min=1.0e-6)

    vel = root_states[:, 7:10]
    vel_h = vel.clone()
    vel_h[:, 2] = 0.0

    forward_speed = torch.sum(vel_h * fwd_h, dim=-1)   # m/s, >0 means running forward

    # bounded, smooth transformation of the speed component
    temp_speed: float = 3.0
    speed_reward = torch.tanh(forward_speed / temp_speed)

    reward = speed_reward

    info: Dict[str, torch.Tensor] = {"forward_speed": forward_speed,
                                     "speed_reward": speed_reward}
    return reward, info
