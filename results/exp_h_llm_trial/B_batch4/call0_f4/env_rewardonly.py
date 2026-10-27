@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor,
                     actions: torch.Tensor,
                     dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Goal-directed straight running: speed towards the target direction, heading alignment,
       upright bonus and penalties for spinning / large control effort."""

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

    pos = root_states[:, 0:3]
    to_target = targets - pos
    to_target[:, 2] = 0.0
    dist = torch.norm(to_target, p=2, dim=-1, keepdim=True)
    dir_target = (to_target / torch.clamp(dist, min=1.0e-6)).squeeze(-1)   # (N, 3)

    forward_speed = torch.sum(vel_h * fwd_h, dim=-1)
    speed_target = torch.sum(vel_h * dir_target, dim=-1)
    align = torch.sum(fwd_h * dir_target, dim=-1)
    upright = torch.clamp(up_world[:, 2], min=-1.0, max=1.0)

    ang_vel = root_states[:, 10:13]
    yaw_rate = ang_vel[:, 2]
    ctrl_cost = torch.sum(actions * actions, dim=-1)
    joint_cost = torch.sum(dof_vel * dof_vel, dim=-1)

    temp_speed: float = 3.0
    speed_term = torch.tanh(speed_target / temp_speed)
    forward_term = torch.tanh(forward_speed / temp_speed)

    w_target_speed: float = 1.0
    w_forward_speed: float = 0.5
    w_align: float = 0.3
    w_upright: float = 0.3
    w_yaw: float = 0.02
    w_ctrl: float = 0.005
    w_joint: float = 0.0002

    reward = (w_target_speed * speed_term
              + w_forward_speed * forward_term
              + w_align * align
              + w_upright * upright
              - w_yaw * yaw_rate * yaw_rate
              - w_ctrl * ctrl_cost
              - w_joint * joint_cost)

    info: Dict[str, torch.Tensor] = {"speed_target": speed_target,
                                     "forward_speed": forward_speed,
                                     "align": align,
                                     "upright": upright,
                                     "yaw_rate": yaw_rate,
                                     "ctrl_cost": ctrl_cost}
    return reward, info
