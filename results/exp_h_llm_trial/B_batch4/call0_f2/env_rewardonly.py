@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward speed with stability / regularisation terms (no spinning, sliding or wild torques)."""

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

    # sideways slipping relative to the body frame
    lateral_vel = vel_h - forward_speed.unsqueeze(-1) * fwd_h
    lateral_speed = torch.norm(lateral_vel, p=2, dim=-1)

    ang_vel = root_states[:, 10:13]
    yaw_rate = ang_vel[:, 2]

    ctrl_cost = torch.sum(actions * actions, dim=-1)
    joint_cost = torch.sum(dof_vel * dof_vel, dim=-1)

    w_speed: float = 1.0
    w_upright: float = 0.5
    w_lateral: float = 0.1
    w_ctrl: float = 0.005
    w_joint: float = 0.0002
    w_yaw: float = 0.02
    alive_bonus: float = 0.5

    speed_term = torch.clamp(forward_speed, min=-2.0, max=8.0)
    reward = (w_speed * speed_term
              + w_upright * upright
              + alive_bonus * torch.ones_like(speed_term)
              - w_lateral * lateral_speed
              - w_ctrl * ctrl_cost
              - w_joint * joint_cost
              - w_yaw * yaw_rate * yaw_rate)

    info: Dict[str, torch.Tensor] = {"forward_speed": forward_speed,
                                     "upright": upright,
                                     "lateral_speed": lateral_speed,
                                     "yaw_rate": yaw_rate,
                                     "ctrl_cost": ctrl_cost}
    return reward, info
