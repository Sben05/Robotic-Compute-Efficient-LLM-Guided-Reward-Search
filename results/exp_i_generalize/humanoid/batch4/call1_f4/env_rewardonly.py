@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, up_vec: torch.Tensor,
                     heading_vec: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Target directed running: exponential speed score gated by goal alignment."""
    vel = root_states[:, 7:10]
    quat = root_states[:, 3:7]
    qw = quat[:, 0:1]
    qxyz = quat[:, 1:4]

    zeros = torch.zeros_like(qw)
    e_x = torch.cat((torch.ones_like(zeros), zeros, zeros), dim=-1)
    t1 = torch.cross(qxyz, e_x, dim=-1)
    t2 = torch.cross(qxyz, t1, dim=-1)
    fwd = e_x + 2.0 * (qw * t1 + t2)

    # horizontal unit vector pointing to the goal
    dx = targets[:, 0:1] - root_states[:, 0:1]
    dy = targets[:, 1:2] - root_states[:, 1:2]
    dz = torch.zeros_like(dx)
    to_target = torch.cat((dx, dy, dz), dim=-1)
    dist = torch.sqrt(torch.sum(to_target * to_target, dim=-1) + 1e-6)
    dir_target = to_target / dist.unsqueeze(-1)

    speed = torch.sum(vel * fwd, dim=-1)
    speed_to_goal = torch.sum(vel * dir_target, dim=-1)
    align = torch.clamp(torch.sum(fwd * dir_target, dim=-1), min=0.0, max=1.0)
    up_proj = up_vec[:, 2]

    # temperature parameters (one per transformed component)
    temp_speed: float = 0.4
    temp_align: float = 2.0
    temp_upright: float = 5.0

    speed_reward = torch.exp(temp_speed * speed) - 1.0
    goal_speed_reward = torch.exp(temp_speed * speed_to_goal) - 1.0
    align_reward = torch.exp(temp_align * (align - 1.0))
    upright_reward = torch.exp(temp_upright * (up_proj - 1.0))

    # heading consistency with the initial run direction (start frame)
    hnorm = heading_vec / (torch.sqrt(torch.sum(heading_vec * heading_vec, dim=-1, keepdim=True)) + 1e-6)
    heading_consistency = torch.clamp(torch.sum(hnorm * fwd, dim=-1), min=0.0, max=1.0)

    fall_penalty = -5.0 * torch.clamp(-up_proj, min=0.0) \
        - 2.0 * torch.clamp(0.8 - root_states[:, 2], min=0.0)
    height_reward = 0.2 * torch.exp(-torch.abs(root_states[:, 2] - 1.3) / 0.3)
    effort_penalty = -0.005 * torch.sum(actions * actions, dim=-1) \
        - 0.002 * torch.sum(dof_vel * dof_vel, dim=-1)

    reward = 2.0 * speed_reward + 1.0 * goal_speed_reward + 0.5 * align_reward \
        + 0.5 * upright_reward + 0.3 * heading_consistency + height_reward \
        + fall_penalty + effort_penalty

    components: Dict[str, torch.Tensor] = {}
    components["speed_exp"] = 2.0 * speed_reward
    components["goal_speed_exp"] = 1.0 * goal_speed_reward
    components["goal_alignment"] = 0.5 * align_reward
    components["upright_exp"] = 0.5 * upright_reward
    components["heading_consistency"] = 0.3 * heading_consistency
    components["torso_height"] = height_reward
    components["fall_penalty"] = fall_penalty
    components["effort_penalty"] = effort_penalty
    return reward, components
