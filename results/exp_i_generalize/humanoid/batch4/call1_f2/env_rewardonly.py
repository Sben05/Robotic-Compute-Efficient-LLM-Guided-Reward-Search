@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, dof_vel: torch.Tensor,
                     actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Exponential (temperature shaped) running reward with stability / fall penalties."""
    vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]
    quat = root_states[:, 3:7]
    qw = quat[:, 0:1]
    qxyz = quat[:, 1:4]

    zeros = torch.zeros_like(qw)
    e_x = torch.cat((torch.ones_like(zeros), zeros, zeros), dim=-1)
    t1 = torch.cross(qxyz, e_x, dim=-1)
    t2 = torch.cross(qxyz, t1, dim=-1)
    fwd = e_x + 2.0 * (qw * t1 + t2)
    world_up = torch.cat((zeros, zeros, torch.ones_like(zeros)), dim=-1)
    side = torch.cross(fwd, world_up, dim=-1)          # horizontal, perpendicular to heading

    speed = torch.sum(vel * fwd, dim=-1)
    lateral = torch.sum(vel * side, dim=-1)
    up_proj = up_vec[:, 2]

    # temperature parameters (one per transformed component)
    temp_speed: float = 0.5
    temp_upright: float = 4.0
    temp_lateral: float = 2.0

    speed_reward = torch.exp(temp_speed * speed) - 1.0
    upright_reward = torch.exp(temp_upright * (up_proj - 1.0))
    lateral_penalty = -temp_lateral * torch.clamp(lateral, min=0.0) \
        - 0.05 * torch.clamp(-lateral, min=0.0)
    twist_penalty = -0.02 * torch.sum(ang_vel * ang_vel, dim=-1)
    bob_penalty = -0.05 * torch.abs(vel[:, 2])
    fall_penalty = -5.0 * torch.clamp(-up_proj, min=0.0) \
        - 2.0 * torch.clamp(0.8 - root_states[:, 2], min=0.0)
    effort_penalty = -0.005 * torch.sum(actions * actions, dim=-1) \
        - 0.002 * torch.sum(dof_vel * dof_vel, dim=-1)
    alive_bonus = 0.5 * torch.ones_like(speed)

    reward = 2.0 * speed_reward + 0.5 * upright_reward + alive_bonus \
        + lateral_penalty + twist_penalty + bob_penalty + fall_penalty + effort_penalty

    components: Dict[str, torch.Tensor] = {}
    components["speed_exp"] = 2.0 * speed_reward
    components["upright_exp"] = 0.5 * upright_reward
    components["alive_bonus"] = alive_bonus
    components["lateral_drift"] = lateral_penalty
    components["twist_penalty"] = twist_penalty
    components["vertical_bob"] = bob_penalty
    components["fall_penalty"] = fall_penalty
    components["effort_penalty"] = effort_penalty
    return reward, components
