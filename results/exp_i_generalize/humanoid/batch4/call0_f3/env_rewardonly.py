@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_vel: torch.Tensor,
                     actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pos = root_states[:, 0:3]
    vel = root_states[:, 7:10]

    # -------- desired running direction (horizontal) --------
    to_target = targets - pos
    to_target[:, 2] = 0.0
    dist = torch.norm(to_target, p=2, dim=-1)
    dir_xy = to_target / (dist.unsqueeze(-1) + 1e-6)
    forward_speed = vel[:, 0] * dir_xy[:, 0] + vel[:, 1] * dir_xy[:, 1]
    lateral_speed = -vel[:, 0] * dir_xy[:, 1] + vel[:, 1] * dir_xy[:, 0]

    # -------- get close to a fast running speed --------
    desired_speed = 3.0
    temp_speed = 1.0
    speed_reward = torch.exp(-torch.abs(forward_speed - desired_speed) / temp_speed)

    # -------- push beyond the desired speed --------
    over_speed = torch.clamp(forward_speed - desired_speed, min=0.0)
    temp_over = 2.0
    over_reward = 1.0 - torch.exp(-over_speed / temp_over)

    # -------- uprightness --------
    q = root_states[:, 3:7]
    up_z = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
    up_quality = torch.clamp(up_z, min=0.0, max=1.0)
    temp_upright = 0.5
    upright_reward = 1.0 - torch.exp(-up_quality / temp_upright)

    # -------- alive bonus --------
    alive_reward = torch.clamp(root_states[:, 2] - 0.6, min=0.0, max=1.0)

    # -------- penalties --------
    lateral_penalty = 0.05 * lateral_speed * lateral_speed
    action_penalty = 0.03 * torch.sum(actions * actions, dim=-1)
    dof_vel_penalty = 1.0e-3 * torch.sum(dof_vel * dof_vel, dim=-1)

    reward = 2.0 * speed_reward + 1.0 * over_reward + 0.5 * upright_reward + 0.5 * alive_reward \
        - lateral_penalty - action_penalty - dof_vel_penalty

    components = {
        "target_speed": 2.0 * speed_reward,
        "over_speed": 1.0 * over_reward,
        "upright": 0.5 * upright_reward,
        "alive": 0.5 * alive_reward,
        "lateral_penalty": -lateral_penalty,
        "action_penalty": -action_penalty,
        "dof_vel_penalty": -dof_vel_penalty,
    }
    return reward, components
