@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    up_axis_idx: int,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso position and linear velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # Compute direction to target (projected onto horizontal plane)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    to_target_norm = to_target / (torch.norm(to_target, dim=-1, keepdim=True) + 1e-8)

    # Primary reward: forward velocity along target direction
    forward_vel = (velocity * to_target_norm).sum(dim=-1)
    forward_temp = 1.0
    forward_reward = forward_vel / forward_temp

    # Alive/height bonus: encourage staying upright
    height = torso_position[:, up_axis_idx]
    height_temp = 0.2
    alive_bonus = torch.exp(-height / height_temp)  # higher when close to ground, lower when up

    # Better: reward for maintaining a minimum height
    height_bonus = torch.tanh(height / 0.25) * 0.1

    # Penalty for falling (height too low)
    fall_penalty = torch.relu(0.4 - height) * 5.0

    # Penalty for excessive angular velocity (tumbling)
    ang_vel_penalty = (ang_velocity ** 2).sum(dim=-1) * 0.01

    # Penalty for large joint velocities (wasting energy)
    dof_vel_penalty = (dof_vel ** 2).sum(dim=-1) * 0.001

    # Penalty for large actions (energy cost)
    action_penalty = (actions ** 2).sum(dim=-1) * 0.02

    # Total reward
    reward = forward_reward + height_bonus - fall_penalty - ang_vel_penalty - dof_vel_penalty - action_penalty

    reward_dict = {
        "forward_reward": forward_reward,
        "height_bonus": height_bonus,
        "fall_penalty": fall_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "dof_vel_penalty": dof_vel_penalty,
        "action_penalty": action_penalty,
    }

    return reward, reward_dict
