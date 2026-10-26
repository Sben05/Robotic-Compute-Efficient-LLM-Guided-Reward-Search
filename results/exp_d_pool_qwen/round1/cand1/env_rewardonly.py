@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # Compute forward direction (projected onto horizontal plane, axis 2 is up)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    to_target_norm = torch.norm(to_target, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    forward_dir = to_target / to_target_norm

    # Forward velocity: dot product of velocity with forward direction
    forward_vel = (velocity * forward_dir).sum(dim=-1)

    # Height of the ant (z-axis is up)
    height = torso_position[:, 2]

    # Reward components
    # Primary reward: forward velocity (encourages fast forward motion)
    forward_reward = forward_vel

    # Fall penalty: if the ant is too low, it's likely fallen over
    fall = (height < 0.3).to(torch.float32)
    fall_penalty = fall * 10.0

    # Action penalty: discourage large joint torques for efficiency
    action_penalty = 0.001 * (actions ** 2).sum(dim=-1)

    # Angular velocity penalty: discourage tumbling/spinning
    ang_vel_penalty = 0.05 * (ang_velocity ** 2).sum(dim=-1)

    # Total reward
    reward = forward_reward - action_penalty - fall_penalty - 0.1 * ang_vel_penalty

    return reward, {
        "forward_vel": forward_reward,
        "action_penalty": action_penalty,
        "fall_penalty": fall_penalty,
        "ang_vel_penalty": ang_vel_penalty * 0.1,
    }
