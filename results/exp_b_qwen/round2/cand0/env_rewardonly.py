@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, actions: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Unit direction to target (2D)
    to_target = targets - torso_position
    to_target = to_target.clone()
    to_target[:, 2] = 0.0

    target_norm = torch.norm(to_target, p=2, dim=-1, keepdim=True)
    target_norm = torch.clamp(target_norm, min=1e-6)
    target_dir = to_target / target_norm

    # Forward velocity (primary signal)
    forward_vel = torch.sum(velocity * target_dir, dim=-1)

    # Quadratic speed bonus: super-linear incentive to go faster
    # At 7 m/s: 0.04 * 49 = 1.96; at 10 m/s: 0.04 * 100 = 4.0
    quadratic_speed_temperature = 0.04
    speed_bonus = quadratic_speed_temperature * forward_vel * forward_vel

    # Height maintenance: target ~0.5 m (natural ant running crouch)
    height = torso_position[:, up_axis_idx]
    height_target = 0.5
    height_reward = -torch.abs(height - height_target)

    # Small angular velocity penalty to encourage stable, efficient gait
    ang_vel = root_states[:, 10:13]
    ang_vel_penalty = -0.01 * torch.sum(ang_vel * ang_vel, dim=-1)

    # Action regularization
    action_penalty_temperature = 0.001
    action_penalty = action_penalty_temperature * torch.sum(actions * actions, dim=-1)

    # Total reward
    forward_reward = forward_vel
    reward = forward_reward + speed_bonus + 1.0 * height_reward + ang_vel_penalty - action_penalty

    components: Dict[str, torch.Tensor] = {
        "forward_reward": forward_reward,
        "speed_bonus": speed_bonus,
        "height_reward": height_reward,
        "ang_vel_penalty": ang_vel_penalty,
        "action_penalty": action_penalty,
    }

    return reward, components
