@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, actions: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # Unit direction toward target in the horizontal plane
    to_target = targets - torso_position
    to_target = to_target.clone()
    to_target[:, 2] = 0.0
    target_norm = torch.norm(to_target, p=2, dim=-1, keepdim=True)
    target_norm = torch.clamp(target_norm, min=1e-6)
    target_dir = to_target / target_norm

    # Project velocity onto target direction (forward speed)
    forward_vel = torch.sum(velocity * target_dir, dim=-1)

    # Quadratic speed bonus: marginal reward grows with speed, pushing beyond plateau
    speed_temperature = 0.15
    speed_bonus = speed_temperature * forward_vel * forward_vel

    # Conditional height penalty: only activates when ant drops below 0.5
    height = torso_position[:, up_axis_idx]
    height_drop = torch.clamp(0.5 - height, min=0.0)
    height_penalty = 2.0 * height_drop

    # Angular velocity penalty: rewards smooth, stable gait
    ang_vel_sq = torch.sum(ang_velocity * ang_velocity, dim=-1)
    ang_vel_penalty = 0.05 * ang_vel_sq

    # Action regularisation (slightly stronger than before)
    action_sq = torch.sum(actions * actions, dim=-1)
    action_penalty = 0.005 * action_sq

    reward = forward_vel + speed_bonus - height_penalty - ang_vel_penalty - action_penalty

    components: Dict[str, torch.Tensor] = {}
    components["forward_reward"] = forward_vel
    components["speed_bonus"] = speed_bonus
    components["height_penalty"] = -height_penalty
    components["ang_vel_penalty"] = -ang_vel_penalty
    components["action_penalty"] = -action_penalty

    return reward, components
