@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor,
    up_axis_idx: int,
    dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant state components
    velocity = root_states[:, 7:10]
    torso_position = root_states[:, 0:3]
    ang_velocity = root_states[:, 10:13]

    # Forward direction: direction toward target, projected onto ground plane
    to_target = targets - torso_position
    to_target[:, up_axis_idx] = 0.0
    to_target_norm = torch.norm(to_target, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    forward_dir = to_target / to_target_norm

    # Main reward: forward velocity component (drive ant to run forward fast)
    forward_vel = (velocity * forward_dir).sum(dim=-1)
    forward_reward = forward_vel

    # Upright reward: keep the ant's torso elevated so it doesn't flip over
    upright_reward = torso_position[:, up_axis_idx]

    # Penalties to encourage smooth, stable locomotion
    action_penalty = (actions * actions).sum(dim=-1)
    ang_vel_penalty = (ang_velocity * ang_velocity).sum(dim=-1)
    joint_vel_penalty = (dof_vel * dof_vel).sum(dim=-1)

    # Weighted combination: forward speed dominates, with stability shaping
    weight_forward = 10.0
    weight_upright = 1.0
    weight_action = 0.01
    weight_ang_vel = 0.01
    weight_joint_vel = 0.001

    reward = (
        weight_forward * forward_reward
        + weight_upright * upright_reward
        - weight_action * action_penalty
        - weight_ang_vel * ang_vel_penalty
        - weight_joint_vel * joint_vel_penalty
    )

    reward_components: Dict[str, torch.Tensor] = {
        "forward_reward": forward_reward,
        "upright_reward": upright_reward,
        "action_penalty": action_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "joint_vel_penalty": joint_vel_penalty,
    }

    return reward, reward_components
