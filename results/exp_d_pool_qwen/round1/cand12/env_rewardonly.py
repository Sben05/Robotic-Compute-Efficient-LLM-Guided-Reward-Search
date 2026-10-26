@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    up_axis_idx: int,
    dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # Direction to target in 2D (ignore z)
    to_target = targets - torso_position
    to_target_2d = torch.stack((to_target[:, 0], to_target[:, 1]), dim=-1)
    to_target_dist = torch.norm(to_target_2d, p=2, dim=-1, keepdim=True) + 1e-6
    to_target_dir_2d = to_target_2d / to_target_dist
    # Extend to 3D with zero z-component for dot product with velocity
    to_target_dir = torch.stack(
        (to_target_dir_2d[:, 0], to_target_dir_2d[:, 1], torch.zeros_like(to_target_dist[:, 0])),
        dim=-1,
    )

    # Forward velocity: how fast the ant moves toward the target
    forward_vel = (velocity * to_target_dir).sum(dim=-1)

    # Temperature parameters
    upright_temp: float = 0.4
    ang_vel_temp: float = 0.3

    # --- Reward components ---

    # 1) Forward velocity reward (primary objective: run as fast as possible)
    forward_vel_reward = forward_vel * 10.0

    # 2) Uprightness reward: keep torso at a reasonable height
    height = torso_position[:, up_axis_idx]
    target_height = 0.8
    upright_reward = torch.exp(-((height - target_height) ** 2) / upright_temp) - 1.0

    # 3) Angular velocity penalty: keep the ant stable
    ang_vel_penalty = -torch.exp(torch.sum(ang_velocity ** 2, dim=-1) / ang_vel_temp) + 1.0

    # 4) Action regularization: discourage excessive joint torques
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.01

    # Total reward
    total_reward = forward_vel_reward + upright_reward + ang_vel_penalty + action_penalty

    components: Dict[str, torch.Tensor] = {}
    components["forward_vel"] = forward_vel_reward
    components["upright"] = upright_reward
    components["ang_vel_penalty"] = ang_vel_penalty
    components["action_penalty"] = action_penalty

    return total_reward, components
