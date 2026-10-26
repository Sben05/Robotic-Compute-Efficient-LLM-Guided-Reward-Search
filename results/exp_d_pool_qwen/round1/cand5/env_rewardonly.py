@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos: torch.Tensor,
    actions: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity of the torso
    velocity = root_states[:, 7:10]
    torso_position = root_states[:, 0:3]

    # Compute forward direction (toward target on ground plane)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # zero out vertical component
    to_target_norm = torch.clamp(torch.norm(to_target, p=2, dim=-1, keepdim=True), min=1e-6)
    forward_direction = to_target / to_target_norm

    # Project velocity onto forward direction -> forward speed
    forward_speed = torch.sum(velocity * forward_direction, dim=-1)

    # Upright score: z-component of the ant's up vector (1.0 when fully upright)
    up_score = up_vec[:, 2]

    # Temperature parameters for reward shaping
    speed_temp: float = 1.0
    up_temp: float = 20.0
    action_temp: float = 0.001
    joint_vel_temp: float = 0.001

    # Primary reward: forward speed (linear, positive for forward motion)
    speed_reward = forward_speed * speed_temp

    # Upright reward: exponentially scaled so staying upright is strongly rewarded
    up_reward = torch.exp(up_score * up_temp) - 1.0

    # Penalties for large actions and joint velocities (encourage smooth, efficient motion)
    action_penalty = -action_temp * torch.sum(actions * actions, dim=-1)
    joint_vel_penalty = -joint_vel_temp * torch.sum(dof_vel * dof_vel, dim=-1)

    # Total reward
    reward = speed_reward + up_reward + action_penalty + joint_vel_penalty

    reward_dict: Dict[str, torch.Tensor] = {
        "speed_reward": speed_reward,
        "up_reward": up_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty,
    }

    return reward, reward_dict
