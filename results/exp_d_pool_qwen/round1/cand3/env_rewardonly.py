@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    up_vec: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity (world frame)
    velocity = root_states[:, 7:10]  # (num_envs, 3)
    torso_pos = root_states[:, 0:3]  # (num_envs, 3)

    # Compute forward direction towards target (ignore z)
    to_target = targets - torso_pos  # (num_envs, 3)
    to_target = torch.cat([to_target[:, 0:1], to_target[:, 1:2], torch.zeros_like(to_target[:, 2:3])], dim=-1)
    to_target_norm = to_target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    forward_dir = to_target / to_target_norm  # (num_envs, 3)

    # Reward 1: Forward velocity (primary objective)
    forward_vel = (velocity * forward_dir).sum(dim=-1)  # (num_envs,)
    temp_forward = 0.1
    forward_vel_reward = torch.exp(temp_forward * forward_vel) - 1.0

    # Reward 2: Uprightness (keep ant standing)
    # up_vec z-component should be close to 1 when upright
    upright_reward = up_vec[:, 2] - 1.0  # ranges from -2 to 0

    # Reward 3: Action regularization (penalize large actions)
    temp_action = 0.01
    action_penalty = temp_action * (actions ** 2).sum(dim=-1)

    # Combine rewards
    weight_forward = 1.0
    weight_upright = 0.5
    weight_action = 1.0

    total_reward = (weight_forward * forward_vel_reward +
                    weight_upright * upright_reward -
                    weight_action * action_penalty)

    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "upright_reward": upright_reward,
        "action_penalty": action_penalty,
    }

    return total_reward, reward_dict
