@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    up_vec: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso position and linear velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Compute heading direction in the xz-plane (toward target)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_norm = torch.norm(to_target, p=2, dim=-1, keepdim=True) + 1e-6
    heading_dir = to_target / target_norm

    # Forward velocity: project linear velocity onto the heading direction
    forward_vel = torch.sum(velocity * heading_dir, dim=-1)

    # Uprightness: z-component of the torso up-vector (1.0 when fully upright)
    upright = up_vec[:, 2]

    # Action magnitude penalty for encouraging efficient movement
    action_magnitude = torch.sum(actions * actions, dim=-1)

    # Temperature / scaling parameters
    forward_temp = 1.0
    upright_temp = 1.0
    action_temp = 0.05

    # Individual reward components
    forward_reward = forward_vel * forward_temp
    upright_reward = upright * upright_temp
    action_reward = -action_magnitude * action_temp

    # Total reward
    reward = forward_reward + upright_reward + action_reward

    rewards = {
        "forward_vel": forward_reward,
        "upright": upright_reward,
        "action_penalty": action_reward,
    }

    return reward, rewards
