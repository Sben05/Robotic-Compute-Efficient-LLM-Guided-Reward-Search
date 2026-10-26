@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor,
                   dof_vel: torch.Tensor, actions: torch.Tensor,
                   up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso position and velocity from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]

    # Compute direction toward target (horizontal)
    to_target = targets - torso_position
    to_target_norm = torch.norm(to_target, p=2, dim=-1, keepdim=True) + 1e-8
    target_direction = to_target / to_target_norm

    # Forward velocity: projection of velocity onto target direction
    forward_vel = (velocity * target_direction).sum(dim=-1)

    # Height bonus: encourage staying upright (Ant ~0.8m tall)
    height = torso_position[:, up_axis_idx]
    height_reward = (height - 0.8).clamp(min=-1.0)

    # Regularization: penalize large actions and joint velocities
    action_cost = (actions * actions).sum(dim=-1)
    dof_vel_cost = (dof_vel * dof_vel).sum(dim=-1)

    # Temperature parameters for shaping
    temp_forward = 1.0
    temp_height = 2.0
    temp_action = 1.0
    temp_dof_vel = 1.0

    # Shape rewards using soft functions for better gradient signal
    reward_forward = forward_vel * temp_forward
    reward_height = torch.tanh(height_reward * temp_height) * 0.5
    reward_action = -0.05 * action_cost * temp_action
    reward_dof_vel = -0.02 * dof_vel_cost * temp_dof_vel

    # Total reward: strongly incentivize forward speed, moderate height, mild regularization
    reward = reward_forward + reward_height + reward_action + reward_dof_vel

    rewards_dict = {
        "forward": reward_forward,
        "height": reward_height,
        "action": reward_action,
        "dof_vel": reward_dof_vel,
    }

    return reward, rewards_dict
