@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, heading_vec: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear velocity from root_states
    linear_velocity = root_states[:, 7:10] # shape (num_envs, 3)

    # 1. Forward velocity reward: Reward for positive velocity in the heading direction.
    # heading_vec is a 3D vector representing the ant's forward direction.
    forward_velocity = torch.sum(linear_velocity * heading_vec, dim=1)
    
    # Use an exponential transformation for the forward reward, with a temperature parameter.
    forward_reward_temp = 0.5
    forward_reward = torch.exp(forward_velocity * forward_reward_temp)

    # 2. Action penalty: Penalize large control efforts.
    action_penalty_temp = 0.1
    action_penalty = torch.sum(torch.square(actions), dim=-1)
    action_reward = -torch.exp(action_penalty * action_penalty_temp) # Exponential penalty for larger actions

    total_reward = forward_reward + action_reward

    rewards = {
        "forward_reward": forward_reward,
        "action_reward": action_reward,
    }
    return total_reward, rewards
