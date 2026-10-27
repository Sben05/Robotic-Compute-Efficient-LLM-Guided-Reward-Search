@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, heading_vec: torch.Tensor, up_vec: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear and angular velocity from root_states
    linear_velocity = root_states[:, 7:10]  # shape (num_envs, 3)
    angular_velocity = root_states[:, 10:13] # shape (num_envs, 3)
    device = root_states.device

    # 1. Forward velocity reward: Reward for positive velocity in the heading direction.
    forward_velocity = torch.sum(linear_velocity * heading_vec, dim=1)
    forward_reward_temp = 0.5
    forward_reward = torch.exp(forward_velocity * forward_reward_temp)

    # 2. Action penalty: Penalize large control efforts.
    action_penalty_temp = 0.1
    action_penalty = torch.sum(torch.square(actions), dim=-1)
    action_reward = -torch.exp(action_penalty * action_penalty_temp)

    # 3. Uprightness reward: Reward for the ant's torso being upright.
    # up_vec is the torso's up vector in the global frame. Global up vector is (0,0,1) for Z-up.
    global_up_vec = torch.zeros_like(up_vec, device=device)
    global_up_vec[:, up_axis_idx] = 1.0
    upright_proj = torch.sum(up_vec * global_up_vec, dim=1) # Dot product between torso up and global up
    upright_reward_temp = 2.0
    # Reward for being perfectly upright (upright_proj = 1.0). Penalize deviation.
    upright_reward = torch.exp((upright_proj - 1.0) * upright_reward_temp) 

    # 4. Angular velocity penalty: Penalize any significant angular velocity to encourage stable locomotion.
    angular_velocity_magnitude = torch.norm(angular_velocity, dim=1)
    ang_vel_penalty_temp = 0.05
    ang_vel_reward = -torch.exp(angular_velocity_magnitude * ang_vel_penalty_temp)

    total_reward = forward_reward + action_reward + upright_reward + ang_vel_reward

    rewards = {
        "forward_reward": forward_reward,
        "action_reward": action_reward,
        "upright_reward": upright_reward,
        "angular_velocity_reward": ang_vel_reward,
    }
    return total_reward, rewards
