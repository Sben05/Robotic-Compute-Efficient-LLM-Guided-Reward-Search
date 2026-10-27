@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, heading_vec: torch.Tensor, up_vec: torch.Tensor, dof_vel: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract linear, angular velocity, and torso position (Z-component)
    linear_velocity = root_states[:, 7:10]
    angular_velocity = root_states[:, 10:13]
    torso_position_z = root_states[:, up_axis_idx]
    device = root_states.device

    # 1. Forward velocity reward:
    forward_velocity = torch.sum(linear_velocity * heading_vec, dim=1)
    forward_reward_temp = 0.5
    forward_reward = torch.exp(forward_velocity * forward_reward_temp)

    # 2. Action penalty:
    action_penalty_temp = 0.1
    action_penalty = torch.sum(torch.square(actions), dim=-1)
    action_reward = -torch.exp(action_penalty * action_penalty_temp)

    # 3. Uprightness reward:
    global_up_vec = torch.zeros_like(up_vec, device=device)
    global_up_vec[:, up_axis_idx] = 1.0
    upright_proj = torch.sum(up_vec * global_up_vec, dim=1)
    upright_reward_temp = 2.0
    upright_reward = torch.exp((upright_proj - 1.0) * upright_reward_temp)

    # 4. Angular velocity penalty:
    angular_velocity_magnitude = torch.norm(angular_velocity, dim=1)
    ang_vel_penalty_temp = 0.05
    ang_vel_reward = -torch.exp(angular_velocity_magnitude * ang_vel_penalty_temp)

    # 5. Height maintenance reward: Reward for staying near a target height.
    # A typical Ant torso height is around 0.6 meters.
    target_height = 0.6
    height_deviation = torch.abs(torso_position_z - target_height)
    height_reward_temp = 5.0
    height_reward = torch.exp(-height_deviation * height_reward_temp) # Exponentially penalize deviation from target height

    # 6. DOF velocity penalty: Penalize large joint velocities to encourage smoother movements.
    dof_vel_magnitude = torch.norm(dof_vel, dim=1) # Sum of squares of all joint velocities
    dof_vel_penalty_temp = 0.005
    dof_vel_reward = -torch.exp(dof_vel_magnitude * dof_vel_penalty_temp)

    total_reward = forward_reward + action_reward + upright_reward + ang_vel_reward + height_reward + dof_vel_reward

    rewards = {
        "forward_reward": forward_reward,
        "action_reward": action_reward,
        "upright_reward": upright_reward,
        "angular_velocity_reward": ang_vel_reward,
        "height_reward": height_reward,
        "dof_vel_reward": dof_vel_reward,
    }
    return total_reward, rewards
