@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dof_vel: torch.Tensor, inv_start_rot: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant states from root_states
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7] # (x, y, z, w) quaternion
    velocity = root_states[:, 7:10]      # Linear velocity (global frame)
    ang_velocity = root_states[:, 10:13] # Angular velocity (global frame)

    # Compute local forward velocity
    torso_quat = quat_mul(torso_rotation, inv_start_rot)
    vel_loc = quat_rotate_inverse(torso_quat, velocity)
    forward_velocity = vel_loc[:, 0]

    # Torso height
    height = torso_position[:, up_axis_idx]

    # Reward components
    # 1. Exponentially scaled forward velocity for stronger incentive
    forward_vel_temp = 2.0
    forward_reward = torch.exp(forward_vel_temp * forward_velocity)

    # 2. Uprightness: Reward for being close to a target height, penalizing deviations exponentially
    target_height = 0.75 # meters
    height_deviate_temp = 5.0
    height_reward = torch.exp(-height_deviate_temp * torch.square(height - target_height))

    # 3. Penalize large actions with exponential decay for smoothness and stability
    action_penalty_temp = 0.01
    action_penalty = -torch.exp(action_penalty_temp * torch.sum(torch.square(actions), dim=-1)) # Negative exponential for penalty
    
    # 4. Penalize high angular velocity to encourage stable, straight-line movement
    ang_vel_temp = 0.05
    angular_velocity_penalty = -ang_vel_temp * torch.sum(torch.square(ang_velocity), dim=-1)

    # 5. Penalize high joint velocities to prevent jerky movements
    dof_vel_penalty_temp = 0.005
    dof_velocity_penalty = -dof_vel_penalty_temp * torch.sum(torch.square(dof_vel), dim=-1)

    total_reward = (forward_reward +
                    height_reward +
                    angular_velocity_penalty +
                    dof_velocity_penalty +
                    action_penalty)

    rewards = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "angular_velocity_penalty": angular_velocity_penalty,
        "dof_velocity_penalty": dof_velocity_penalty,
        "action_penalty": action_penalty,
    }

    return total_reward, rewards
