@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor, dof_limits_lower: torch.Tensor, dof_limits_upper: torch.Tensor, inv_start_rot: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
    # 1. Exponentially scaled forward velocity for strong incentive
    forward_vel_temp = 2.5
    forward_reward = torch.exp(forward_vel_temp * forward_velocity)

    # 2. Uprightness: Reward for being within an acceptable height range, exponential penalty outside
    target_height = 0.75 # meters
    height_tolerance = 0.2 # meters
    height_reward_temp = 10.0
    height_reward = torch.where(
        (height >= target_height - height_tolerance) & (height <= target_height + height_tolerance),
        torch.ones_like(height) * 1.0, # Full reward if within tolerance
        torch.exp(-height_reward_temp * torch.square(height - target_height)) # Exponential penalty outside tolerance
    )

    # 3. Penalize excessive actions (L2 norm)
    action_penalty_temp = 0.005
    action_penalty = -action_penalty_temp * torch.sum(torch.square(actions), dim=-1)
    
    # 4. Penalize high joint velocities for smoother and more natural movement
    dof_vel_penalty_temp = 0.003
    dof_velocity_penalty = -dof_vel_penalty_temp * torch.sum(torch.square(dof_vel), dim=-1)

    # 5. Penalize joint positions being close to their limits to avoid uncomfortable poses
    # Normalize dof_pos to [-1, 1] range
    range_size = dof_limits_upper - dof_limits_lower
    # Replace zero ranges with 1.0 to avoid division by zero (e.g., for fixed joints)
    range_size = torch.where(range_size == 0, torch.ones_like(range_size), range_size)
    dof_pos_scaled = 2.0 * (dof_pos - dof_limits_lower) / range_size - 1.0
    
    limit_threshold = 0.9 # Normalized position close to 1 or -1
    joint_limit_penalty_temp = 0.2
    joint_limit_penalty = -joint_limit_penalty_temp * torch.sum(torch.where(
        (dof_pos_scaled.abs() > limit_threshold),
        torch.square(dof_pos_scaled.abs() - limit_threshold),
        torch.zeros_like(dof_pos_scaled)
    ), dim=-1)

    # 6. Penalize high angular velocity around body y and z axes (pitch and yaw) to encourage straight movement
    # Assuming local x is forward, y is left, z is up. ang_velocity is in global frame.
    # vel_loc, angvel_loc = quat_rotate_inverse(torso_quat, velocity/ang_velocity)
    # So we need angvel_loc. Let's recompute it for clarity.
    angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)
    ang_vel_yz_temp = 0.08
    angular_stability_penalty = -ang_vel_yz_temp * torch.sum(torch.square(angvel_loc[:, 1:3]), dim=-1)


    total_reward = (forward_reward +
                    height_reward +
                    action_penalty +
                    dof_velocity_penalty +
                    joint_limit_penalty +
                    angular_stability_penalty)

    rewards = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "action_penalty": action_penalty,
        "dof_velocity_penalty": dof_velocity_penalty,
        "joint_limit_penalty": joint_limit_penalty,
        "angular_stability_penalty": angular_stability_penalty,
    }

    return total_reward, rewards
