@torch.jit.script
def compute_reward(
    # Input variables are attributes of the Ant environment class
    root_states: torch.Tensor,
    obs_buf: torch.Tensor,
    actions: torch.Tensor,
    up_axis_idx: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # --- Reward component temperatures and scales ---
    # These parameters can be tuned to balance the different reward components.
    # They are internal to the reward function and not input variables.
    
    # Temperature for forward velocity reward; higher values amplify reward for higher velocity.
    forward_vel_temp = 0.7 
    # Temperature for height reward; higher values make the agent more sensitive to height deviations.
    height_temp = 5.0
    # Desired height of the Ant's torso.
    height_target = 0.75 
    # Scale for angular velocity penalty; higher values penalize spinning more.
    ang_vel_cost_scale = 0.05
    # Scale for action effort penalty; higher values penalize larger control efforts.
    action_cost_scale = 0.01

    # --- 1. Forward velocity reward ---
    # `vel_loc` (local linear velocity) is stored in obs_buf at indices 1:4.
    # The first component of `vel_loc` (index 0) typically corresponds to forward velocity.
    vel_loc = obs_buf[:, 1:4]
    forward_velocity = vel_loc[:, 0]
    
    # Reward for positive forward velocity.
    # Using an exponential transformation to encourage higher velocities and normalize.
    # Negative velocities will result in a very small positive reward, implicitly penalizing backward movement.
    forward_reward = torch.exp(forward_vel_temp * forward_velocity)

    # --- 2. Height reward ---
    # `torso_position` is the first 3 elements of `root_states`.
    # `up_axis_idx` specifies which axis corresponds to height (e.g., 2 for Z-up).
    torso_position = root_states[:, 0:3]
    height = torso_position[:, up_axis_idx]
    
    # Reward for maintaining a desired torso height.
    # Uses an exponential of negative squared error, providing maximum reward at `height_target`.
    height_reward = torch.exp(-height_temp * torch.square(height - height_target))

    # --- 3. Angular velocity penalty ---
    # `angvel_loc` (local angular velocity) is stored in obs_buf at indices 4:7.
    angvel_loc = obs_buf[:, 4:7]
    # Calculate the magnitude of the angular velocity.
    ang_vel_norm = torch.norm(angvel_loc, p=2, dim=-1)
    
    # Penalize large angular velocities to encourage stable, straight movement.
    ang_vel_penalty = -ang_vel_cost_scale * ang_vel_norm

    # --- 4. Action effort penalty ---
    # Penalize the sum of squared actions to encourage energy-efficient and smooth movements.
    action_penalty = -action_cost_scale * torch.sum(torch.square(actions), dim=-1)

    # --- Combine all reward components ---
    total_reward = forward_reward + height_reward + ang_vel_penalty + action_penalty

    # Create a dictionary for individual reward components for analysis and debugging.
    rewards = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "ang_vel_penalty": ang_vel_penalty,
        "action_penalty": action_penalty,
    }

    return total_reward, rewards
