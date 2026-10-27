@torch.jit.script
def compute_reward(
    self_root_states: torch.Tensor,
    self_heading_vec: torch.Tensor,
    self_up_vec: torch.Tensor,
    self_up_axis_idx: int,
    self_actions: torch.Tensor,
    self_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward function for the Ant agent to run forward as fast as possible.

    Args:
        self_root_states: (num_envs, 13) tensor containing [pos, rot, lin_vel, ang_vel]
                          of the ant's root.
        self_heading_vec: (num_envs, 3) tensor representing the ant's forward direction in global frame.
        self_up_vec: (num_envs, 3) tensor representing the ant's up direction in global frame.
        self_up_axis_idx: Integer indicating the index of the up axis (e.g., 2 for Z-up).
        self_actions: (num_envs, num_actions) tensor of actions taken by the agent.
        self_dof_vel: (num_envs, num_dofs) tensor of joint velocities.

    Returns:
        A tuple containing:
            - total_reward: (num_envs,) tensor, the sum of all reward components.
            - reward_components: Dictionary of individual reward components for analysis.
    """
    
    # Ensure all new tensors are on the same device as input tensors
    device = self_root_states.device
    
    # --- 1. Forward Velocity Reward ---
    # Extract global linear velocity of the torso (indices 7-9)
    velocity = self_root_states[:, 7:10]
    
    # Project the global velocity onto the ant's heading vector to get forward speed
    forward_speed = torch.sum(velocity * self_heading_vec, dim=-1)
    
    # Use an exponential reward for forward speed to encourage higher velocities
    vel_reward_temp = 1.0 # Temperature parameter to scale the exponential growth
    vel_reward = torch.exp(forward_speed * vel_reward_temp)
    
    # --- 2. Upright Reward ---
    # Define the global 'up' vector based on the up_axis_idx
    global_up_vec = torch.zeros_like(self_up_vec, device=device)
    global_up_vec[:, self_up_axis_idx] = 1.0
    
    # Calculate the alignment of the ant's up vector with the global up vector (dot product)
    up_alignment = torch.sum(self_up_vec * global_up_vec, dim=-1)
    
    # Reward for staying upright, penalizing deviation from perfect alignment
    # The reward is 1.0 when up_alignment is 0.7, and increases exponentially as it approaches 1.0
    upright_temp = 2.0 # Temperature parameter to control sensitivity to uprightness
    upright_reward = torch.exp((up_alignment - 0.7) * upright_temp)
    
    # --- 3. Height Reward ---
    # Extract the torso height
    torso_height = self_root_states[:, self_up_axis_idx]
    
    # Define a target height for the ant's torso (e.g., typical standing height for Ant)
    target_height = 0.7 # meters
    
    # Penalize deviations from the target height using an exponential decay
    height_diff = torch.abs(torso_height - target_height)
    height_temp = 5.0 # Temperature parameter to control height penalty sensitivity
    height_reward = torch.exp(-height_diff * height_temp)
    
    # --- 4. Action Penalty ---
    # Penalize large actions to encourage smooth and energy-efficient movements
    action_magnitude_sq = torch.sum(torch.square(self_actions), dim=-1)
    action_cost_scale = 0.005 # Scaling factor for the action penalty
    action_penalty = -action_magnitude_sq * action_cost_scale
    
    # --- 5. Joint Velocity Penalty ---
    # Penalize high joint velocities to prevent jerky movements and potential instability
    joint_vel_magnitude_sq = torch.sum(torch.square(self_dof_vel), dim=-1)
    joint_vel_cost_scale = 0.0001 # Scaling factor for the joint velocity penalty
    joint_vel_penalty = -joint_vel_magnitude_sq * joint_vel_cost_scale
    
    # --- Total Reward ---
    total_reward = vel_reward + upright_reward + height_reward + action_penalty + joint_vel_penalty
    
    # Collect individual reward components for debugging and analysis
    reward_components = {
        "vel_reward": vel_reward,
        "upright_reward": upright_reward,
        "height_reward": height_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty,
    }
    
    return total_reward, reward_components
