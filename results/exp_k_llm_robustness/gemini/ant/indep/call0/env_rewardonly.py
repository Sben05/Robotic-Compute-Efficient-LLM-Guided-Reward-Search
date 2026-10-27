@torch.jit.script
def compute_reward(
    self_root_states: torch.Tensor,
    self_actions: torch.Tensor,
    self_up_axis_idx: int
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward function for the Ant to run forward as fast as possible.

    Args:
        self_root_states (torch.Tensor): Tensor containing the root states of the Ant bodies.
                                         Specifically, self_root_states[:, 7:10] contains
                                         the linear velocity (x, y, z) of the torso,
                                         and self_root_states[:, self_up_axis_idx] contains
                                         the height of the torso along the 'up' axis.
        self_actions (torch.Tensor): Tensor containing the actions taken by the agent.
        self_up_axis_idx (int): The index corresponding to the 'up' axis (e.g., 2 for Z-axis).

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing:
            - total_reward (torch.Tensor): The computed total reward for each environment.
            - rewards (Dict[str, torch.Tensor]): A dictionary of individual reward components.
    """
    # Ensure all tensors are on the correct device
    device = self_root_states.device

    # --- Reward Component 1: Forward Velocity ---
    # The Ant's forward direction is typically along the X-axis in many environments.
    # self_root_states[:, 7:10] contains the linear velocities (x, y, z) of the torso.
    forward_vel = self_root_states[:, 7] # Extracting the X-component of linear velocity

    # Encourage positive forward velocity. An exponential reward function helps to
    # differentiate between small positive velocities and strongly incentivizes higher speeds.
    # It also implicitly penalizes negative (backward) velocity as exp(negative) will be < 1.
    forward_reward_temp = 0.5  # Temperature parameter to scale the exponential reward
    reward_forward_velocity = torch.exp(forward_vel * forward_reward_temp)

    # --- Reward Component 2: Action Effort Penalty ---
    # Penalize large actions to encourage energy-efficient and smooth movement.
    # This helps prevent the agent from "wobbling" or expending excessive energy.
    # self_actions is a tensor of shape (num_envs, num_actions).
    action_penalty_weight = 0.005 # Weight for the action penalty
    actions_magnitude_squared = torch.sum(self_actions**2, dim=-1)
    reward_action_penalty = -action_penalty_weight * actions_magnitude_squared

    # --- Reward Component 3: Height Penalty ---
    # Penalize the ant if its torso falls below a certain height threshold.
    # This encourages the agent to maintain an upright and stable posture, which is
    # necessary for locomotion.
    torso_height = self_root_states[:, self_up_axis_idx]
    
    min_height_threshold = 0.35 # Minimum desired height for the torso in meters
    height_penalty_weight = 10.0 # Strong penalty for being below the minimum height

    # Calculate the height violation: positive if height is below threshold, 0 otherwise.
    height_violation = torch.relu(min_height_threshold - torso_height)
    reward_height_penalty = -height_penalty_weight * height_violation

    # --- Total Reward ---
    # Sum of all individual reward components.
    total_reward = reward_forward_velocity + reward_action_penalty + reward_height_penalty

    # --- Individual Reward Components for Logging and Analysis ---
    rewards = {
        "reward_forward_velocity": reward_forward_velocity,
        "reward_action_penalty": reward_action_penalty,
        "reward_height_penalty": reward_height_penalty,
    }

    return total_reward, rewards
