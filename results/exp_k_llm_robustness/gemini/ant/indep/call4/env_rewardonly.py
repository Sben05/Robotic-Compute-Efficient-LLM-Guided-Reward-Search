@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    up_axis_idx: int,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the Ant agent to run forward as fast as possible.

    Args:
        root_states (torch.Tensor): Tensor containing the root state of the ant (position, rotation, velocity, ang_velocity).
                                    Shape: (num_envs, 13)
                                    Relevant slices:
                                        - velocity: root_states[:, 7:10] (linear velocity in global frame)
        heading_vec (torch.Tensor): Tensor representing the ant's forward direction in the global frame.
                                    Shape: (num_envs, 3)
        up_vec (torch.Tensor): Tensor representing the ant's 'up' direction in the global frame.
                               Shape: (num_envs, 3)
        up_axis_idx (int): The index of the global 'up' axis (e.g., 2 for Z-axis).
        actions (torch.Tensor): Tensor containing the actions taken by the agent.
                                Shape: (num_envs, num_actions)

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - total_reward (torch.Tensor): The total reward for each environment.
            - rewards (Dict[str, torch.Tensor]): A dictionary of individual reward components.
    """

    # --- Temperature parameters for reward shaping ---
    # These parameters control the sensitivity and magnitude of each reward component.
    # They should be tuned based on the learning performance.
    forward_vel_temp: float = 0.5  # Higher temp -> flatter reward curve for forward velocity
    upright_temp: float = 0.5      # Higher temp -> flatter reward curve for uprightness
    action_penalty_temp: float = 0.1 # Higher temp -> less severe action penalty

    # --- Extract relevant states ---
    velocity = root_states[:, 7:10] # Linear velocity (x, y, z) in the global frame

    # --- 1. Forward Velocity Reward ---
    # Goal: Maximize velocity in the desired forward direction (heading_vec).
    # The dot product of velocity and heading_vec gives the speed along the heading.
    # We use an exponential function to strongly reward higher forward speeds.
    forward_vel = (velocity * heading_vec).sum(dim=-1)
    forward_reward = torch.exp(forward_vel / forward_vel_temp)

    # --- 2. Upright Reward ---
    # Goal: Encourage the ant to stay upright.
    # up_vec[:, up_axis_idx] is the component of the ant's local 'up' vector
    # that aligns with the global 'up' axis. A value close to 1 means the ant is perfectly upright.
    # An exponential reward heavily penalizes deviation from upright.
    upright_score = up_vec[:, up_axis_idx]
    upright_reward = torch.exp(upright_score / upright_temp)

    # --- 3. Action Penalty ---
    # Goal: Encourage smooth and energy-efficient movements by penalizing large actions.
    # Sum of squared actions serves as a measure of action magnitude.
    # An exponential decay ensures a reward of 1 for zero actions and rapid decay for larger actions.
    action_penalty = torch.sum(actions**2, dim=-1)
    action_cost = torch.exp(-action_penalty / action_penalty_temp)

    # --- Combine Reward Components ---
    # The total reward is a sum of the individual components.
    total_reward = forward_reward + upright_reward + action_cost

    # --- Prepare rewards dictionary for logging/analysis ---
    rewards = {
        "forward_reward": forward_reward,
        "upright_reward": upright_reward,
        "action_cost": action_cost
    }

    return total_reward, rewards
