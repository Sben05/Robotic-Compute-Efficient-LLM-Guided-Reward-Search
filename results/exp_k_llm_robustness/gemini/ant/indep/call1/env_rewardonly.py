@torch.jit.script
def compute_reward(root_states: torch.Tensor, heading_vec: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward function for making the Ant run forward as fast as possible.

    Args:
        root_states (torch.Tensor): Tensor containing the root states of the Ant.
                                    Specifically, root_states[:, 7:10] contains the linear velocity.
        heading_vec (torch.Tensor): Tensor representing the Ant's forward heading vector in the global frame.
        up_axis_idx (int): The index corresponding to the 'up' axis (e.g., 2 for Z-axis).

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing:
            - total_reward (torch.Tensor): The total reward for each environment.
            - reward_components (Dict[str, torch.Tensor]): A dictionary of individual reward components.
    """
    # Extract linear velocity from root_states.
    # root_states[:, 7:10] contains (vx, vy, vz) in the global frame for each ant.
    velocity = root_states[:, 7:10]

    # Create a horizontally projected heading vector to define "forward" movement
    # primarily in the horizontal plane. This avoids rewarding vertical movement
    # if the ant is pitched up or down.
    heading_vec_horz = heading_vec.clone()
    # Set the vertical component to zero. For the Ant environment, up_axis_idx is typically 2 (Z-axis).
    heading_vec_horz[:, up_axis_idx] = 0.0
    
    # Normalize the horizontal heading vector.
    # Add a small epsilon to prevent division by zero in case of a (0,0,0) vector,
    # though the heading vector should generally not be zero or have a zero horizontal projection.
    heading_vec_horz_norm = torch.nn.functional.normalize(heading_vec_horz, p=2, dim=-1, eps=1e-6)

    # Calculate the forward velocity component.
    # This is the projection of the actual linear velocity onto the normalized
    # horizontal forward direction of the ant.
    # A positive value means the ant is moving forward, a negative value means it's moving backward.
    forward_velocity_reward = torch.sum(velocity * heading_vec_horz_norm, dim=-1)

    # For the task "run forward as fast as possible", directly maximizing the forward velocity
    # is the most straightforward and effective reward. A linear reward ensures that
    # faster forward movement gets proportionally higher reward, and backward movement
    # receives a proportional penalty (negative reward).
    # No exponential transformation is applied here, as applying `torch.exp` would
    # make negative velocities (backward movement) yield a positive reward, which
    # is contrary to the goal of encouraging *forward* movement.
    
    # The total reward for this task is solely based on the forward velocity.
    total_reward = forward_velocity_reward

    # Store individual reward components for debugging and analysis.
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
    }

    return total_reward, reward_components
