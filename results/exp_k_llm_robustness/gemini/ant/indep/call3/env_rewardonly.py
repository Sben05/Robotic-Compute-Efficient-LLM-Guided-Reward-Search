@torch.jit.script
def compute_reward(obs_buf: torch.Tensor, root_states: torch.Tensor, actions: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the reward for the Ant agent to run forward as fast as possible.

    Reward components:
    1. Forward velocity reward: Encourages the ant to move quickly in its forward direction.
    2. Uprightness reward: Encourages the ant to stay upright and not fall.
    3. Action effort penalty: Penalizes large actions to encourage energy-efficient locomotion.
    4. Joint velocity penalty: Penalizes high joint velocities to promote smoother movements.

    Args:
        obs_buf (torch.Tensor): The observation buffer containing various state information.
                                 obs_buf[:, 1] corresponds to the forward velocity (vel_loc_x).
        root_states (torch.Tensor): The root state of the ant, including position and orientation.
                                    root_states[:, 2] corresponds to the torso height (Z position).
        actions (torch.Tensor): The actions taken by the agent.
        dof_vel (torch.Tensor): The velocities of the degrees of freedom (joints) of the ant.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - Total reward (torch.Tensor): A tensor containing the total reward for each environment.
            - Reward components (Dict[str, torch.Tensor]): A dictionary of individual reward components.
    """
    # Ensure all new tensors are on the same device as the input tensors
    device = obs_buf.device

    # --- Reward Component Weights and Parameters ---
    # Weight for forward velocity reward
    vel_reward_w = torch.tensor(2.0, device=device)

    # Parameters for uprightness reward
    min_height = torch.tensor(0.5, device=device) # Minimum desirable height for the torso
    upright_temp = torch.tensor(5.0, device=device) # Temperature for exponential uprightness reward

    # Weight for action effort penalty
    action_cost_w = torch.tensor(-0.01, device=device)

    # Weight for joint velocity penalty
    dof_vel_cost_w = torch.tensor(-0.001, device=device)

    # --- Compute Individual Reward Components ---

    # 1. Forward Velocity Reward
    # obs_buf[:, 1] is vel_loc_x (forward velocity in the ant's local frame)
    forward_vel = obs_buf[:, 1]
    vel_reward = vel_reward_w * forward_vel

    # 2. Uprightness Reward
    # root_states[:, 2] is the Z coordinate of the torso (height)
    torso_height = root_states[:, 2]
    # Exponential reward: High when above min_height, rapidly decays to a penalty when below.
    upright_reward = torch.exp(upright_temp * (torso_height - min_height))

    # 3. Action Effort Penalty
    # Sum of squared actions to penalize high motor effort
    actions_sum_sq = torch.sum(torch.square(actions), dim=-1)
    action_cost = action_cost_w * actions_sum_sq

    # 4. Joint Velocity Penalty
    # Sum of squared joint velocities to encourage smoother motion
    dof_vel_sum_sq = torch.sum(torch.square(dof_vel), dim=-1)
    dof_vel_cost = dof_vel_cost_w * dof_vel_sum_sq

    # --- Total Reward ---
    total_reward = vel_reward + upright_reward + action_cost + dof_vel_cost

    # --- Reward Components Dictionary (for debugging and analysis) ---
    reward_components = {
        "vel_reward": vel_reward,
        "upright_reward": upright_reward,
        "action_cost": action_cost,
        "dof_vel_cost": dof_vel_cost,
    }

    return total_reward, reward_components
