@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Ensure all constants are on the correct device
    device = root_states.device

    # Reward component scales (these act as "temperature" parameters for linear scaling)
    # 1. Forward velocity reward scale: encourages the ant to move quickly in its forward direction.
    forward_vel_scale = torch.tensor(1.0, device=device)
    # 2. Upright bonus scale: encourages the ant to stay upright.
    upright_bonus_scale = torch.tensor(0.5, device=device)
    # 3. Action penalty scale: penalizes large actions to encourage energy-efficient locomotion.
    action_penalty_scale = torch.tensor(0.001, device=device)

    # Extract relevant states
    linear_velocity = root_states[:, 7:10]  # Linear velocity (vx, vy, vz) of the torso

    # 1. Forward velocity reward
    # The heading_vec represents the ant's forward direction in the world frame.
    # The dot product of linear_velocity and heading_vec gives the speed in the forward direction.
    forward_velocity = torch.sum(linear_velocity * heading_vec, dim=-1)
    forward_vel_reward = forward_velocity * forward_vel_scale

    # 2. Upright bonus
    # up_vec[:, 2] represents the Z-component of the ant's up-vector in world coordinates.
    # A value of 1 means perfectly upright (ant's body Z-axis aligned with world Z-axis).
    # A value close to 0 means the ant is horizontal. A negative value means it's upside down.
    # This reward component encourages the ant to maintain an upright posture.
    upright_bonus = up_vec[:, 2] * upright_bonus_scale

    # 3. Action penalty
    # Penalize the sum of squared actions. This discourages overly large or jerky movements.
    action_penalty = torch.sum(torch.square(actions), dim=-1) * action_penalty_scale

    # Total reward is the sum of all components
    total_reward = forward_vel_reward + upright_bonus - action_penalty

    # Store individual reward components for analysis
    rewards = {
        "forward_vel_reward": forward_vel_reward,
        "upright_bonus": upright_bonus,
        "action_penalty": action_penalty,
    }

    return total_reward, rewards
