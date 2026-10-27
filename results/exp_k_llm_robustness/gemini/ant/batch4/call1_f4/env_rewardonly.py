@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor, inv_start_rot: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
    # 1. Strong quadratic reward for forward velocity to prioritize speed
    forward_vel_temp = 1.5
    forward_reward = forward_vel_temp * torch.square(forward_velocity)

    # 2. Survival bonus: a small constant reward for every step the ant is alive
    survival_bonus = 0.1 * torch.ones_like(forward_velocity)

    # 3. Uprightness: Reward for being above a minimum height, with exponential fall-off if below
    min_height = 0.6 # meters
    height_fall_temp = 10.0
    height_reward = torch.exp(-height_fall_temp * torch.relu(min_height - height))

    # 4. Energy efficiency: Penalize power consumption (absolute value of torque * joint velocity)
    # Assumes actions correspond to torques/forces
    power_temp = 0.0005
    power_consumption = torch.sum(torch.abs(actions * dof_vel), dim=-1)
    energy_penalty = -power_temp * power_consumption

    # 5. Penalize deviation from target angular velocity (ideally zero for straight movement)
    ang_vel_deviate_temp = 0.05
    angular_deviation_penalty = -ang_vel_deviate_temp * torch.sum(torch.square(ang_velocity), dim=-1)

    # 6. Small constant action penalty to generally reduce control effort
    action_penalty_temp = 0.0001
    action_penalty = -action_penalty_temp * torch.sum(torch.square(actions), dim=-1)
    
    total_reward = (forward_reward +
                    survival_bonus +
                    height_reward +
                    energy_penalty +
                    angular_deviation_penalty +
                    action_penalty)

    rewards = {
        "forward_reward": forward_reward,
        "survival_bonus": survival_bonus,
        "height_reward": height_reward,
        "energy_penalty": energy_penalty,
        "angular_deviation_penalty": angular_deviation_penalty,
        "action_penalty": action_penalty,
    }

    return total_reward, rewards
