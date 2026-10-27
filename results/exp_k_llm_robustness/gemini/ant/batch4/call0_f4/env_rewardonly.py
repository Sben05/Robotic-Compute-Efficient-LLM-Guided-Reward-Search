@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, heading_vec: torch.Tensor, up_vec: torch.Tensor, dof_vel: torch.Tensor, up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract necessary state variables
    linear_velocity = root_states[:, 7:10]
    angular_velocity = root_states[:, 10:13]
    torso_position_z = root_states[:, up_axis_idx]
    device = root_states.device

    # Reward component scaling factors (replacing 'temperature' for linear/quadratic components)
    forward_vel_scale = 1.0 
    action_cost_scale = 0.005
    dof_vel_cost_scale = 0.001
    upright_alignment_scale = 0.5 
    angular_vel_cost_scale = 0.01
    height_deviation_cost_scale = 1.0
    survival_bonus = 0.1 

    # Target parameters
    target_height = 0.6 

    # 1. Forward velocity reward: Reward for positive velocity in the heading direction.
    forward_velocity = torch.sum(linear_velocity * heading_vec, dim=1)
    forward_reward = forward_vel_scale * forward_velocity

    # 2. Action penalty: Penalize large control efforts using a quadratic penalty.
    action_penalty = torch.sum(torch.square(actions), dim=-1)
    action_reward = -action_cost_scale * action_penalty

    # 3. DOF velocity penalty: Penalize large joint velocities (jerky movements) using a quadratic penalty.
    dof_vel_penalty = torch.sum(torch.square(dof_vel), dim=-1)
    dof_vel_reward = -dof_vel_cost_scale * dof_vel_penalty

    # 4. Uprightness reward: Reward for torso's up vector aligning with global up.
    global_up_vec = torch.zeros_like(up_vec, device=device)
    global_up_vec[:, up_axis_idx] = 1.0
    upright_proj = torch.sum(up_vec * global_up_vec, dim=1)
    # Reward is higher when upright_proj is close to 1.0. A linear reward component.
    upright_reward = upright_alignment_scale * (upright_proj - 0.7) # Encourage proj > 0.7
    upright_reward = torch.clamp(upright_reward, min=-0.5, max=0.5) # Clamp to prevent extreme values

    # 5. Angular velocity penalty: Penalize excessive rotational velocity.
    angular_velocity_magnitude = torch.norm(angular_velocity, dim=1)
    angular_vel_reward = -angular_vel_cost_scale * angular_velocity_magnitude

    # 6. Height maintenance reward/penalty: Encourage staying near target_height with a quadratic penalty.
    height_deviation = torch.square(torso_position_z - target_height)
    height_reward = -height_deviation_cost_scale * height_deviation

    # 7. Survival bonus: Constant positive reward per step.
    survival_reward = torch.ones_like(forward_reward, device=device) * survival_bonus

    total_reward = forward_reward + action_reward + dof_vel_reward + upright_reward + angular_vel_reward + height_reward + survival_reward

    rewards = {
        "forward_reward": forward_reward,
        "action_reward": action_reward,
        "dof_vel_reward": dof_vel_reward,
        "upright_reward": upright_reward,
        "angular_velocity_reward": angular_vel_reward,
        "height_reward": height_reward,
        "survival_reward": survival_reward,
    }
    return total_reward, rewards
