@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant kinematic states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # Primary: forward velocity (dot product of linear velocity with heading direction)
    forward_velocity = torch.sum(velocity * heading_vec, dim=-1)

    # Secondary: torso height (along up direction) to keep ant upright
    torso_height = torch.sum(torso_position * up_vec, dim=-1)

    # Penalty: angular velocity magnitude (stability)
    ang_vel_mag = torch.norm(ang_velocity, p=2, dim=-1)

    # Penalty: action magnitude (efficiency)
    action_mag = torch.sum(actions * actions, dim=-1)

    # Penalty: joint velocity magnitude (jerk)
    dof_vel_mag = torch.sum(dof_vel * dof_vel, dim=-1)

    # Temperature / scaling parameters
    temp_height = 0.1
    temp_angvel = 0.05
    temp_action = 0.001
    temp_dofvel = 0.001

    # Reward components
    forward_reward = forward_velocity
    height_reward = temp_height * torso_height
    angvel_penalty = -temp_angvel * (ang_vel_mag * ang_vel_mag)
    action_penalty = -temp_action * action_mag
    dofvel_penalty = -temp_dofvel * dof_vel_mag

    total_reward = forward_reward + height_reward + angvel_penalty + action_penalty + dofvel_penalty

    reward_components: Dict[str, torch.Tensor] = {}
    reward_components["forward_velocity"] = forward_reward
    reward_components["height_bonus"] = height_reward
    reward_components["angvel_penalty"] = angvel_penalty
    reward_components["action_penalty"] = action_penalty
    reward_components["dofvel_penalty"] = dofvel_penalty

    return total_reward, reward_components
