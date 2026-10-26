@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_limits_lower: torch.Tensor,
    dof_limits_upper: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract world velocity
    velocity = root_states[:, 7:10]

    # Project velocity and heading to ground plane (x-y)
    vel_ground = velocity.clone()
    vel_ground[:, 2] = 0.0
    heading_ground = heading_vec.clone()
    heading_ground[:, 2] = 0.0
    heading_norm = torch.norm(heading_ground, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    heading_ground = heading_ground / heading_norm

    # Forward speed: dot product of ground velocity with heading direction
    forward_speed = (vel_ground * heading_ground).sum(dim=-1)

    # Upright reward: z-component of up vector (1.0 when perfectly upright)
    upright = up_vec[:, 2]

    # Action cost: penalize large control inputs for energy efficiency
    action_cost = (actions ** 2).sum(dim=-1)

    # Joint limits penalty: discourage joints from exceeding their range
    below = torch.clamp(dof_limits_lower - dof_pos, min=0.0)
    above = torch.clamp(dof_pos - dof_limits_upper, min=0.0)
    joint_penalty = (below ** 2).sum(dim=-1) + (above ** 2).sum(dim=-1)

    # Temperate (scoring) parameters for each component
    speed_temp = 1.0
    upright_temp = 5.0
    action_temp = 0.01
    joint_temp = 1.0

    reward_speed = speed_temp * forward_speed
    reward_upright = upright_temp * upright
    reward_action = -action_temp * action_cost
    reward_joint = -joint_temp * joint_penalty

    total_reward = reward_speed + reward_upright + reward_action + reward_joint

    return total_reward, {
        "forward_speed": reward_speed,
        "upright": reward_upright,
        "action_cost": reward_action,
        "joint_penalty": reward_joint,
    }
