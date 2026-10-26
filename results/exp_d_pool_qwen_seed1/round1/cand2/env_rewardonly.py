@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    vec_sensor_tensor: torch.Tensor,
    dt: float,
    contact_force_scale: float,
    dof_vel_scale: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    # --- Forward velocity reward (primary objective) ---
    # Project world velocity onto the ant's heading direction
    forward_vel = (velocity * heading_vec).sum(dim=-1)

    forward_vel_temp = 0.1
    forward_vel_reward = torch.exp(forward_vel * forward_vel_temp)

    # --- Position / target tracking reward (shaping) ---
    to_target = targets - torso_position
    to_target_z = torch.zeros_like(to_target[:, 0:1])
    to_target = torch.cat([to_target[:, 0:1], to_target[:, 1:2], to_target_z], dim=-1)  # zero out z
    target_dist = torch.norm(to_target, p=2, dim=-1)
    pos_temp = 0.01
    pos_reward = -target_dist * pos_temp

    # --- Upright reward (stability) ---
    # Reward for keeping the ant's local up aligned with world up
    # up_vec shape is [num_envs, 3]; world up is [0, 0, 1]
    world_up = torch.tensor([0.0, 0.0, 1.0], device=root_states.device)
    upright_proj = (up_vec * world_up).sum(dim=-1)
    upright_reward = upright_proj * 0.1

    # --- Angular velocity penalty (smoothness / stability) ---
    ang_vel_penalty = -(ang_velocity ** 2).sum(dim=-1) * 0.02

    # --- Joint velocity penalty (avoid jerky movements) ---
    dof_vel_penalty = -(dof_vel ** 2).sum(dim=-1) * 0.001

    # --- Action magnitude penalty (energy efficiency) ---
    action_penalty = -(actions ** 2).sum(dim=-1) * 0.001

    # --- Contact force penalty (avoid excessive forces) ---
    force_magnitude = torch.norm(vec_sensor_tensor.view(-1, 24), p=2, dim=-1) * contact_force_scale
    force_penalty = -force_magnitude * 0.001

    # Total reward
    reward = (
        forward_vel_reward
        + pos_reward
        + upright_reward
        + ang_vel_penalty
        + dof_vel_penalty
        + action_penalty
        + force_penalty
    )

    reward_dict = {
        "forward_vel": forward_vel_reward,
        "position": pos_reward,
        "upright": upright_reward,
        "ang_vel_penalty": ang_vel_penalty,
        "dof_vel_penalty": dof_vel_penalty,
        "action_penalty": action_penalty,
        "force_penalty": force_penalty,
    }

    return reward, reward_dict
