@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward horizontal running speed while excluding vertical motion caused by falling.
    velocity_xy = root_states[:, 7:9]
    planar_speed = torch.sqrt(
        torch.sum(velocity_xy * velocity_xy, dim=-1) + 1.0e-8
    )

    # Compute the world-space torso up vector from the wxyz root quaternion.
    quat_w = root_states[:, 3]
    quat_vec = root_states[:, 4:7]
    up_local = torch.zeros_like(quat_vec)
    up_local[:, 2] = 1.0

    up_world = up_local + 2.0 * torch.cross(
        quat_vec,
        torch.cross(quat_vec, up_local, dim=-1)
        + quat_w.unsqueeze(-1) * up_local,
        dim=-1,
    )
    uprightness = torch.clamp(up_world[:, 2], min=0.0, max=1.0)

    # Prevent diving or sliding from earning the speed reward.
    upright_gate_scale = 0.85
    upright_gate_offset = 0.15
    upright_gate = torch.clamp(
        (uprightness - upright_gate_offset) / upright_gate_scale,
        min=0.0,
        max=1.0,
    )

    running_speed_scale = 1.0
    running_speed = running_speed_scale * planar_speed * upright_gate

    # A small control penalty favors a smooth, repeatable running gait.
    action_penalty_scale = 0.001
    action_penalty = -action_penalty_scale * torch.sum(
        actions * actions, dim=-1
    )

    upright_reward_scale = 0.2
    upright_reward = upright_reward_scale * uprightness

    total_reward = running_speed + upright_reward + action_penalty

    components: Dict[str, torch.Tensor] = {}
    components["running_speed"] = running_speed
    components["upright_reward"] = upright_reward
    components["action_penalty"] = action_penalty

    return total_reward, components
