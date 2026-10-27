@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
    targets: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Strategy 4 - exponential shaping around a target forward speed.

    ``exp(-gap^2 / (2 sigma^2))`` with ``gap = max(0, v_target - v_forward)``
    and a named temperature ``speed_sigma``.  Only the shortfall is penalised
    so that running faster than the reference speed is never punished (a weak
    linear term rewards the overspeed).  On top of that the ant is rewarded for
    aligning its heading with the direction it has to travel (towards the
    target) and with its own velocity (no sideways sliding).
    """
    lin_vel = root_states[:, 7:10]
    heading_len = torch.norm(heading_vec, dim=-1, keepdim=True).clamp_min(1e-6)
    heading_unit = heading_vec / heading_len
    forward_speed = torch.sum(lin_vel * heading_unit, dim=-1)

    # ---------------- exponential speed shaping -----------------------------
    target_speed: float = 4.0
    speed_sigma: float = 1.5
    speed_gap = torch.clamp(target_speed - forward_speed, min=0.0)
    speed_exp = torch.exp(-torch.square(speed_gap) / (2.0 * speed_sigma * speed_sigma))
    speed_reward = 2.0 * speed_exp
    overspeed = torch.clamp(forward_speed - target_speed, min=0.0)
    overspeed_reward = 0.2 * overspeed

    # ---------------- heading alignment -------------------------------------
    dx = targets[:, 0] - root_states[:, 0]
    dy = targets[:, 1] - root_states[:, 1]
    dist_xy = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
    align_target = heading_unit[:, 0] * (dx / dist_xy) + heading_unit[:, 1] * (dy / dist_xy)
    vel_len = torch.norm(lin_vel, dim=-1, keepdim=True).clamp_min(1e-6)
    vel_unit = lin_vel / vel_len
    align_vel = torch.sum(heading_unit * vel_unit, dim=-1)
    heading_reward = 0.5 * align_target + 0.25 * align_vel

    # ---------------- regularisers ------------------------------------------
    up_z = torch.clamp(up_vec[:, 2], min=0.0, max=1.0)
    tilt_penalty = -0.10 * (1.0 - up_z)
    joint_vel_penalty = -0.0005 * torch.mean(torch.abs(dof_vel), dim=-1)
    action_penalty = -0.005 * torch.mean(torch.square(actions), dim=-1)
    alive_bonus = torch.ones_like(forward_speed) * 0.05

    reward = (speed_reward + overspeed_reward + heading_reward + alive_bonus
              + tilt_penalty + joint_vel_penalty + action_penalty)

    components = {
        "speed_reward": speed_reward,
        "overspeed_reward": overspeed_reward,
        "heading_reward": heading_reward,
        "alive_bonus": alive_bonus,
        "tilt_penalty": tilt_penalty,
        "joint_vel_penalty": joint_vel_penalty,
        "action_penalty": action_penalty,
    }
    return reward, components
