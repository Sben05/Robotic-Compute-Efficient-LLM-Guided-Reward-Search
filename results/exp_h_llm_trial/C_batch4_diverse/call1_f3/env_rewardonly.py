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
    """Strategy 3 - progress (potential difference) based.

    The environment already stores ``potentials = -||to_target|| / dt`` and the
    previous step value, so ``potentials - prev_potentials`` is exactly the
    average speed (m/s) with which the ant closed the distance to the target
    during the last step.  This is used as the main reward; it is complemented
    by survival terms (alive bonus, upright bonus, anti-tumble penalties).
    """
    # ---------------- progress towards the target ---------------------------
    progress_speed = potentials - prev_potentials     # m/s towards the target
    progress_reward = progress_speed                  # linear, no saturation

    # ---------------- survival terms ----------------------------------------
    alive_bonus = torch.ones_like(progress_speed) * 0.10
    up_z = torch.clamp(up_vec[:, 2], min=0.0, max=1.0)
    upright_bonus = 0.5 * up_z
    upside_down_penalty = -0.5 * torch.clamp(-up_vec[:, 2], min=0.0, max=1.0)
    ang_vel = root_states[:, 10:13]
    ang_vel_penalty = -0.01 * torch.sum(torch.square(ang_vel), dim=-1)
    joint_vel_penalty = -0.0005 * torch.mean(torch.abs(dof_vel), dim=-1)
    action_penalty = -0.005 * torch.mean(torch.square(actions), dim=-1)

    # ---------------- small tie breaker: keep pushing forward --------------
    lin_vel = root_states[:, 7:10]
    heading_len = torch.norm(heading_vec, dim=-1, keepdim=True).clamp_min(1e-6)
    heading_unit = heading_vec / heading_len
    forward_speed = torch.sum(lin_vel * heading_unit, dim=-1)
    speed_bonus = 0.1 * forward_speed

    reward = (progress_reward + alive_bonus + upright_bonus + upside_down_penalty
              + ang_vel_penalty + joint_vel_penalty + action_penalty + speed_bonus)

    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "upright_bonus": upright_bonus,
        "upside_down_penalty": upside_down_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "joint_vel_penalty": joint_vel_penalty,
        "action_penalty": action_penalty,
        "speed_bonus": speed_bonus,
    }
    return reward, components
