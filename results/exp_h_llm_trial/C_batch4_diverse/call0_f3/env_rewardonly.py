@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor,
                     potentials: torch.Tensor, prev_potentials: torch.Tensor,
                     dof_vel: torch.Tensor, actions: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Strategy 3 - progress (potential difference) based.

    The env stores potentials = -||target - torso|| / dt, so the potential
    difference times dt is exactly the distance closed towards the target
    during the step.  That progress is the dominant reward; survival terms
    (alive bonus + uprightness) keep the ant alive, and a small velocity
    tie-breaker avoids stalling.
    """
    quat = root_states[:, 3:7]
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    # ---------------- potential difference (progress) -------------------------
    progress = (potentials - prev_potentials) * dt          # metres gained, + = closer
    progress_clipped = torch.clamp(progress, min=-0.25, max=0.25)
    progress_reward = 8.0 * progress_clipped

    # ---------------- survival terms -----------------------------------------
    alive = torch.ones_like(progress)
    up = _body_axis_3(quat, 2)
    up_z = torch.clamp(up[:, 2], min=0.0, max=1.0)
    upright_survival = 0.5 * up_z

    # ---------------- velocity tie-breaker toward the target ------------------
    to_target_xy = targets[:, 0:2] - root_states[:, 0:2]
    to_target_hat = to_target_xy / torch.norm(to_target_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    vel_xy = torch.stack((lin_vel[:, 0], lin_vel[:, 1]), dim=-1)
    v_to_target = torch.sum(vel_xy * to_target_hat, dim=-1)
    speed_bonus = 0.05 * torch.clamp(v_to_target, min=0.0)

    # ---------------- regularisers -------------------------------------------
    ang_vel_penalty = -0.001 * torch.sum(torch.square(ang_vel), dim=-1)
    dof_vel_penalty = -0.00005 * torch.sum(torch.square(dof_vel), dim=-1)
    action_penalty = -0.002 * torch.sum(torch.square(actions), dim=-1)

    reward = (progress_reward + 0.5 * alive + upright_survival + speed_bonus +
              ang_vel_penalty + dof_vel_penalty + action_penalty)

    components = {
        "progress_reward": progress_reward,
        "raw_progress": progress,
        "alive_bonus": 0.5 * alive,
        "upright_survival": upright_survival,
        "speed_bonus": speed_bonus,
        "ang_vel_penalty": ang_vel_penalty,
        "dof_vel_penalty": dof_vel_penalty,
        "action_penalty": action_penalty,
    }
    return reward, components
