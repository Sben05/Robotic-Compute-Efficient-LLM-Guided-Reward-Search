@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   targets: torch.Tensor,
                   potentials: torch.Tensor,
                   prev_potentials: torch.Tensor,
                   dof_vel: torch.Tensor,
                   actions: torch.Tensor,
                   dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    device = root_states.device
    num_envs = root_states.shape[0]

    # ----------------------------- weights -----------------------------
    w_progress: float = 4.0      # per meter of distance closed toward the target
    w_speed: float = 0.5        # per m/s of velocity pointing at the target
    w_forward: float = 0.15     # per m/s of velocity along the torso forward axis
    w_alive: float = 0.1        # survival bonus per step
    w_action: float = 0.01      # action magnitude penalty
    w_dofvel: float = 1.0e-4    # joint velocity penalty

    # ------------------------------------------------------------------
    # 1) Progress term (potential difference).
    #    potentials = -||target - pos||_xy / dt
    #    => (potentials - prev_potentials) * dt = distance closed [m]
    # ------------------------------------------------------------------
    progress_dist = (potentials - prev_potentials) * dt
    progress_dist = torch.clamp(progress_dist, min=-1.0, max=1.0)
    r_progress = w_progress * progress_dist

    # ------------------------------------------------------------------
    # 2) Running speed in the direction of the target (horizontal).
    # ------------------------------------------------------------------
    pos = root_states[:, 0:3]
    lin_vel = root_states[:, 7:10]
    to_target = targets - pos
    to_target_xy = torch.stack((to_target[:, 0], to_target[:, 1]), dim=-1)
    dist_xy = torch.norm(to_target_xy, dim=-1) + 1.0e-6
    dir_xy = to_target_xy / dist_xy.unsqueeze(-1)
    speed_toward = torch.sum(lin_vel[:, 0:2] * dir_xy, dim=-1)
    speed_toward = torch.clamp(speed_toward, min=-5.0, max=10.0)
    r_speed = w_speed * speed_toward

    # ------------------------------------------------------------------
    # 3) Speed along the ant's own forward axis (+x of the torso),
    #    obtained by rotating the body forward axis with the root quaternion.
    # ------------------------------------------------------------------
    quat = root_states[:, 3:7]
    forward_body = torch.zeros(num_envs, 3, device=device)
    forward_body[:, 0] = 1.0
    q_w = quat[:, 0:1]
    q_vec = quat[:, 1:4]
    t = 2.0 * torch.cross(q_vec, forward_body, dim=-1)
    forward_world = forward_body + q_w * t + torch.cross(q_vec, t, dim=-1)
    forward_speed = torch.sum(lin_vel[:, 0:2] * forward_world[:, 0:2], dim=-1)
    forward_speed = torch.clamp(forward_speed, min=-2.0, max=10.0)
    r_forward = w_forward * forward_speed

    # ------------------------------------------------------------------
    # 4) Survival bonus.
    # ------------------------------------------------------------------
    r_alive = w_alive * torch.ones(num_envs, device=device)

    # ------------------------------------------------------------------
    # 5) Smoothness / energy regularization.
    # ------------------------------------------------------------------
    action_sq = torch.sum(actions * actions, dim=-1)
    r_action = -w_action * action_sq
    dof_vel_sq = torch.sum(dof_vel * dof_vel, dim=-1)
    r_dofvel = -w_dofvel * dof_vel_sq

    reward = r_progress + r_speed + r_forward + r_alive + r_action + r_dofvel

    components: Dict[str, torch.Tensor] = {
        "progress": r_progress,
        "speed_to_target": r_speed,
        "forward_speed": r_forward,
        "alive": r_alive,
        "action_penalty": r_action,
        "dof_vel_penalty": r_dofvel,
    }

    return reward, components
