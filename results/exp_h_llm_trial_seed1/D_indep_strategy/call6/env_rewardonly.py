@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # --------------------------- hyper parameters ---------------------------
    progress_scale: float = 0.02     # metres closed per step that correspond to 1.0 reward
    max_progress_dist: float = 0.5   # clipping of per-step progress (metres)
    speed_scale: float = 2.0         # m/s : temperature used to squash the speed terms
    max_speed: float = 10.0          # m/s : clipping of the speed terms
    progress_weight: float = 2.0
    speed_weight: float = 1.0
    forward_weight: float = 0.5
    alive_weight: float = 0.5
    upright_weight: float = 0.5
    action_weight: float = 0.02

    # --------------------------- geometry helpers ---------------------------
    vel_x = root_states[:, 7]
    vel_y = root_states[:, 8]
    to_target = targets[:, 0:2] - root_states[:, 0:2]
    dist = torch.sqrt(torch.sum(to_target * to_target, dim=-1))
    dist_safe = torch.clamp(dist, 1e-6, 1e6)
    unit_x = to_target[:, 0] / dist_safe
    unit_y = to_target[:, 1] / dist_safe

    # --------------------------- progress (potential difference) ---------------------------
    # potentials = -||target - pos||_xy / dt , prev_potentials = value of the previous step
    # => potentials - prev_potentials = (d_prev - d_now) / dt   (progress towards the target)
    progress = potentials - prev_potentials
    # recover the time step used to build the potential so that the progress can be
    # expressed in metres independently of the internal dt convention
    dt_est = dist / torch.clamp(-potentials, 1e-6)
    progress_dist = torch.clamp(progress * dt_est, -max_progress_dist, max_progress_dist)
    r_progress = progress_dist / progress_scale

    # --------------------------- speed towards the target (m/s) ---------------------------
    speed_to_target = vel_x * unit_x + vel_y * unit_y
    r_speed = torch.clamp(speed_to_target, -max_speed, max_speed) / speed_scale

    # --------------------------- forward speed in the body frame ---------------------------
    quat = root_states[:, 3:7]
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    # torso forward axis (body x) expressed in world coordinates
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_norm = torch.clamp(torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y), 1e-6)
    fwd_x = fwd_x / fwd_norm
    fwd_y = fwd_y / fwd_norm
    forward_speed = vel_x * fwd_x + vel_y * fwd_y
    r_forward = torch.tanh(forward_speed / speed_scale)

    # --------------------------- survival / posture ---------------------------
    r_alive = torch.ones_like(progress)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)          # world-z component of the torso up axis
    r_upright = torch.clamp(up_z, 0.0, 1.0)

    # --------------------------- action regularisation ---------------------------
    r_action = -torch.mean(actions * actions, dim=-1)

    # --------------------------- total ---------------------------
    reward = (progress_weight * r_progress
              + speed_weight * r_speed
              + forward_weight * r_forward
              + alive_weight * r_alive
              + upright_weight * r_upright
              + action_weight * r_action)

    components: Dict[str, torch.Tensor] = {
        "progress": progress_weight * r_progress,
        "speed_to_target": speed_weight * r_speed,
        "forward_speed": forward_weight * r_forward,
        "alive": alive_weight * r_alive,
        "upright": upright_weight * r_upright,
        "action_cost": action_weight * r_action,
    }
    return reward, components
