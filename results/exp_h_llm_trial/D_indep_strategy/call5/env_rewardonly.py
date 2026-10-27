@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   actions: Optional[torch.Tensor] = None,
                   dt: float = 0.02) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # ---------------- hyper parameters ----------------
    speed_coef: float = 1.0        # weight on the posture-gated forward speed
    speed_max: float = 20.0        # saturation of the forward speed (m/s)
    speed_min: float = -5.0        # how strongly backward motion is penalised
    gate_threshold: float = 0.3    # up-projection at which the ant counts as upright
    gate_temp: float = 0.1         # temperature (slope) of the smooth uprightness gate
    upright_coef: float = 0.05     # small bonus for remaining upright
    ctrl_coef: float = 0.05        # control effort penalty (per second)
    lateral_coef: float = 0.05     # sideways speed penalty
    angvel_coef: float = 0.002     # body rotation rate penalty

    # ---------------- parse the root state ----------------
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]
    quat = root_states[:, 3:7]

    forward_vel = lin_vel[:, 0]
    lateral_vel = lin_vel[:, 1]

    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]
    q_norm = torch.sqrt(qx * qx + qy * qy + qz * qz + qw * qw).clamp(min=1e-6)
    qx = qx / q_norm
    qy = qy / q_norm
    qz = qz / q_norm
    qw = qw / q_norm

    # world-frame z component of the torso up axis (body +z is up)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # ---------------- smooth posture gate ----------------
    # ~0 when the ant is upside down / tumbling, ~1 when standing
    upright_gate = torch.sigmoid((up_z - gate_threshold) / gate_temp)

    # ---------------- reward components ----------------
    fwd = torch.clamp(forward_vel, min=speed_min, max=speed_max)
    gated_speed = fwd * upright_gate

    ctrl = torch.zeros_like(forward_vel)
    if actions is not None:
        ctrl = torch.sum(actions * actions, dim=1)

    lateral_sq = lateral_vel * lateral_vel
    ang_vel_sq = torch.sum(ang_vel * ang_vel, dim=1)

    run_term = speed_coef * gated_speed
    upright_term = upright_coef * upright_gate
    ctrl_term = -ctrl_coef * ctrl * dt
    lateral_term = -lateral_coef * lateral_sq
    angvel_term = -angvel_coef * ang_vel_sq

    total = run_term + upright_term + ctrl_term + lateral_term + angvel_term

    components = {}
    components["gated_forward_speed"] = run_term
    components["forward_speed"] = fwd
    components["upright_gate"] = upright_gate
    components["upright_bonus"] = upright_term
    components["control_cost"] = ctrl_term
    components["lateral_penalty"] = lateral_term
    components["angvel_penalty"] = angvel_term

    return total, components
