@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor,
                   dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    pi: float = 3.141592653589793

    # ----------------------------- hyper-parameters -----------------------------
    speed_temperature: float = 6.0      # m/s, soft-saturation scale of the speed terms
    align_temperature: float = 1.0      # heading tolerance (x pi rad); 1.0 = full range
    lateral_temperature: float = 1.0    # m/s, scale of the lateral-drift penalty
    action_cost_scale: float = 0.002   # small: fast running needs large torques
    joint_vel_cost_scale: float = 0.0005  # small: this cost opposes high-speed gaits
    fall_weight: float = 2.0
    fall_height: float = 0.30           # m, torso height below which the ant has fallen

    speed_weight: float = 2.5           # tanh(|v|) progress bonus
    linear_speed_weight: float = 0.8    # linear progress bonus (pushes for ever faster)
    facing_weight: float = 0.5          # bonus for moving *along* the body forward axis
    align_weight: float = 2.0           # monotone heading-alignment bonus
    upright_weight: float = 0.5
    alive_weight: float = 0.1
    lateral_weight: float = 0.1

    # ------------------------------ state extraction -----------------------------
    quat = root_states[:, 3:7]         # Isaac Gym quaternion (x, y, z, w)
    vel = root_states[:, 7:10]          # world-frame linear velocity of the torso
    pos_z = root_states[:, 2]            # world-frame torso height

    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    # body axes expressed in the world frame (body +x = forward, body +z = up)
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # horizontal (xy) projection of the body forward axis, unit length
    horiz = torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + 1.0e-8)
    fwd_hx = fwd_x / horiz
    fwd_hy = fwd_y / horiz

    vx = vel[:, 0]
    vy = vel[:, 1]
    speed_h = torch.sqrt(vx * vx + vy * vy + 1.0e-8)      # horizontal speed magnitude
    facing_speed = vx * fwd_hx + vy * fwd_hy              # speed along the body forward axis

    yaw = torch.atan2(fwd_hy, fwd_hx)                     # heading of the torso in [-pi, pi]

    # effective forward progress: speed weighted by how well the ant faces world +x
    progress = speed_h * fwd_hx

    # ------------------------------ reward components ----------------------------
    # 1) main objective: run forward (+x) as fast as possible
    forward_speed = speed_weight * torch.tanh(progress / speed_temperature) \
        + linear_speed_weight * progress
    # 2) run *along* the body axis instead of sliding sideways
    facing_speed_term = facing_weight * torch.tanh(facing_speed / speed_temperature)
    # 3) monotone heading alignment (constant-sign gradient, no 180 deg dead zone)
    alignment = 1.0 - torch.abs(yaw) / (align_temperature * pi)
    heading_align = align_weight * alignment
    # 4) stay on its legs (linear projection -> no saturation / dead gradient)
    upright = upright_weight * up_z
    # 5) cheap survival bonus (small, so it does not mask the speed signal)
    alive = alive_weight * torch.ones(root_states.shape[0], device=root_states.device,
                                      dtype=root_states.dtype)
    # 6) penalise lateral drifting
    lateral_drift = -lateral_weight * torch.tanh(vy / lateral_temperature)
    # 7) mild control / smoothness regularisers
    action_cost = -action_cost_scale * torch.sum(actions * actions, dim=-1)
    joint_vel_cost = -joint_vel_cost_scale * torch.mean(dof_vel * dof_vel, dim=-1)
    # 8) punish collapsing
    fall_penalty = -fall_weight * torch.clamp(fall_height - pos_z, min=0.0)

    reward = (forward_speed + facing_speed_term + heading_align + upright + alive
              + lateral_drift + action_cost + joint_vel_cost + fall_penalty)

    components: Dict[str, torch.Tensor] = {
        "forward_speed": forward_speed,
        "facing_speed": facing_speed_term,
        "heading_align": heading_align,
        "upright": upright,
        "alive": alive,
        "lateral_drift": lateral_drift,
        "action_cost": action_cost,
        "joint_vel_cost": joint_vel_cost,
        "fall_penalty": fall_penalty,
    }

    return reward, components
