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
    """Strategy 2 - posture gated.

    The forward velocity is multiplied by a smooth uprightness gate, so a
    tumbling / upside-down ant earns (almost) nothing for moving fast.  The
    gate is built from the torso "up" direction (world frame) and is
    cross-checked with the root quaternion (z component of R(q) e_z =
    1 - 2 (qx^2 + qy^2)), which is exactly cos(tilt).
    """
    lin_vel = root_states[:, 7:10]
    heading_len = torch.norm(heading_vec, dim=-1, keepdim=True).clamp_min(1e-6)
    heading_unit = heading_vec / heading_len
    forward_speed = torch.sum(lin_vel * heading_unit, dim=-1)

    # ---------------- uprightness gate --------------------------------------
    up_z_env = up_vec[:, 2]                            # world-frame up direction
    q = root_states[:, 3:7]                            # torso quaternion (w,x,y,z)
    up_z_quat = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
    up_z = torch.minimum(up_z_env, up_z_quat)
    upright = torch.clamp(up_z, min=0.0, max=1.0)       # 1 upright, 0 on side
    gate = torch.square(upright)                        # smooth, 0 when tumbling

    # ---------------- gated forward speed -----------------------------------
    speed_temp: float = 6.0
    gated_speed = gate * speed_temp * torch.tanh(forward_speed / speed_temp)

    # ---------------- regularisers ------------------------------------------
    ang_vel = root_states[:, 10:13]
    ang_vel_penalty = -0.02 * torch.sum(torch.square(ang_vel), dim=-1)
    joint_vel_penalty = -0.001 * torch.mean(torch.abs(dof_vel), dim=-1)
    action_penalty = -0.005 * torch.mean(torch.square(actions), dim=-1)
    posture_reward = 0.5 * upright
    tilt_penalty = -0.5 * (1.0 - upright)
    alive_bonus = torch.ones_like(forward_speed) * 0.05

    reward = (gated_speed + posture_reward + tilt_penalty + alive_bonus
              + ang_vel_penalty + joint_vel_penalty + action_penalty)

    components = {
        "gated_speed": gated_speed,
        "posture_reward": posture_reward,
        "tilt_penalty": tilt_penalty,
        "alive_bonus": alive_bonus,
        "ang_vel_penalty": ang_vel_penalty,
        "joint_vel_penalty": joint_vel_penalty,
        "action_penalty": action_penalty,
    }
    return reward, components
