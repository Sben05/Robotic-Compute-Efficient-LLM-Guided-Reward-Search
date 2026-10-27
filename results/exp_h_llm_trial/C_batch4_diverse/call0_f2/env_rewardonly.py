@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Strategy 2 - posture gated.

    The forward-velocity reward is multiplied by a strict uprightness gate
    (cos of the torso tilt, squared).  When the ant tumbles the gate closes,
    so neither the speed term nor the survival bonus is paid out; an explicit
    tumble penalty additionally discourages flipping over.
    """
    quat = root_states[:, 3:7]
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    fwd = _body_axis_2(quat, 0)
    up = _body_axis_2(quat, 2)

    fwd_xy = torch.stack((fwd[:, 0], fwd[:, 1]), dim=-1)
    vel_xy = torch.stack((lin_vel[:, 0], lin_vel[:, 1]), dim=-1)
    fwd_hat = fwd_xy / torch.norm(fwd_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    lat_hat = torch.stack((-fwd_hat[:, 1], fwd_hat[:, 0]), dim=-1)

    v_fwd = torch.sum(vel_xy * fwd_hat, dim=-1)
    v_lat = torch.sum(vel_xy * lat_hat, dim=-1)
    up_z = torch.clamp(up[:, 2], min=0.0, max=1.0)

    # ---------------- posture gate -------------------------------------------
    gate = torch.pow(up_z, 2.0)          # 1 = perfectly upright, 0 = on its back

    # ---------------- gated speed --------------------------------------------
    speed_raw = torch.log1p(torch.clamp(v_fwd, min=0.0)) / 2.4
    gated_speed = 2.0 * gate * speed_raw
    gated_alive = 0.5 * gate

    # ---------------- regularisers / tumble penalty --------------------------
    tumble_penalty = -0.5 * (1.0 - gate)
    lateral_penalty = -0.02 * torch.abs(v_lat)
    ang_vel_penalty = -0.001 * torch.sum(torch.square(ang_vel), dim=-1)
    dof_vel_penalty = -0.00005 * torch.sum(torch.square(dof_vel), dim=-1)
    action_penalty = -0.002 * torch.sum(torch.square(actions), dim=-1)

    reward = (gated_speed + gated_alive + tumble_penalty + lateral_penalty +
              ang_vel_penalty + dof_vel_penalty + action_penalty)

    components = {
        "gated_speed": gated_speed,
        "upright_gate": gate,
        "gated_alive": gated_alive,
        "tumble_penalty": tumble_penalty,
        "lateral_penalty": lateral_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "dof_vel_penalty": dof_vel_penalty,
        "action_penalty": action_penalty,
    }
    return reward, components
