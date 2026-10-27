@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Strategy 1 - speed centric.

    Main term: a smooth, deliberately *wide* forward-speed term (tanh with a
    large scale + a log1p term that never saturates) so that running speeds of
    5-8 m/s are still in the active region of the reward.  Small regularisers
    (uprightness, lateral drift, angular / joint velocity, action energy) keep
    the gait sane without dominating the speed signal.
    """
    quat = root_states[:, 3:7]
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    fwd = _body_axis_1(quat, 0)   # ant forward (body +x) in world frame
    up = _body_axis_1(quat, 2)    # ant up (body +z) in world frame

    fwd_xy = torch.stack((fwd[:, 0], fwd[:, 1]), dim=-1)
    vel_xy = torch.stack((lin_vel[:, 0], lin_vel[:, 1]), dim=-1)
    fwd_hat = fwd_xy / torch.norm(fwd_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    lat_hat = torch.stack((-fwd_hat[:, 1], fwd_hat[:, 0]), dim=-1)

    v_fwd = torch.sum(vel_xy * fwd_hat, dim=-1)
    v_lat = torch.sum(vel_xy * lat_hat, dim=-1)
    up_z = up[:, 2]

    # ---------------- speed term (wide, non-saturating) -----------------------
    v_scale = 10.0                                   # far beyond normal ant running speeds
    speed_tanh = torch.tanh(torch.clamp(v_fwd, min=-1.0, max=40.0) / v_scale)
    speed_log = torch.log1p(torch.clamp(v_fwd, min=0.0)) / 2.4   # ~log(1+v)/log(11), unbounded
    forward_speed = 0.6 * speed_tanh + 0.8 * speed_log

    # ---------------- small regularisers --------------------------------------
    alive = torch.ones_like(v_fwd)
    upright_penalty = -0.05 * torch.clamp(1.0 - up_z, min=0.0)
    lateral_penalty = -0.02 * torch.abs(v_lat)
    ang_vel_penalty = -0.001 * torch.sum(torch.square(ang_vel), dim=-1)
    dof_vel_penalty = -0.00005 * torch.sum(torch.square(dof_vel), dim=-1)
    action_penalty = -0.002 * torch.sum(torch.square(actions), dim=-1)

    reward = (forward_speed + 0.5 * alive + upright_penalty + lateral_penalty +
              ang_vel_penalty + dof_vel_penalty + action_penalty)

    components = {
        "forward_speed": forward_speed,
        "speed_tanh": 0.6 * speed_tanh,
        "speed_log": 0.8 * speed_log,
        "alive_bonus": 0.5 * alive,
        "upright_penalty": upright_penalty,
        "lateral_penalty": lateral_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "dof_vel_penalty": dof_vel_penalty,
        "action_penalty": action_penalty,
    }
    return reward, components
