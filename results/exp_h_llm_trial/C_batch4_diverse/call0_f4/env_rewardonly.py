@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor,
                     dof_vel: torch.Tensor,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Strategy 4 - exponential shaping.

    exp(-gap / tau) of the shortfall between the current forward speed and a
    desired target speed (so the reward is ~1 at the target speed and decays
    smoothly when too slow), plus a linear bonus for exceeding the target
    speed and an exponential heading-alignment term toward the target.
    """
    quat = root_states[:, 3:7]
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    fwd = _body_axis_4(quat, 0)
    up = _body_axis_4(quat, 2)

    fwd_xy = torch.stack((fwd[:, 0], fwd[:, 1]), dim=-1)
    vel_xy = torch.stack((lin_vel[:, 0], lin_vel[:, 1]), dim=-1)
    fwd_hat = fwd_xy / torch.norm(fwd_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    v_fwd = torch.sum(vel_xy * fwd_hat, dim=-1)

    # heading alignment with the direction to the target
    to_target_xy = targets[:, 0:2] - root_states[:, 0:2]
    to_target_hat = to_target_xy / torch.norm(to_target_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    align = torch.sum(fwd_hat * to_target_hat, dim=-1)      # in [-1, 1]

    # ---------------- temperatures / targets ---------------------------------
    v_target_speed = 5.0     # desired forward speed (m/s)
    tau_speed = 1.0          # temperature of the speed shaping
    tau_head = 0.5           # temperature of the heading shaping

    # ---------------- exponential shaping ------------------------------------
    gap = torch.clamp(v_target_speed - v_fwd, min=0.0)
    speed_exp = torch.exp(-gap / tau_speed)
    overspeed = torch.clamp(v_fwd - v_target_speed, min=0.0)
    heading = torch.exp(-(1.0 - align) / tau_head)

    # ---------------- survival + regularisers --------------------------------
    alive = torch.ones_like(v_fwd)
    upright_penalty = -0.05 * torch.clamp(1.0 - up[:, 2], min=0.0)
    ang_vel_penalty = -0.001 * torch.sum(torch.square(ang_vel), dim=-1)
    dof_vel_penalty = -0.00005 * torch.sum(torch.square(dof_vel), dim=-1)
    action_penalty = -0.002 * torch.sum(torch.square(actions), dim=-1)

    reward = (2.0 * speed_exp + 0.05 * overspeed + 0.5 * heading + 0.5 * alive +
              upright_penalty + ang_vel_penalty + dof_vel_penalty + action_penalty)

    components = {
        "speed_exp": 2.0 * speed_exp,
        "speed_gap": gap,
        "overspeed_bonus": 0.05 * overspeed,
        "heading_alignment": 0.5 * heading,
        "alive_bonus": 0.5 * alive,
        "upright_penalty": upright_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "dof_vel_penalty": dof_vel_penalty,
        "action_penalty": action_penalty,
    }
    return reward, components
