@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_pos: torch.Tensor,
                     dof_limits_lower: torch.Tensor, dof_limits_upper: torch.Tensor,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Variant 4 - Survival + exponential speed with posture / joint-limit shaping.
    The agent gets a constant "alive" bonus (so it learns not to flip over),
    an exponentially increasing reward for forward speed, an upright bonus and
    penalties for collapsing (torso too low) and for pushing the joints past
    their limits, plus a small motor-control cost.
    """
    vel_temp: float = 5.0        # temperature of the exponential speed reward
    alive_bonus: float = 1.0
    upright_coeff: float = 1.0
    height_coeff: float = 3.0    # penalty for dropping the torso
    limit_coeff: float = 0.5     # penalty for exceeding joint limits
    ctrl_coeff: float = 0.005
    min_height: float = 0.3

    lin_vel = root_states[:, 7:10]
    torso_z = root_states[:, 2]
    fwd_vel = lin_vel[:, 0]

    run_reward = torch.exp(torch.clamp(fwd_vel, min=-1.0, max=12.0) / vel_temp)
    alive = torch.ones_like(fwd_vel) * alive_bonus

    # torso uprightness from quaternion
    qx = root_states[:, 4]
    qy = root_states[:, 5]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright = torch.clamp(up_z, min=0.0, max=1.0)

    # collapsing / falling penalty
    fall_pen = torch.clamp(min_height - torso_z, min=0.0)

    # joint limit violation penalty
    lower_viol = torch.clamp(dof_limits_lower - dof_pos, min=0.0)
    upper_viol = torch.clamp(dof_pos - dof_limits_upper, min=0.0)
    limit_pen = torch.sum(lower_viol + upper_viol, dim=-1)

    ctrl_cost = torch.sum(torch.square(actions), dim=-1)

    total = (run_reward
             + alive
             + upright_coeff * upright
             - height_coeff * fall_pen
             - limit_coeff * limit_pen
             - ctrl_coeff * ctrl_cost)
    return total, {"run_speed": run_reward, "alive": alive, "upright": upright,
                   "fall_pen": fall_pen, "limit_pen": limit_pen, "ctrl_cost": ctrl_cost}
