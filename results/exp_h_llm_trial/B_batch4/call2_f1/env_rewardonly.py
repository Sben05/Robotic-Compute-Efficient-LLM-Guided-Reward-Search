@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Variant 1 - Smooth exponential speed reward.
    The forward (+x world) running speed is turned into a bounded, strictly
    increasing reward with exp(v / vel_temp).  A small upright bonus keeps the
    torso vertical (computed from the torso quaternion) and a quadratic control
    cost discourages jittery actuator behaviour.
    """
    vel_temp: float = 4.0        # temperature of the exponential speed shaping
    upright_coeff: float = 0.5   # weight of the "stay vertical" bonus
    ctrl_coeff: float = 0.01     # quadratic penalty on the motor commands

    lin_vel = root_states[:, 7:10]
    fwd_vel = lin_vel[:, 0]

    # running speed reward (increasing, saturating function of forward speed)
    run_reward = torch.exp(fwd_vel / vel_temp)

    # torso up-vector z-component from the quaternion (w, x, y, z)
    qx = root_states[:, 4]
    qy = root_states[:, 5]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright = torch.clamp(up_z, min=0.0, max=1.0)

    # control effort
    ctrl_cost = torch.sum(torch.square(actions), dim=-1)

    total = run_reward + upright_coeff * upright - ctrl_coeff * ctrl_cost
    return total, {"run_speed": run_reward, "upright": upright, "ctrl_cost": ctrl_cost}
