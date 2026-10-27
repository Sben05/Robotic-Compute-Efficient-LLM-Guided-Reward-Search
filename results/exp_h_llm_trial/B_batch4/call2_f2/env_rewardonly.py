@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, dof_vel_scale: float,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Variant 2 - Dense linear speed reward with stability / energy penalties.
    The (clipped) forward velocity is rewarded linearly, which gives a strong
    constant gradient for every extra m/s.  Stability is enforced with a torso
    tilt penalty, an angular-velocity penalty and a (scaled) joint-velocity
    energy penalty, plus a small motor-control cost.
    """
    max_speed: float = 10.0      # speeds above this give no extra reward
    tilt_coeff: float = 1.0
    ang_coeff: float = 0.05
    joint_coeff: float = 0.05
    ctrl_coeff: float = 0.005

    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    # forward running speed (never rewarded for moving backwards)
    fwd_vel = torch.clamp(lin_vel[:, 0], min=0.0, max=max_speed)

    # torso tilt (0 when perfectly vertical)
    qx = root_states[:, 4]
    qy = root_states[:, 5]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    tilt_pen = torch.square(1.0 - torch.clamp(up_z, min=-1.0, max=1.0))

    # body spinning / shaking
    ang_pen = torch.sum(torch.square(ang_vel), dim=-1)

    # joint movement energy (using the env's normalisation scale)
    scaled_dof_vel = dof_vel * dof_vel_scale
    joint_pen = torch.sum(torch.square(scaled_dof_vel), dim=-1)

    ctrl_pen = torch.sum(torch.square(actions), dim=-1)

    total = (fwd_vel
             - tilt_coeff * tilt_pen
             - ang_coeff * ang_pen
             - joint_coeff * joint_pen
             - ctrl_coeff * ctrl_pen)
    return total, {"fwd_speed": fwd_vel, "tilt_pen": tilt_pen, "ang_pen": ang_pen,
                   "joint_pen": joint_pen, "ctrl_pen": ctrl_pen}
