@torch.jit.script
def compute_reward(ball_positions: torch.Tensor,
                     ball_linvels: torch.Tensor,
                     dof_positions: torch.Tensor,
                     dof_velocities: torch.Tensor,
                     sensor_forces: torch.Tensor,
                     sensor_torques: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Full composite reward: every physically meaningful cue is used, each with its own
    temperature, and the total reward is normalised to roughly [-1, 1].

      height   : ball stays at table-top height,
      center   : ball stays over the middle of the table,
      still    : ball is not moving,
      contact  : the table sensor still feels the load of the ball (ball on the table),
      dof      : the table is moved smoothly (small joint velocities),
      effort   : the joint torques stay small (energy efficient, gentle control),
      posture  : the actuated joints do not drift to extreme configurations,
      fall     : hard penalty once the ball has left the table.
    """
    # ------------------- tunables (constants, not env inputs) -------------------
    target_ball_z: float = 0.85
    floor_z: float = 0.30
    sigma_z: float = 0.15
    sigma_xy: float = 0.40
    sigma_v: float = 0.50
    sigma_force: float = 1.0
    sigma_dof: float = 1.50
    sigma_torque: float = 2.0
    w_height: float = 1.0
    w_center: float = 1.0
    w_still: float = 1.0
    w_contact: float = 1.0
    w_dof: float = 0.5
    w_effort: float = 0.3
    w_posture: float = 0.2
    w_fall: float = 3.0
    torque_limit: float = 5.0
    joint_limit: float = 1.0
    eps: float = 1e-8
    # -----------------------------------------------------------------------------

    px = ball_positions[:, 0]
    py = ball_positions[:, 1]
    pz = ball_positions[:, 2]
    vx = ball_linvels[:, 0]
    vy = ball_linvels[:, 1]
    vz = ball_linvels[:, 2]

    # height (one sided: never rewarded for throwing the ball up)
    dz = torch.clamp(target_ball_z - pz, min=0.0)
    r_height = torch.exp(-(dz * dz) / (2.0 * sigma_z * sigma_z))

    # centring over the table
    r_center = torch.exp(-(px * px + py * py) / (2.0 * sigma_xy * sigma_xy))

    # ball at rest
    r_still = torch.exp(-(vx * vx + vy * vy + vz * vz) / (2.0 * sigma_v * sigma_v))

    # contact load on the table sensor
    f = sensor_forces[:, 0, :]
    fx = f[:, 0]
    fy = f[:, 1]
    fz = f[:, 2]
    f_mag = torch.sqrt(fx * fx + fy * fy + fz * fz + eps)
    r_contact = 1.0 - torch.exp(-f_mag / sigma_force)

    # smooth table motion (actuated dofs are indices 1, 3, 5)
    dv1 = dof_velocities[:, 1]
    dv2 = dof_velocities[:, 3]
    dv3 = dof_velocities[:, 5]
    dof_sq = dv1 * dv1 + dv2 * dv2 + dv3 * dv3
    r_dof = torch.exp(-dof_sq / (2.0 * sigma_dof * sigma_dof))

    # control effort (joint torques)
    t = sensor_torques[:, 0, :]
    tx = t[:, 0]
    ty = t[:, 1]
    tz = t[:, 2]
    t_mag = torch.sqrt(tx * tx + ty * ty + tz * tz + eps)
    r_effort = -w_effort * torch.clamp(t_mag / torque_limit, min=0.0, max=1.0)

    # posture regularisation of the actuated joints (hinge, only beyond the limit)
    a1 = dof_positions[:, 1]
    a2 = dof_positions[:, 3]
    a3 = dof_positions[:, 5]
    post_sq = a1 * a1 + a2 * a2 + a3 * a3
    r_posture = -w_posture * torch.clamp(post_sq - joint_limit * joint_limit, min=0.0)

    # the ball fell off the table
    fall = torch.clamp((floor_z - pz) / floor_z, min=0.0, max=1.0)
    r_fall = -w_fall * fall

    reward = (w_height * r_height + w_center * r_center + w_still * r_still
              + w_contact * r_contact + w_dof * r_dof + r_effort + r_posture + r_fall)

    components: Dict[str, torch.Tensor] = {}
    components["height"] = r_height
    components["center"] = r_center
    components["still"] = r_still
    components["contact"] = r_contact
    components["dof_smooth"] = r_dof
    components["effort"] = r_effort
    components["posture"] = r_posture
    components["fall"] = r_fall
    return reward, components
