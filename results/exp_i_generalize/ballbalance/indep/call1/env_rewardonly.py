@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,      # self.ball_positions   (num_envs, 3)
    ball_linvels: torch.Tensor,        # self.ball_linvels     (num_envs, 3)
    dof_velocities: torch.Tensor,      # self.dof_velocities   (num_envs, num_dofs)
    sensor_forces: torch.Tensor,       # self.sensor_forces    (num_envs, ..., 3)
    sensor_torques: torch.Tensor,      # self.sensor_torques   (num_envs, ..., 3)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Task: keep the ball on the table top without letting it fall.

    Design (all terms are per environment):
      * ball_stable   : the ball should be (almost) motionless  -> it is balanced
      * ball_centered : the ball should stay near the middle of the table
      * ball_height   : the ball must stay in the plausible "on the table" height
                        band (one sided, so throwing/bouncing the ball is never
                        rewarded, but sinking/falling through is punished)
      * no_falling    : explicit penalty for downward (falling) motion
      * inside_table  : the ball must stay inside the table footprint
                        (+ linear term that gives a gradient to bring it back)
      * smooth_motion : small joint velocities
      * low_force / low_torque : do not bang on the table with large forces

    NOTE: table_center / table_radius / z_ball_low / z_ball_high describe the
    table in the world frame (z up) and should be tuned to the asset.
    """

    num_envs = ball_positions.shape[0]

    # ------------------------------------------------------------------ #
    # table geometry (world frame, z is up)
    # ------------------------------------------------------------------ #
    table_center_x = 0.0
    table_center_y = 0.0
    table_radius = 0.45          # radius of the table top in the xy plane [m]
    z_ball_low = 0.45            # ball centre below this -> it left the table
    z_ball_high = 1.25           # ball centre above this -> lifted / flying
    z_slack_low = 0.15           # dead-band of the lower height limit
    z_slack_high = 0.50          # dead-band of the upper height limit

    # ------------------------------------------------------------------ #
    # reward weights
    # ------------------------------------------------------------------ #
    w_stable = 2.0
    w_center = 1.0
    w_height = 1.0
    w_fall = 1.0
    w_inside = 1.5
    w_outside = 1.0
    w_smooth = 0.05
    w_force = 0.05
    w_torque = 0.05

    # ------------------------------------------------------------------ #
    # temperatures (one per transformed component)
    # ------------------------------------------------------------------ #
    tau_stable = 0.50            # [m/s]   ball speed scale
    tau_center = 0.30            # [m]     distance to table centre scale
    tau_fall = 0.25              # [m/s]   downward speed scale
    tau_inside = 0.10            # [m]     distance beyond the table edge
    tau_smooth = 1.00            # [rad/s]^2
    tau_force = 1.00
    tau_torque = 1.00
    force_scale = 20.0           # same normalisation as in the observations
    torque_scale = 20.0

    # ------------------------------------------------------------------ #
    # ball kinematics
    # ------------------------------------------------------------------ #
    dx = ball_positions[:, 0] - table_center_x
    dy = ball_positions[:, 1] - table_center_y
    xy_dist = torch.sqrt(dx * dx + dy * dy)
    z = ball_positions[:, 2]

    vx = ball_linvels[:, 0]
    vy = ball_linvels[:, 1]
    vz = ball_linvels[:, 2]
    speed = torch.sqrt(vx * vx + vy * vy + vz * vz)

    # ------------------------------------------------------------------ #
    # reward components
    # ------------------------------------------------------------------ #
    # 1) the ball should be almost at rest -> it is balanced on the table
    r_stable = torch.exp(-speed / tau_stable)

    # 2) the ball should stay close to the centre of the table
    r_center = torch.exp(-xy_dist / tau_center)

    # 3) the ball must stay in the plausible "resting on the table" height band
    height_low = torch.clamp((z_ball_low - z) / z_slack_low, min=0.0, max=1.0)
    height_high = torch.clamp((z - z_ball_high) / z_slack_high, min=0.0, max=1.0)
    r_height = 1.0 - height_low - height_high

    # 4) explicit "do not fall" term (downward motion of the ball)
    fall_speed = torch.clamp(-vz, min=0.0)
    r_fall = torch.exp(-fall_speed / tau_fall)

    # 5) the ball must stay inside the table footprint
    off_dist = torch.clamp(xy_dist - table_radius, min=0.0)
    r_inside = torch.exp(-off_dist / tau_inside)

    # 6) the robot should move smoothly
    dof_vel_sq = torch.mean(dof_velocities.reshape(num_envs, -1) ** 2, dim=1)
    r_smooth = torch.exp(-dof_vel_sq / tau_smooth)

    # 7) the robot should not bang on the table with huge forces / torques
    force_sq = torch.mean(sensor_forces.reshape(num_envs, -1) ** 2, dim=1)
    r_force = torch.exp(-force_sq / (force_scale * force_scale) / tau_force)
    torque_sq = torch.mean(sensor_torques.reshape(num_envs, -1) ** 2, dim=1)
    r_torque = torch.exp(-torque_sq / (torque_scale * torque_scale) / tau_torque)

    # ------------------------------------------------------------------ #
    # total reward
    # ------------------------------------------------------------------ #
    reward = (
        w_stable * r_stable
        + w_center * r_center
        + w_height * r_height
        + w_fall * r_fall
        + w_inside * r_inside
        - w_outside * off_dist
        + w_smooth * r_smooth
        + w_force * r_force
        + w_torque * r_torque
    )

    components: Dict[str, torch.Tensor] = {
        "ball_stable": r_stable,
        "ball_centered": r_center,
        "ball_height": r_height,
        "no_falling": r_fall,
        "inside_table": r_inside,
        "smooth_motion": r_smooth,
        "low_force": r_force,
        "low_torque": r_torque,
    }
    return reward, components
