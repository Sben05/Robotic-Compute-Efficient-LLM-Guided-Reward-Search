@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,   # (N, 3) world position of the ball
    ball_linvels: torch.Tensor,     # (N, 3) linear velocity of the ball
    sensor_forces: torch.Tensor,    # (N, S, 3) force sensor readings
    sensor_torques: torch.Tensor,   # (N, S, 3) torque sensor readings
    dof_velocities: torch.Tensor,   # (N, num_dofs)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ------------------------- fixed constants -------------------------
    table_center_x: float = 0.0     # assumed centre of the table top (world frame)
    table_center_y: float = 0.0
    floor_z: float = 0.25           # below this height the ball has reached the ground
    fall_speed_thresh: float = 0.2  # downward speed (m/s) that counts as "falling"
    force_ref: float = 5.0          # normal force (N) that counts as a solid contact

    # ------------------------- temperatures -----------------------------
    temp_contact: float = 5.0       # 1/force scale of the tanh transform
    temp_still: float = 0.5         # (m/s)^2
    temp_center: float = 0.5        # m^2
    temp_torque: float = 20.0       # normaliser of the torque effort term

    # ------------------------- reward weights --------------------------
    w_alive: float = 1.0
    w_contact: float = 2.0
    w_still: float = 1.0
    w_center: float = 0.5
    w_fall_speed: float = 3.0
    w_fall_height: float = 2.0
    w_ctrl: float = 0.01
    w_torque: float = 0.0005

    # ------------------------- ball state ------------------------------
    bx = ball_positions[:, 0] - table_center_x
    by = ball_positions[:, 1] - table_center_y
    bz = ball_positions[:, 2]

    vx = ball_linvels[:, 0]
    vy = ball_linvels[:, 1]
    vz = ball_linvels[:, 2]

    xy_dist_sq = bx * bx + by * by
    speed_sq = vx * vx + vy * vy + vz * vz

    # ------------------------- contact with the table -------------------
    # normal (z) component of every force sensor, strongest one is used
    fz = torch.abs(sensor_forces[:, :, 2])
    f_contact = torch.max(fz, dim=1)[0]
    r_contact = torch.tanh(f_contact / temp_contact)

    # ------------------------- ball should be calm ----------------------
    r_still = torch.exp(-speed_sq / temp_still)

    # ------------------------- ball should stay on the table ------------
    r_center = torch.exp(-xy_dist_sq / temp_center)

    # ------------------------- fall penalties ---------------------------
    fall_v = torch.clamp(-vz - fall_speed_thresh, min=0.0)
    p_fall_speed = fall_v * fall_v

    fall_h = torch.clamp(floor_z - bz, min=0.0)
    p_fall_height = fall_h * fall_h

    # ------------------------- effort / smoothness ----------------------
    actuated_dof_indices = torch.tensor([1, 3, 5], device=ball_positions.device)
    dof_vel = dof_velocities[:, actuated_dof_indices]
    p_ctrl = torch.sum(dof_vel * dof_vel, dim=1)

    torque_sq = torch.sum(sensor_torques * sensor_torques, dim=2)
    p_torque = torch.sum(torque_sq, dim=1) / temp_torque

    # ------------------------- total reward -----------------------------
    r_alive = torch.ones_like(bz)

    reward = (
        w_alive * r_alive
        + w_contact * r_contact
        + w_still * r_still
        + w_center * r_center
        - w_fall_speed * p_fall_speed
        - w_fall_height * p_fall_height
        - w_ctrl * p_ctrl
        - w_torque * p_torque
    )

    components: Dict[str, torch.Tensor] = {
        "alive": r_alive,
        "contact": r_contact,
        "still": r_still,
        "center": r_center,
        "fall_speed": p_fall_speed,
        "fall_height": p_fall_height,
        "control_effort": p_ctrl,
        "torque_effort": p_torque,
    }
    return reward, components
