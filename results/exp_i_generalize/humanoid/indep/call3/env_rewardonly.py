@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dof_vel: torch.Tensor,
                   targets: torch.Tensor, potentials: torch.Tensor,
                   prev_potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    num_envs = root_states.shape[0]
    device = root_states.device

    # ---------------- parse the root state ----------------
    torso_pos = root_states[:, 0:3]
    quat = root_states[:, 3:7]          # (x, y, z, w)
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]

    qx = quat[:, 0:1]
    qy = quat[:, 1:2]
    qz = quat[:, 2:3]
    qw = quat[:, 3:4]

    one = torch.ones((num_envs, 1), device=device)
    zero = torch.zeros((num_envs, 1), device=device)

    # ---- world-frame body forward axis (+x of the torso) ----
    tfx = 2.0 * (qy * zero - qz * zero)
    tfy = 2.0 * (qz * one - qx * zero)
    tfz = 2.0 * (qx * zero - qy * one)
    fwd_x = one + qw * tfx + (qy * tfz - qz * tfy)
    fwd_y = zero + qw * tfy + (qz * tfx - qx * tfz)
    fwd_z = zero + qw * tfz + (qx * tfy - qy * tfx)
    fwd_world = torch.cat((fwd_x, fwd_y, fwd_z), dim=1)

    # ---- world-frame body up axis (+z of the torso) ----
    tux = 2.0 * (qy * one - qz * zero)
    tuy = 2.0 * (qz * zero - qx * one)
    tuz = 2.0 * (qx * zero - qy * zero)
    up_x = zero + qw * tux + (qy * tuz - qz * tuy)
    up_y = zero + qw * tuy + (qz * tux - qx * tuz)
    up_z = one + qw * tuz + (qx * tuy - qy * tux)
    up_world = torch.cat((up_x, up_y, up_z), dim=1)

    # ---------------- running speed ----------------
    fwd_h = fwd_world[:, 0:2]
    fwd_dir = fwd_h / (torch.norm(fwd_h, dim=1, keepdim=True) + 1e-6)

    vel_h = lin_vel[:, 0:2]
    fwd_speed = (vel_h * fwd_dir).sum(dim=1)
    fwd_speed = torch.clamp(fwd_speed, min=0.0)

    # saturating speed reward (temperature = desired max running speed, m/s)
    speed_temperature = 6.0
    r_speed = speed_temperature * torch.tanh(fwd_speed / speed_temperature)

    # ---------------- staying alive / upright ----------------
    r_alive = torch.ones(num_envs, device=device)
    r_upright = torch.clamp(up_world[:, 2], min=0.0, max=1.0)

    # ---------------- direction towards the goal ----------------
    to_target = targets[:, 0:2] - torso_pos[:, 0:2]
    to_target_dir = to_target / (torch.norm(to_target, dim=1, keepdim=True) + 1e-6)
    r_target_align = torch.clamp((fwd_dir * to_target_dir).sum(dim=1), min=0.0, max=1.0)

    # potential-based progress rate towards the goal (m/s)
    r_progress = torch.clamp(prev_potentials - potentials, min=-speed_temperature,
                             max=speed_temperature)

    # ---------------- smoothness / energy regularizers ----------------
    r_action_pen = -0.01 * torch.sum(actions * actions, dim=1)
    r_angvel_pen = -0.005 * torch.sum(ang_vel * ang_vel, dim=1)
    r_joint_pen = -0.0005 * torch.mean(torch.clamp(dof_vel * dof_vel, min=0.0, max=100.0), dim=1)

    # ---------------- weights ----------------
    w_speed = 1.0
    w_alive = 0.5
    w_upright = 1.0
    w_target = 0.5
    w_progress = 0.3

    comp_speed = w_speed * r_speed
    comp_alive = w_alive * r_alive
    comp_upright = w_upright * r_upright
    comp_target = w_target * r_target_align
    comp_progress = w_progress * r_progress

    total = (comp_speed + comp_alive + comp_upright + comp_target + comp_progress
             + r_action_pen + r_angvel_pen + r_joint_pen)

    components = {
        "run_speed": comp_speed,
        "alive": comp_alive,
        "upright": comp_upright,
        "target_alignment": comp_target,
        "progress_to_target": comp_progress,
        "action_penalty": r_action_pen,
        "ang_vel_penalty": r_angvel_pen,
        "joint_vel_penalty": r_joint_pen,
    }

    return total, components
