@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,       # [N, 13]: pos(0:3), quat(3:7), lin_vel(7:10), ang_vel(10:13)
    heading_vec: torch.Tensor,       # [N, 3]: body forward axis in world frame
    up_vec: torch.Tensor,            # [N, 3]: body up axis in world frame
    actions: torch.Tensor,           # [N, n_dofs]
    dof_pos: torch.Tensor,           # [N, n_dofs]
    dof_limits_lower: torch.Tensor,  # [n_dofs]
    dof_limits_upper: torch.Tensor,  # [n_dofs]
    dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- hyper-parameters (not env inputs) ----------------
    upright_threshold: float = 0.5     # cos(60 deg): tilt where the gate starts closing
    upright_temperature: float = 0.15  # sharpness of the soft uprightness gate
    upright_floor: float = 0.35        # hard cut-off: tumbling gets no speed reward
    max_forward_speed: float = 5.0     # clip speed outliers
    action_reg_coef: float = 0.002
    joint_limit_coef: float = 0.05
    min_torso_height: float = 0.2
    height_coef: float = 1.0
    alive_rate: float = 1.0            # per second

    lin_vel = root_states[:, 7:10]
    torso_z = root_states[:, 2]

    # --------------- forward speed along the body's heading ---------------
    fwd_dir = heading_vec[:, 0:3]
    forward_vel = torch.sum(lin_vel * fwd_dir, dim=-1)
    forward_vel = torch.clamp(forward_vel, min=-max_forward_speed, max=max_forward_speed)

    # ------------------------ posture gate -------------------------------
    up_z = up_vec[:, 2]
    soft_gate = torch.sigmoid((up_z - upright_threshold) / upright_temperature)
    hard_gate = torch.clamp((up_z - upright_floor) / (1.0 - upright_floor), min=0.0, max=1.0)
    upright_gate = soft_gate * hard_gate

    forward_speed_reward = forward_vel * upright_gate

    # --------------------------- penalties -------------------------------
    height_violation = torch.clamp(min_torso_height - torso_z, min=0.0)
    height_penalty = -height_coef * height_violation

    limit_violation = (
        torch.clamp(dof_pos - dof_limits_upper, min=0.0)
        + torch.clamp(dof_limits_lower - dof_pos, min=0.0)
    )
    joint_limit_penalty = -joint_limit_coef * torch.sum(limit_violation, dim=-1)

    action_penalty = -action_reg_coef * torch.sum(actions * actions, dim=-1)

    alive_bonus = torch.ones_like(forward_vel) * (alive_rate * dt)

    total_reward = (
        forward_speed_reward
        + height_penalty
        + joint_limit_penalty
        + action_penalty
        + alive_bonus
    )

    components = {
        "forward_speed": forward_speed_reward,      # main term: gated forward speed (m/s)
        "upright_gate": upright_gate,                # diagnostic: posture gate value
        "alive_bonus": alive_bonus,
        "height_penalty": height_penalty,
        "joint_limit_penalty": joint_limit_penalty,
        "action_penalty": action_penalty,
    }

    return total_reward, components
