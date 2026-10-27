@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   actions: torch.Tensor,
                   dof_vel: torch.Tensor,
                   dof_vel_scale: float,
                   basis_vec0: torch.Tensor,
                   basis_vec1: torch.Tensor,
                   up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Speed-centric reward: make the ant run forward (the way its torso faces) as fast as possible.

    Main term  : wide, smooth forward-velocity reward (tanh with a large temperature so it
                 does not saturate at realistic running speeds).
    Regularisers: uprightness, tiny alive bonus, small action-MSE and dof-velocity penalties.
    """

    # ------------------- hyper-parameters (temperatures included) -------------------
    speed_temp: float = 4.0      # width (m/s) of the smooth speed term -- deliberately wide
    w_speed: float = 1.0         # weight of the forward-speed reward
    w_upright: float = 0.25      # small regulariser: stay on the feet
    w_alive: float = 0.10        # small regulariser: stay healthy
    w_action: float = 0.02       # small regulariser: smooth / small actions
    w_dof_vel: float = 0.002     # small regulariser: smooth joint velocities

    num_envs = root_states.shape[0]

    # ------------------- state extraction -------------------
    lin_vel = root_states[:, 7:10]      # world-frame linear velocity
    quat = root_states[:, 3:7]          # torso orientation (x, y, z, w)

    # body basis vectors expressed in the world frame (env supplies them)
    fwd_basis = basis_vec1.reshape(1, 3).expand(num_envs, 3)   # heading / forward axis
    up_basis = basis_vec0.reshape(1, 3).expand(num_envs, 3)    # up axis

    forward_world = _quat_rotate(quat, fwd_basis)   # (N, 3) where the torso is pointing
    up_world = _quat_rotate(quat, up_basis)         # (N, 3) body's up axis in world frame

    # ------------------- forward speed -------------------
    v_fwd = torch.sum(lin_vel * forward_world, dim=-1)   # signed body-forward speed

    # smooth, non-saturating speed reward (monotone, dense gradient at v_fwd = 0)
    forward_speed = speed_temp * torch.tanh(v_fwd / speed_temp)

    # ------------------- small regularisers -------------------
    up_proj = torch.clamp(up_world[:, up_axis_idx], min=-1.0, max=1.0)   # ~1 when upright
    upright = up_proj
    alive = torch.ones_like(v_fwd)

    action_cost = torch.mean(actions * actions, dim=-1)
    dof_vel_norm = dof_vel * dof_vel_scale
    dof_vel_cost = torch.mean(dof_vel_norm * dof_vel_norm, dim=-1)

    # ------------------- combine -------------------
    rew_speed = w_speed * forward_speed
    rew_upright = w_upright * upright
    rew_alive = w_alive * alive
    pen_action = -w_action * action_cost
    pen_dof_vel = -w_dof_vel * dof_vel_cost

    total = rew_speed + rew_upright + rew_alive + pen_action + pen_dof_vel

    components: Dict[str, torch.Tensor] = {
        "forward_speed": rew_speed,        # main term
        "forward_velocity": v_fwd,          # raw signed body-forward speed (logging)
        "upright": rew_upright,
        "alive": rew_alive,
        "action_penalty": pen_action,
        "dof_vel_penalty": pen_dof_vel,
    }
    return total, components
