@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
    targets: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Strategy 1 - speed centric.

    The dominant term is the forward velocity of the torso measured along the
    ant's own heading direction (``heading_vec`` is the world-frame forward
    axis of the torso).  The smooth (tanh) shaping uses a *wide* temperature
    (8 m/s) so that realistic running speeds (<= ~6 m/s) stay in the almost
    linear regime and the reward never saturates.  Everything else is a small
    regulariser (joint velocity, action magnitude, mild tilt, alive bonus).
    """
    # ---------------- forward speed along the ant's heading -----------------
    lin_vel = root_states[:, 7:10]                      # world linear velocity
    heading_len = torch.norm(heading_vec, dim=-1, keepdim=True).clamp_min(1e-6)
    heading_unit = heading_vec / heading_len            # unit forward direction
    forward_speed = torch.sum(lin_vel * heading_unit, dim=-1)

    # wide temperature -> no saturation at running speeds
    speed_temp: float = 8.0
    speed_reward = speed_temp * torch.tanh(forward_speed / speed_temp)

    # ---------------- small regularisers -------------------------------------
    joint_vel_penalty = -0.0005 * torch.mean(torch.abs(dof_vel), dim=-1)
    action_penalty = -0.005 * torch.mean(torch.square(actions), dim=-1)
    # keep the torso roughly upright (small weight, does not dominate)
    up_z = torch.clamp(up_vec[:, 2], min=0.0, max=1.0)
    tilt_penalty = -0.10 * (1.0 - up_z)
    alive_bonus = torch.ones_like(forward_speed) * 0.05

    reward = (speed_reward + alive_bonus + joint_vel_penalty
              + action_penalty + tilt_penalty)

    components = {
        "speed_reward": speed_reward,
        "alive_bonus": alive_bonus,
        "joint_vel_penalty": joint_vel_penalty,
        "action_penalty": action_penalty,
        "tilt_penalty": tilt_penalty,
    }
    return reward, components
