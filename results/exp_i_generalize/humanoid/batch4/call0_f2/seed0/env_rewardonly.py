@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # -------- running speed, mapped to (0, 1) --------
    vel = root_states[:, 7:10]
    speed = torch.sqrt(vel[:, 0] * vel[:, 0] + vel[:, 1] * vel[:, 1] + 1e-6)
    temp_speed = 2.0
    speed_reward = 1.0 - torch.exp(-speed / temp_speed)

    # -------- uprightness from the torso quaternion (w, x, y, z) --------
    q = root_states[:, 3:7]
    qx = q[:, 1]
    qy = q[:, 2]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)          # world-z component of the torso up axis
    up_quality = torch.clamp(up_z, min=0.0, max=1.0)
    temp_upright = 0.5
    upright_reward = 1.0 - torch.exp(-up_quality / temp_upright)

    # -------- alive bonus from torso height --------
    height = root_states[:, 2]
    alive_reward = torch.clamp(height - 0.6, min=0.0, max=1.0)

    # -------- penalties --------
    action_penalty = 0.03 * torch.sum(actions * actions, dim=-1)
    dof_vel_penalty = 1.0e-3 * torch.sum(dof_vel * dof_vel, dim=-1)

    reward = 2.0 * speed_reward + 0.5 * upright_reward + 0.5 * alive_reward \
        - action_penalty - dof_vel_penalty

    components = {
        "speed": 2.0 * speed_reward,
        "upright": 0.5 * upright_reward,
        "alive": 0.5 * alive_reward,
        "action_penalty": -action_penalty,
        "dof_vel_penalty": -dof_vel_penalty,
    }
    return reward, components
