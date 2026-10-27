@torch.jit.script
def compute_reward(root_states: torch.Tensor, up_vec: torch.Tensor, dof_vel: torch.Tensor,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Baseline running reward: forward speed, uprightness and small control penalties."""
    # torso linear velocity (world frame) and torso orientation (quaternion w,x,y,z)
    vel = root_states[:, 7:10]
    quat = root_states[:, 3:7]
    qw = quat[:, 0:1]
    qxyz = quat[:, 1:4]

    # body x-axis (forward) expressed in the world frame
    zeros = torch.zeros_like(qw)
    e_x = torch.cat((torch.ones_like(zeros), zeros, zeros), dim=-1)
    t1 = torch.cross(qxyz, e_x, dim=-1)
    t2 = torch.cross(qxyz, t1, dim=-1)
    fwd = e_x + 2.0 * (qw * t1 + t2)

    fwd_speed = torch.sum(vel * fwd, dim=-1)          # forward running speed [m/s]

    up_proj = up_vec[:, 2]                            # torso up axis vs. world up
    upright = torch.clamp(up_proj, min=0.0, max=1.0)
    fallen = torch.clamp(-up_proj, min=0.0)            # >0 when inverted
    torso_height = root_states[:, 2]

    # reward components
    speed_reward = 1.0 * fwd_speed
    upright_reward = 0.5 * upright
    fall_penalty = -3.0 * fallen - 1.0 * torch.clamp(0.8 - torso_height, min=0.0)
    height_reward = 0.2 * torch.exp(-torch.abs(torso_height - 1.3) / 0.3)
    energy_penalty = -0.002 * torch.sum(dof_vel * dof_vel, dim=-1)
    action_penalty = -0.01 * torch.sum(actions * actions, dim=-1)

    reward = speed_reward + upright_reward + fall_penalty + height_reward \
        + energy_penalty + action_penalty

    components: Dict[str, torch.Tensor] = {}
    components["forward_speed"] = speed_reward
    components["upright_bonus"] = upright_reward
    components["fall_penalty"] = fall_penalty
    components["torso_height"] = height_reward
    components["joint_energy"] = energy_penalty
    components["action_magnitude"] = action_penalty
    return reward, components
