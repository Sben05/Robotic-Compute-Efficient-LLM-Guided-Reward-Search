@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ----------------- reward coefficients -----------------
    forward_vel_coef: float = 1.0        # weight on forward speed (m/s)
    min_forward_vel: float = -1.0        # clip: running backwards is penalized but bounded
    alive_coef: float = 0.2              # bonus for staying upright
    action_cost_coef: float = 0.01       # control effort penalty
    dof_vel_cost_coef: float = 1.0e-4    # joint speed (energy) penalty
    lateral_vel_cost_coef: float = 0.02  # side-slip penalty (keep running straight)

    # ----------------- body frame axes in world frame -----------------
    # Isaac Gym root-state quaternion layout: (w, x, y, z)
    quat = root_states[:, 3:7]
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    # body forward axis (x) expressed in world coordinates
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + w * z)
    forward_xy = torch.stack((fwd_x, fwd_y), dim=-1)
    forward_xy = forward_xy / torch.clamp(torch.norm(forward_xy, dim=-1, keepdim=True), min=1.0e-6)

    # body up axis (z) expressed in world coordinates -> world-z component
    up_z = 1.0 - 2.0 * (x * x + y * y)

    # body lateral axis (y) projected on the ground plane
    lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=-1)

    # ----------------- kinematics -----------------
    vel_xy = root_states[:, 7:9]

    forward_vel = torch.sum(vel_xy * forward_xy, dim=-1)
    forward_vel = torch.clamp(forward_vel, min=min_forward_vel)

    lateral_vel = torch.sum(vel_xy * lateral_xy, dim=-1)

    # ----------------- reward components -----------------
    forward_velocity_reward = forward_vel_coef * forward_vel
    upright_alive_reward = alive_coef * torch.clamp(up_z, min=0.0, max=1.0)
    action_cost = -action_cost_coef * torch.sum(actions * actions, dim=-1)
    joint_velocity_cost = -dof_vel_cost_coef * torch.sum(dof_vel * dof_vel, dim=-1)
    lateral_velocity_cost = -lateral_vel_cost_coef * lateral_vel * lateral_vel

    total_reward = (forward_velocity_reward
                    + upright_alive_reward
                    + action_cost
                    + joint_velocity_cost
                    + lateral_velocity_cost)

    components: Dict[str, torch.Tensor] = {
        "forward_velocity": forward_velocity_reward,
        "upright_alive": upright_alive_reward,
        "action_cost": action_cost,
        "joint_velocity_cost": joint_velocity_cost,
        "lateral_velocity_cost": lateral_velocity_cost,
    }

    return total_reward, components
