@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                     dof_vel: torch.Tensor,
                     dof_vel_scale: float,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- kinematics ----------------
    lin_vel = root_states[:, 7:10]          # world-frame linear velocity [m/s]
    forward_vel = lin_vel[:, 0]             # +x is the running direction

    # ---------------- main speed term ----------------
    # *Wide* soft saturation: tanh only compresses far beyond realistic sprint
    # speeds (ant ~4-6 m/s), so the learning signal does NOT flatten out at
    # running speed (a narrow tanh/clip would give ~zero gradient there).
    vel_soft_cap = 8.0
    speed_reward = vel_soft_cap * torch.tanh(forward_vel / vel_soft_cap)

    # ---------------- small regularisers ----------------
    ctrl_cost = torch.sum(torch.square(actions), dim=1)
    dof_vel_cost = torch.sum(torch.square(dof_vel * dof_vel_scale), dim=1)
    lateral_penalty = torch.square(lin_vel[:, 1])
    backward_penalty = torch.relu(-forward_vel)

    speed_weight = 1.0
    ctrl_weight = 0.01
    dof_vel_weight = 0.0005
    lateral_weight = 0.02
    backward_weight = 0.05

    total = (speed_weight * speed_reward
             - ctrl_weight * ctrl_cost
             - dof_vel_weight * dof_vel_cost
             - lateral_weight * lateral_penalty
             - backward_weight * backward_penalty)

    components = {
        "forward_speed": speed_weight * speed_reward,
        "control_cost": -ctrl_weight * ctrl_cost,
        "dof_velocity_cost": -dof_vel_weight * dof_vel_cost,
        "lateral_velocity": -lateral_weight * lateral_penalty,
        "backward_velocity": -backward_weight * backward_penalty,
    }
    return total, components
