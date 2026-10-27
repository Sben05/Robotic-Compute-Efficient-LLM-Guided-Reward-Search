@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                     up_vec: torch.Tensor,
                     dof_vel: torch.Tensor,
                     dof_vel_scale: float,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- kinematics / posture ----------------
    lin_vel = root_states[:, 7:10]
    forward_vel = lin_vel[:, 0]
    upright = up_vec[:, 2]                 # +1 upright, 0 on its side, -1 upside down

    # ---------------- posture gate ----------------
    # Smooth sigmoid gate: only an (almost) upright torso collects the speed
    # reward, so tumbling/flipping can never be rewarded.
    gate_temp = 0.15
    upright_gate = torch.sigmoid((upright - 0.6) / gate_temp)

    gated_speed = forward_vel * upright_gate
    vel_soft_cap = 8.0
    speed_reward = vel_soft_cap * torch.tanh(gated_speed / vel_soft_cap)

    # ---------------- tumbling penalties ----------------
    tumble_penalty = torch.relu(0.4 - upright)   # not standing -> negative
    flip_penalty = torch.relu(-upright)          # upside down -> extra negative

    # ---------------- small regularisers ----------------
    ctrl_cost = torch.sum(torch.square(actions), dim=1)
    dof_vel_cost = torch.sum(torch.square(dof_vel * dof_vel_scale), dim=1)

    speed_weight = 1.0
    posture_bonus_weight = 0.1
    tumble_weight = 1.0
    flip_weight = 0.5
    ctrl_weight = 0.01
    dof_vel_weight = 0.0005

    total = (speed_weight * speed_reward
             + posture_bonus_weight * torch.relu(upright - 0.5)
             - tumble_weight * tumble_penalty
             - flip_weight * flip_penalty
             - ctrl_weight * ctrl_cost
             - dof_vel_weight * dof_vel_cost)

    components = {
        "gated_forward_speed": speed_weight * speed_reward,
        "upright_gate": upright_gate,
        "posture_bonus": posture_bonus_weight * torch.relu(upright - 0.5),
        "tumble_penalty": -tumble_weight * tumble_penalty,
        "flip_penalty": -flip_weight * flip_penalty,
        "control_cost": -ctrl_weight * ctrl_cost,
        "dof_velocity_cost": -dof_vel_weight * dof_vel_cost,
    }
    return total, components
