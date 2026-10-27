@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                     targets: torch.Tensor,
                     heading_vec: torch.Tensor,
                     actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- kinematics ----------------
    lin_vel = root_states[:, 7:10]
    forward_vel = lin_vel[:, 0]

    # ---------------- exponential shaping around a target speed ----------------
    target_speed = 4.0
    vel_temp = 2.0
    speed_gap = torch.abs(forward_vel - target_speed) / vel_temp
    speed_reward = torch.exp(-speed_gap)          # 1.0 exactly at target speed

    # ---------------- heading alignment ----------------
    # cosine between torso forward axis and horizontal direction to the target
    pos = root_states[:, 0:3]
    to_target = targets - pos
    to_target = torch.stack((to_target[:, 0], to_target[:, 1],
                             torch.zeros_like(to_target[:, 2])), dim=1)
    dir_norm = torch.norm(to_target, p=2, dim=1).unsqueeze(1) + 1.0e-6
    dir_to_target = to_target / dir_norm
    head_norm = torch.norm(heading_vec, p=2, dim=1).unsqueeze(1) + 1.0e-6
    head = heading_vec / head_norm
    alignment = torch.sum(dir_to_target * head, dim=1)     # in [-1, 1]

    # ---------------- regularisers ----------------
    ctrl_cost = torch.sum(torch.square(actions), dim=1)

    speed_weight = 1.0
    align_weight = 0.3
    ctrl_weight = 0.01

    total = (speed_weight * speed_reward
             + align_weight * alignment
             - ctrl_weight * ctrl_cost)

    components = {
        "speed_shaping": speed_weight * speed_reward,
        "heading_alignment": align_weight * alignment,
        "control_cost": -ctrl_weight * ctrl_cost,
    }
    return total, components
