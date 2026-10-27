@torch.jit.script
def compute_reward(potentials: torch.Tensor,
                     prev_potentials: torch.Tensor,
                     root_states: torch.Tensor,
                     up_vec: torch.Tensor,
                     actions: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- potential-based progress ----------------
    # potentials = -||target - pos|| / dt  ->  difference is the closing rate [m/s]
    progress_rate = potentials - prev_potentials
    progress_dist = progress_rate * dt            # signed distance closed this step [m]
    progress_weight = 10.0
    progress_reward = progress_weight * progress_dist

    # ---------------- survival terms ----------------
    upright = up_vec[:, 2]
    alive = torch.ones_like(progress_dist)
    survival_reward = 0.1 * alive * torch.clamp(upright, min=0.0)  # alive AND on its feet

    # ---------------- small speed disambiguation ----------------
    # (rewards sprinting rather than slow shuffling towards the target)
    forward_vel = root_states[:, 7]
    speed_reward = 0.2 * forward_vel

    # ---------------- regularisers ----------------
    ctrl_cost = torch.sum(torch.square(actions), dim=1)
    ctrl_weight = 0.01

    total = (progress_reward
             + survival_reward
             + speed_reward
             - ctrl_weight * ctrl_cost)

    components = {
        "progress": progress_reward,
        "survival": survival_reward,
        "forward_speed": speed_reward,
        "control_cost": -ctrl_weight * ctrl_cost,
    }
    return total, components
