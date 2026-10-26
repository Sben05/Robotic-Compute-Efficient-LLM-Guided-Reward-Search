@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    actions: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward for making the ant run forward (toward the target) as fast as possible.

    Root-state layout (Isaac Gym): [pos(3), quat(4), lin_vel(3), ang_vel(3), ...]
    heading_vec: horizontal unit vector pointing from the torso toward the target
                 (i.e. the task's "forward" direction).
    up_vec:     torso up-direction projected on the world horizontal basis (world up = +z).
    """

    # ---- Primary term: forward running speed (m/s along the forward/heading direction) ----
    lin_vel = root_states[:, 7:10]
    forward_speed = torch.sum(lin_vel * heading_vec, dim=-1)
    speed_reward = forward_speed

    # ---- Posture term: keep the torso upright so the agent does not fall and flip over ----
    upright = torch.clamp(up_vec[:, 2], -1.0, 1.0)
    upright_reward = 0.5 * upright

    # ---- Survival bonus: small constant to discourage early termination / collapsing ----
    alive_reward = 0.1 * torch.ones_like(forward_speed)

    # ---- Control effort penalty: discourages wild, high-energy motions that destabilize ----
    action_penalty = -0.01 * torch.sum(actions * actions, dim=-1)

    total_reward = speed_reward + upright_reward + alive_reward + action_penalty

    components: Dict[str, torch.Tensor] = {
        "forward_speed": speed_reward,
        "upright": upright_reward,
        "alive": alive_reward,
        "action_penalty": action_penalty,
    }
    return total_reward, components
