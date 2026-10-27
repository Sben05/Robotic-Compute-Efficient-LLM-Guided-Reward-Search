@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    actions: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # World-space horizontal velocity of the torso.
    horizontal_velocity = root_states[:, 7:9]

    # Direction the humanoid is currently facing, projected onto the ground plane.
    horizontal_heading = heading_vec[:, 0:2]
    heading_norm = torch.norm(horizontal_heading, p=2, dim=1, keepdim=True)
    horizontal_heading = horizontal_heading / (heading_norm + 1.0e-6)

    # Only forward motion is rewarded; backward motion can be made useful by turning.
    forward_speed = torch.sum(horizontal_velocity * horizontal_heading, dim=1)
    forward_speed = torch.clamp(forward_speed, min=0.0)

    # Bounded speed reward retains a gradient over useful humanoid running speeds.
    speed_temperature: float = 4.0
    normalized_speed = torch.tanh(forward_speed / speed_temperature)

    # Falling or excessive leaning removes most of the running reward.
    uprightness = torch.clamp(up_vec[:, 2], min=0.0, max=1.0)
    running_reward = normalized_speed * uprightness

    # Small regularizers discourage energy-wasting motion and unstable exploits.
    action_penalty = 5.0e-4 * torch.sum(torch.square(actions), dim=-1)
    vertical_velocity_penalty = 5.0e-3 * torch.square(root_states[:, 9])
    angular_velocity_penalty = 1.0e-4 * torch.sum(
        torch.square(root_states[:, 10:13]), dim=1
    )

    total_reward = (
        running_reward
        - action_penalty
        - vertical_velocity_penalty
        - angular_velocity_penalty
    )

    reward_components = {
        "forward_running_reward": running_reward,
        "action_penalty": -action_penalty,
        "vertical_velocity_penalty": -vertical_velocity_penalty,
        "angular_velocity_penalty": -angular_velocity_penalty,
    }

    return total_reward, reward_components
