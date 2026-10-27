@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward fast forward locomotion while remaining upright and using
    reasonably smooth actions.

    root_states layout:
        [root_position(3), root_quaternion(wxyz)(4),
         linear_velocity(3), angular_velocity(3)]
    """

    linear_velocity = root_states[:, 7:10]
    horizontal_velocity = linear_velocity[:, 0:2]

    # Root orientation in world coordinates.
    quat = root_states[:, 3:7]
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]

    # World-space direction of the humanoid's local +X (forward) axis.
    forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    forward_y = 2.0 * (qx * qy + qw * qz)
    forward_z = 2.0 * (qx * qz - qw * qy)
    forward_xy = torch.stack((forward_x, forward_y), dim=1)
    forward_xy = forward_xy / (
        torch.norm(forward_xy, p=2, dim=1, keepdim=True) + 1.0e-6
    )

    # Horizontal speed specifically in the direction the humanoid faces.
    forward_speed = torch.sum(horizontal_velocity * forward_xy, dim=1)

    # World-space vertical component of the humanoid's local +Z (up) axis.
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Soft speed transformation keeps the reward bounded while retaining a
    # monotonically increasing incentive to run faster.
    speed_temperature: float = 8.0
    soft_forward_speed = speed_temperature * torch.tanh(
        forward_speed / speed_temperature
    )

    # Smooth uprightness/aliveness factor. Multiplying locomotion reward by
    # this factor prevents the agent from earning speed reward while fallen.
    upright_temperature: float = 0.25
    upright_factor = torch.sigmoid(
        (body_up_z - 0.25) / upright_temperature
    )

    forward_speed_reward = upright_factor * soft_forward_speed
    upright_bonus = upright_factor

    # Small regularizers discourage excessive control effort and bouncing.
    control_cost = 0.001 * torch.sum(actions * actions, dim=1)
    vertical_velocity_cost = 0.02 * torch.abs(linear_velocity[:, 2])

    total_reward = (
        forward_speed_reward
        + upright_bonus
        - control_cost
        - vertical_velocity_cost
    )

    components: Dict[str, torch.Tensor] = {
        "forward_speed_reward": forward_speed_reward,
        "upright_bonus": upright_bonus,
        "control_cost": -control_cost,
        "vertical_velocity_cost": -vertical_velocity_cost,
    }

    return total_reward, components
