@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_vel_scale: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Root-state linear velocity is in world coordinates.
    planar_velocity = root_states[:, 7:9]
    angular_velocity = root_states[:, 10:13]

    # Isaac Gym root-state quaternions are ordered (qx, qy, qz, qw).
    quaternion = root_states[:, 3:7]
    qx = quaternion[:, 0]
    qy = quaternion[:, 1]
    qz = quaternion[:, 2]
    qw = quaternion[:, 3]

    # World-space projection of the torso's local forward (+x) axis.
    heading_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    heading_y = 2.0 * (qx * qy + qz * qw)
    heading_norm = torch.sqrt(
        heading_x * heading_x + heading_y * heading_y + 1.0e-6
    )
    heading_x = heading_x / heading_norm
    heading_y = heading_y / heading_norm

    signed_forward_speed = (
        planar_velocity[:, 0] * heading_x
        + planar_velocity[:, 1] * heading_y
    )
    forward_speed = torch.clamp(signed_forward_speed, min=0.0)
    backward_speed = torch.clamp(-signed_forward_speed, min=0.0)

    # Velocity perpendicular to the direction the torso is facing.
    lateral_speed = (
        planar_velocity[:, 0] * heading_y
        - planar_velocity[:, 1] * heading_x
    )

    # Positive when upright; independent of yaw.
    upright_score = torch.clamp(
        1.0 - 2.0 * (qx * qx + qy * qy), min=0.0, max=1.0
    )

    # Gating speed by uprightness prevents reward from exploiting falls.
    running_speed_reward = forward_speed * upright_score
    upright_reward = upright_score
    backward_velocity_penalty = backward_speed * backward_speed
    lateral_velocity_penalty = lateral_speed * lateral_speed
    angular_velocity_penalty = torch.mean(
        angular_velocity * angular_velocity, dim=1
    )
    action_penalty = torch.mean(actions * actions, dim=1)

    scaled_joint_velocity = dof_vel * dof_vel_scale
    joint_velocity_penalty = torch.mean(
        scaled_joint_velocity * scaled_joint_velocity, dim=1
    )

    speed_weight = 2.0
    upright_weight = 0.25
    backward_penalty_weight = 0.5
    lateral_penalty_weight = 0.05
    angular_penalty_weight = 0.02
    action_penalty_weight = 0.01
    joint_velocity_penalty_weight = 0.002

    speed_component = speed_weight * running_speed_reward
    upright_component = upright_weight * upright_reward
    backward_component = -backward_penalty_weight * backward_velocity_penalty
    lateral_component = -lateral_penalty_weight * lateral_velocity_penalty
    angular_component = -angular_penalty_weight * angular_velocity_penalty
    action_component = -action_penalty_weight * action_penalty
    joint_velocity_component = (
        -joint_velocity_penalty_weight * joint_velocity_penalty
    )

    total_reward = (
        speed_component
        + upright_component
        + backward_component
        + lateral_component
        + angular_component
        + action_component
        + joint_velocity_component
    )

    components: Dict[str, torch.Tensor] = {
        "forward_speed": speed_component,
        "upright": upright_component,
        "backward_velocity": backward_component,
        "lateral_velocity": lateral_component,
        "angular_velocity": angular_component,
        "action_magnitude": action_component,
        "joint_velocity": joint_velocity_component,
    }

    return total_reward, components
