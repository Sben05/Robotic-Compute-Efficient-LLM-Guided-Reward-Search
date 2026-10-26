@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   actions: torch.Tensor,
                   dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward for making the ant run forward (+x world frame) as fast as possible.

    root_states : (num_envs, 13) -> [pos(3), quat(w,x,y,z)(4), lin_vel(3), ang_vel(3)]
    actions     : (num_envs, num_actions) last applied joint torques
    dof_vel     : (num_envs, num_dofs) joint velocities

    The main term is a saturating reward on the forward (+x) linear velocity.
    Small regularization terms (control effort, body tilt, vertical bouncing,
    body spinning, joint speed) discourage degenerate "fast but unstable"
    solutions (e.g. flipping over and sliding) that make the gait unrepeatable.
    """

    # ---------------- named local constants ----------------
    speed_temperature = 2.0        # m/s, scale of the tanh speed reward
    control_weight = 0.05          # mean squared action penalty
    tilt_weight = 0.10             # penalty for leaning/rolling over
    backward_heading_weight = 0.05 # penalty for pointing the body away from +x
    vertical_velocity_weight = 0.02# penalty for bouncing up/down
    angular_velocity_weight = 0.02 # penalty for spinning/rocking the torso
    dof_velocity_weight = 0.005    # penalty for excessive joint speed

    # ---------------- state decomposition ----------------
    lin_vel = root_states[:, 7:10]
    ang_vel = root_states[:, 10:13]
    quat = root_states[:, 3:7]

    # ---------------- forward speed reward ----------------
    forward_vel = lin_vel[:, 0]
    forward_vel = torch.where(torch.isfinite(forward_vel),
                              forward_vel,
                              torch.zeros_like(forward_vel))
    # smooth, saturating reward: ~0.76 at 2 m/s, ~0.96 at 4 m/s, negative when running backwards
    forward_reward = torch.tanh(forward_vel / speed_temperature)

    # ---------------- body orientation ----------------
    quat_norm = quat / (torch.sqrt(torch.sum(quat * quat, dim=-1, keepdim=True)) + 1.0e-8)
    qw = quat_norm[:, 0]
    qx = quat_norm[:, 1]
    qy = quat_norm[:, 2]
    qz = quat_norm[:, 3]

    # world-space z component of the torso "up" axis (1 when perfectly upright)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    tilt = torch.clamp(1.0 - up_z, min=0.0)

    # world-space x component of the torso forward axis (negative => facing backwards)
    forward_axis_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    backward_heading = torch.clamp(-forward_axis_x, min=0.0)

    # ---------------- regularization costs ----------------
    vertical_vel = lin_vel[:, 2]
    control_cost = torch.mean(actions * actions, dim=-1)
    angular_velocity_cost = torch.mean(ang_vel * ang_vel, dim=-1)
    dof_velocity_cost = torch.mean(dof_vel * dof_vel, dim=-1)

    # ---------------- signed reward components ----------------
    c_forward = forward_reward
    c_control = -control_weight * control_cost
    c_tilt = -tilt_weight * tilt
    c_backward_heading = -backward_heading_weight * backward_heading
    c_vertical = -vertical_velocity_weight * vertical_vel * vertical_vel
    c_angular = -angular_velocity_weight * angular_velocity_cost
    c_dof_vel = -dof_velocity_weight * dof_velocity_cost

    reward = (c_forward + c_control + c_tilt + c_backward_heading +
              c_vertical + c_angular + c_dof_vel)

    components: Dict[str, torch.Tensor] = {
        "forward_speed": c_forward,
        "control_cost": c_control,
        "tilt_cost": c_tilt,
        "backward_heading_cost": c_backward_heading,
        "vertical_velocity_cost": c_vertical,
        "angular_velocity_cost": c_angular,
        "dof_velocity_cost": c_dof_vel,
    }
    return reward, components
