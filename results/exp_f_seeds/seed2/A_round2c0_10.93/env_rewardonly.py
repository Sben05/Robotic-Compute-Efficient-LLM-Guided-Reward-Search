@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   actions: torch.Tensor,
                   dof_pos: torch.Tensor,
                   dof_limits_lower: torch.Tensor,
                   dof_limits_upper: torch.Tensor,
                   up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ant task: run forward (+x world frame) as fast as possible.

    Changes w.r.t. the previous reward:
      * forward speed shaping is much less saturated (temperature 4 -> 10 m/s)
      * an extra, incremental "faster than 5 m/s" bonus provides a constant
        gradient in the 5-14 m/s range (breaks the previous ~6 m/s plateau)
      * upright term is now tilt-sensitive instead of pinned at its maximum,
        plus a separate penalty that only fires when the ant actually tips over
      * control / lateral / roll-pitch weights reduced (they were limiting the
        aggressive gait without providing shaping)
      * hard (always zero) joint-limit term replaced by an active soft one
    """

    # ---------------- named shaping scales / temperatures ----------------
    forward_speed_temperature = 10.0   # m/s, tanh scale of the signed speed term
    high_speed_threshold = 5.0         # m/s, speed where the extra bonus starts
    high_speed_cap = 14.0              # m/s, speed where the extra bonus saturates
    upright_temperature = 0.5          # up_z range mapped to [0, 1]
    tilt_threshold = 0.4               # up_z below this counts as tipped over

    # ---------------- weights ----------------
    forward_weight = 1.0
    high_speed_weight = 0.30
    upright_weight = 0.15
    tilt_penalty_weight = 1.0
    control_weight = 0.005
    lateral_velocity_weight = 0.02
    roll_pitch_angular_velocity_weight = 0.002
    soft_joint_limit_weight = 0.05

    # ---------------- state extraction ----------------
    lin_vel = root_states[:, 7:10]     # world-frame linear velocity
    ang_vel = root_states[:, 10:13]    # world-frame angular velocity
    forward_vel = lin_vel[:, 0]        # +x running speed (m/s)
    lateral_vel = lin_vel[:, 1]        # sideways drift (m/s)
    up_z = up_vec[:, 2]                # world-z component of torso up axis

    # ---------------- reward components ----------------
    # Signed forward speed, weakly saturating: still negative when moving
    # backwards, but keeps a useful gradient at high speeds.
    forward_reward = forward_weight * torch.tanh(forward_vel / forward_speed_temperature)

    # Incremental bonus that only switches on above the speed the policy
    # currently plateaus at -> constant pressure towards a faster gait.
    high_speed = torch.clamp(forward_vel - high_speed_threshold, min=0.0,
                              max=high_speed_cap - high_speed_threshold)
    high_speed_reward = high_speed_weight * high_speed / (high_speed_cap - high_speed_threshold)

    # Upright posture: responsive to leaning instead of being pinned at max.
    upright_reward = upright_weight * torch.clamp(
        (up_z - (1.0 - upright_temperature)) / upright_temperature, min=0.0, max=1.0)

    # Strong penalty only when the ant really tips over (guards against
    # "dive and tumble" exploits of the speed reward).
    tilt_cost = tilt_penalty_weight * torch.relu(tilt_threshold - up_z) ** 2

    # Control effort (mild regularizer).
    control_cost = control_weight * torch.sum(actions * actions, dim=-1)

    # Sideways drift.
    lateral_velocity_cost = lateral_velocity_weight * torch.abs(lateral_vel)

    # Excessive roll/pitch rates (tumbling), very mild so fast gaits are allowed.
    roll_pitch_angular_velocity_cost = roll_pitch_angular_velocity_weight * (
        ang_vel[:, 0] * ang_vel[:, 0] + ang_vel[:, 1] * ang_vel[:, 1])

    # Always-active soft joint-limit penalty on normalized joint positions.
    dof_range = torch.clamp(dof_limits_upper - dof_limits_lower, min=1e-6)
    scaled_dof_pos = 2.0 * (dof_pos - dof_limits_lower) / dof_range - 1.0
    soft_joint_limit_cost = soft_joint_limit_weight * torch.sum(
        torch.relu(torch.abs(scaled_dof_pos) - 0.9) ** 2, dim=-1)

    # ---------------- total reward ----------------
    total_reward = (forward_reward
                    + high_speed_reward
                    + upright_reward
                    - tilt_cost
                    - control_cost
                    - lateral_velocity_cost
                    - roll_pitch_angular_velocity_cost
                    - soft_joint_limit_cost)

    reward_components: Dict[str, torch.Tensor] = {}
    reward_components["forward_speed"] = forward_reward
    reward_components["high_speed_bonus"] = high_speed_reward
    reward_components["upright_posture"] = upright_reward
    reward_components["tilt_cost"] = -tilt_cost
    reward_components["control_cost"] = -control_cost
    reward_components["lateral_velocity_cost"] = -lateral_velocity_cost
    reward_components["roll_pitch_angular_velocity_cost"] = -roll_pitch_angular_velocity_cost
    reward_components["soft_joint_limit_cost"] = -soft_joint_limit_cost

    return total_reward, reward_components
