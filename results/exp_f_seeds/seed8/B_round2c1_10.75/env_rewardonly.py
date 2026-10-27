@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   actions: torch.Tensor,
                   up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward the Ant for running forward (+x world frame) as fast as possible,
    while staying upright, alive and reasonably straight.

    Root-state layout: [pos(0:3), quat(3:7), lin_vel(7:10), ang_vel(10:13)].
    """

    # ---------------- named parameters (temperatures / weights) ----------------
    # Temperature of the saturating speed term (m/s). Larger than before so the
    # term does not saturate at ~6 m/s and keeps a useful gradient at high speed.
    forward_speed_temperature = 6.0
    forward_tanh_weight = 0.6

    # Linear (clipped) speed term: gives continuous pressure to run faster.
    forward_linear_weight = 0.2
    forward_speed_max = 15.0     # clip upper end (m/s) to avoid runaway reward
    backward_speed_min = -5.0    # clip lower end (m/s)

    # Small per-step survival bonus to protect long episodes.
    alive_bonus_weight = 0.1

    # Tilt penalty (replaces the saturated upright bonus).
    tilt_weight = 0.5

    # Regularizers (all intentionally small so they do not cap the top speed).
    control_weight = 0.01
    lateral_velocity_weight = 0.02
    roll_pitch_angular_velocity_weight = 0.002

    # ---------------- state extraction ----------------
    lin_vel = root_states[:, 7:10]      # world-frame linear velocity
    ang_vel = root_states[:, 10:13]     # world-frame angular velocity
    forward_vel = lin_vel[:, 0]         # forward (+x) speed
    lateral_vel = lin_vel[:, 1]         # sideways speed

    # ---------------- forward speed ----------------
    forward_speed_tanh = forward_tanh_weight * torch.tanh(forward_vel / forward_speed_temperature)

    forward_vel_clipped = torch.clamp(forward_vel, min=backward_speed_min, max=forward_speed_max)
    forward_speed_linear = forward_linear_weight * forward_vel_clipped

    # ---------------- survival ----------------
    alive_bonus = alive_bonus_weight * torch.ones_like(forward_vel)

    # ---------------- posture (tilt) ----------------
    up_z = up_vec[:, 2]                              # world-z of torso up axis
    tilt_deviation = torch.clamp(1.0 - up_z, min=0.0)
    tilt_cost = tilt_weight * tilt_deviation * tilt_deviation

    # ---------------- regularizers ----------------
    control_cost = control_weight * torch.sum(actions * actions, dim=-1)

    lateral_velocity_cost = lateral_velocity_weight * torch.abs(lateral_vel)

    roll_pitch_angular_velocity_cost = roll_pitch_angular_velocity_weight * (
        ang_vel[:, 0] * ang_vel[:, 0] + ang_vel[:, 1] * ang_vel[:, 1])

    # ---------------- total reward ----------------
    total_reward = (forward_speed_tanh
                    + forward_speed_linear
                    + alive_bonus
                    - tilt_cost
                    - control_cost
                    - lateral_velocity_cost
                    - roll_pitch_angular_velocity_cost)

    reward_components: Dict[str, torch.Tensor] = {}
    reward_components["forward_speed_tanh"] = forward_speed_tanh
    reward_components["forward_speed_linear"] = forward_speed_linear
    reward_components["alive_bonus"] = alive_bonus
    reward_components["tilt_cost"] = -tilt_cost
    reward_components["control_cost"] = -control_cost
    reward_components["lateral_velocity_cost"] = -lateral_velocity_cost
    reward_components["roll_pitch_angular_velocity_cost"] = -roll_pitch_angular_velocity_cost

    return total_reward, reward_components
