@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    actions: torch.Tensor,
    up_vec: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward for making the ant run forward (+x world frame) as fast as possible.

    root_states : [pos(3), rot(4), lin_vel(3), ang_vel(3)] per environment
    actions     : last applied action vector per environment
    up_vec      : world-frame direction of the torso's up axis (used for a
                  mild "stay upright" penalty).
    """
    # ---------------- named hyper-parameters ----------------
    forward_speed_coefficient: float = 1.0   # main dense speed reward
    speed_bonus_coefficient: float = 0.25    # extra (saturating) fast-speed bonus
    speed_temperature: float = 2.0           # temperature of the exp() speed shaping
    lateral_drift_coefficient: float = 0.05  # discourage sideways drifting
    control_cost_coefficient: float = 0.001  # small effort/smoothness penalty
    posture_cost_coefficient: float = 0.2    # discourage tipping over
    alive_bonus: float = 0.1                 # reward for remaining upright/alive

    # ---------------- forward running speed ----------------
    lin_vel = root_states[:, 7:10]           # world-frame linear velocity
    forward_vel = lin_vel[:, 0]               # +x is the "forward" direction
    lateral_vel = lin_vel[:, 1]

    forward_speed_reward = forward_speed_coefficient * forward_vel

    # Saturating bonus so that very fast running is still preferred, while
    # keeping the reward numerically bounded for extreme speeds.
    clamped_forward_vel = torch.clamp(forward_vel, min=-10.0, max=20.0)
    speed_bonus = speed_bonus_coefficient * (
        torch.exp(clamped_forward_vel / speed_temperature) - 1.0
    )

    # ---------------- regularizers ----------------
    lateral_drift_cost = -lateral_drift_coefficient * torch.abs(lateral_vel)

    control_cost = -control_cost_coefficient * torch.sum(torch.square(actions), dim=-1)

    # up_vec[:, 2] == cos(tilt angle): 1.0 when perfectly upright.
    up_z = torch.clamp(up_vec[:, 2], min=-1.0, max=1.0)
    posture_cost = -posture_cost_coefficient * torch.clamp(1.0 - up_z, min=0.0, max=2.0)

    alive_reward = torch.full_like(forward_vel, alive_bonus)

    # ---------------- total ----------------
    total_reward = (
        forward_speed_reward
        + speed_bonus
        + lateral_drift_cost
        + control_cost
        + posture_cost
        + alive_reward
    )

    components: Dict[str, torch.Tensor] = {
        "forward_speed_reward": forward_speed_reward,
        "speed_bonus": speed_bonus,
        "lateral_drift_cost": lateral_drift_cost,
        "control_cost": control_cost,
        "posture_cost": posture_cost,
        "alive_bonus": alive_reward,
    }

    return total_reward, components
