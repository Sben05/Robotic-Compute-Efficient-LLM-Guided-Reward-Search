@torch.jit.script
def compute_reward(
    ball_positions: torch.Tensor,
    ball_linvels: torch.Tensor,
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    BallBalance: keep the ball on the table top without letting it fall.

    ball_positions : [N, 3] world (or env-frame) position of the ball
    ball_linvels   : [N, 3] linear velocity of the ball
    dof_positions  : [N, D] joint positions (actuated joints are 1, 3, 5)
    dof_velocities : [N, D] joint velocities (actuated joints are 1, 3, 5)
    """

    # ------------------------------------------------------------------
    # Named hyper-parameters (internal to the reward, not inputs)
    # ------------------------------------------------------------------
    radial_temperature: float = 0.40     # m, softness of the "close to table centre" term
    height_temperature: float = 0.12     # m, softness of the "at table height" term
    floor_temperature: float = 0.03      # m, softness of the floor gate
    fall_reference_height: float = 0.08  # m, below this the ball lies on the floor
    speed_reference: float = 1.0         # m/s, normalisation of the ball speed

    weight_support: float = 1.0          # ball resting on the table top
    weight_centre: float = 0.3           # ball close to the table centre
    weight_fall: float = 2.0             # ball close to / on the floor
    weight_ball_speed: float = 0.2       # ball moving fast (will roll off)
    weight_fall_speed: float = 0.5       # ball moving downwards (falling)
    weight_effort: float = 0.01          # violent joint motion knocks the ball off
    weight_posture: float = 0.005        # stay close to the neutral (level) pose

    # ------------------------------------------------------------------
    # Ball kinematics
    # ------------------------------------------------------------------
    ball_x = ball_positions[:, 0]
    ball_y = ball_positions[:, 1]
    ball_z = ball_positions[:, 2]

    vel_x = ball_linvels[:, 0]
    vel_y = ball_linvels[:, 1]
    vel_z = ball_linvels[:, 2]

    # ------------------------------------------------------------------
    # Reference geometry (estimated, so the reward is not tied to a fixed table)
    # ------------------------------------------------------------------
    # Height of the table top: estimated from the batch, since the balls rest on it.
    table_height_ref = torch.mean(ball_positions[:, 2])
    # Table centre: assumed to be at the origin of the observation frame.
    dist_from_centre = torch.sqrt(ball_x * ball_x + ball_y * ball_y)

    # ------------------------------------------------------------------
    # Positive terms
    # ------------------------------------------------------------------
    # (a) the ball is at the height of the table top -> it is supported by it
    support_score = torch.exp(-torch.pow((ball_z - table_height_ref) / height_temperature, 2.0))
    # a ball that already reached the floor is never supported by the table
    floor_gate = torch.sigmoid((ball_z - fall_reference_height) / floor_temperature)
    support_score = support_score * floor_gate

    # (b) the ball is close to the centre of the table (stays away from the edges)
    centre_score = torch.exp(-torch.pow(dist_from_centre / radial_temperature, 2.0))

    # ------------------------------------------------------------------
    # Negative terms
    # ------------------------------------------------------------------
    fall_penalty = torch.clamp(fall_reference_height - ball_z, min=0.0)

    speed_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z
    speed_cost = speed_sq / (speed_reference * speed_reference)

    fall_speed = torch.clamp(-vel_z, min=0.0) / speed_reference

    effort = (
        torch.pow(dof_velocities[:, 1], 2.0)
        + torch.pow(dof_velocities[:, 3], 2.0)
        + torch.pow(dof_velocities[:, 5], 2.0)
    )
    posture = (
        torch.pow(dof_positions[:, 1], 2.0)
        + torch.pow(dof_positions[:, 3], 2.0)
        + torch.pow(dof_positions[:, 5], 2.0)
    )

    # ------------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------------
    reward = (
        weight_support * support_score
        + weight_centre * centre_score
        - weight_fall * fall_penalty
        - weight_ball_speed * speed_cost
        - weight_fall_speed * fall_speed
        - weight_effort * effort
        - weight_posture * posture
    )

    components: Dict[str, torch.Tensor] = {
        "support": weight_support * support_score,
        "centre": weight_centre * centre_score,
        "fall": -weight_fall * fall_penalty,
        "ball_speed": -weight_ball_speed * speed_cost,
        "fall_speed": -weight_fall_speed * fall_speed,
        "effort": -weight_effort * effort,
        "posture": -weight_posture * posture,
    }

    return reward, components
