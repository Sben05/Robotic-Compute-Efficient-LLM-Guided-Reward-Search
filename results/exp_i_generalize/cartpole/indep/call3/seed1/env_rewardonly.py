@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward for balancing a pole on a cart.

    Assumptions (matching the observation layout):
      dof_pos[:, 0] : cart position       dof_vel[:, 0] : cart linear velocity
      dof_pos[:, 1] : pole joint angle    dof_vel[:, 1] : pole angular velocity
    A pole angle of 0 rad corresponds to the upright configuration.

    The reward is dense and shaped:
      + smooth upright bonus (peaks at pole_angle = 0)
      - |pole angle| penalty (strong gradient back toward vertical)
      - cart position / velocity penalties (keeps the cart near the center and still)
      - pole angular velocity penalty (discovers damping)
    """

    # ---------------- hyper-parameters ----------------
    angle_temperature: float = 0.25   # rad^2, sharpness of the upright bonus
    pole_angle_weight: float = 0.5    # weight on |pole angle|
    cart_pos_weight: float = 0.1      # weight on cart position^2
    cart_vel_weight: float = 0.05     # weight on cart velocity^2
    pole_vel_weight: float = 0.02     # weight on pole angular velocity^2

    # ---------------- state extraction ----------------
    # .squeeze(-1) makes the code work with [N, D] or [N, D, 1] DOF tensors.
    cart_pos = dof_pos[:, 0].squeeze(-1)
    pole_angle = dof_pos[:, 1].squeeze(-1)
    cart_vel = dof_vel[:, 0].squeeze(-1)
    pole_vel = dof_vel[:, 1].squeeze(-1)

    # ---------------- reward components ----------------
    angle_error_sq = torch.square(pole_angle)
    upright_bonus = torch.exp(-angle_error_sq / angle_temperature)

    pole_angle_penalty = pole_angle_weight * torch.abs(pole_angle)
    cart_position_penalty = cart_pos_weight * torch.square(cart_pos)
    cart_velocity_penalty = cart_vel_weight * torch.square(cart_vel)
    pole_velocity_penalty = pole_vel_weight * torch.square(pole_vel)

    total_reward = (
        upright_bonus
        - pole_angle_penalty
        - cart_position_penalty
        - cart_velocity_penalty
        - pole_velocity_penalty
    )

    components: Dict[str, torch.Tensor] = {
        "upright_bonus": upright_bonus,
        "pole_angle_penalty": -pole_angle_penalty,
        "cart_position_penalty": -cart_position_penalty,
        "cart_velocity_penalty": -cart_velocity_penalty,
        "pole_velocity_penalty": -pole_velocity_penalty,
    }

    return total_reward, components
