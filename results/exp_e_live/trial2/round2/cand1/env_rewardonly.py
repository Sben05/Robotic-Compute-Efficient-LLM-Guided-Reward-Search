@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dof_vel: torch.Tensor
                   ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # ----------------------------- hyper-parameters -----------------------------
    speed_sat_temperature: float = 6.0      # m/s, soft-saturation scale of the world +x speed
    facing_temperature: float = 6.0         # m/s, soft-saturation scale of the facing-direction speed
    align_temperature: float = 0.5          # dimensionless, sharpness of the heading alignment term
    upright_temperature: float = 0.5        # dimensionless, sharpness of the torso-upright term
    lateral_temperature: float = 2.0        # m/s, scale of the lateral drift penalty
    excess_speed_threshold: float = 5.0     # m/s, speed above which the "go faster" bonus kicks in
    yaw_rate_threshold: float = 2.0         # rad/s, yaw rate above which spinning is penalized
    fall_height: float = 0.40               # m, torso height below which the ant is considered fallen

    linear_speed_weight: float = 1.0        # weight of the (unbounded) linear world +x speed
    speed_sat_weight: float = 1.0           # weight of the saturating world +x speed bonus
    excess_speed_weight: float = 0.6        # weight of the high-speed bonus
    facing_weight: float = 1.5              # weight of the facing-direction speed bonus
    align_weight: float = 1.0               # weight of the heading (+x) alignment bonus
    upright_weight: float = 0.6             # weight of the torso-upright bonus
    alive_weight: float = 0.2               # small constant "stay alive" bonus
    lateral_weight: float = 0.6             # weight of the lateral drift penalty
    action_cost_scale: float = 0.005        # weight of the action magnitude penalty
    joint_vel_cost_scale: float = 0.001     # weight of the joint velocity penalty
    yaw_rate_weight: float = 0.2            # weight of the excess yaw-rate (spinning) penalty
    fall_weight: float = 5.0                # weight of the fall penalty

    n_envs = root_states.shape[0]
    device = root_states.device
    dtype = root_states.dtype

    # ----------------------------- state extraction -----------------------------
    quat = root_states[:, 3:7]        # Isaac Gym quaternion (x, y, z, w)
    vel = root_states[:, 7:10]        # world-frame linear velocity of the torso
    ang_vel = root_states[:, 10:13]   # world-frame angular velocity of the torso
    pos_z = root_states[:, 2]          # world-frame height of the torso

    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    # body axes in world frame
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)   # body forward (local +x) -> world x
    fwd_y = 2.0 * (qx * qy + qw * qz)          # body forward             -> world y
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)     # body up (local +z)      -> world z

    # horizontal unit vector of the body forward axis
    horiz = torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + 1.0e-8)
    fwd_hx = fwd_x / horiz
    fwd_hy = fwd_y / horiz

    vx = vel[:, 0]
    vy = vel[:, 1]
    yaw_rate = torch.abs(ang_vel[:, 2])

    # speed along the direction the ant is actually facing (horizontal plane)
    facing_speed = vx * fwd_hx + vy * fwd_hy

    # ----------------------------- reward components ----------------------------
    # 1) primary objective: forward (+x) speed, with a linear term that never saturates
    speed_linear = linear_speed_weight * vx
    # 2) bounded speed bonus (wide temperature so it still discriminates at high speed)
    speed_sat = speed_sat_weight * torch.tanh(vx / speed_sat_temperature)
    # 3) explicit bonus for exceeding the previously observed plateau -> keep pushing faster
    excess_speed = excess_speed_weight * torch.clamp(vx - excess_speed_threshold, min=0.0)
    # 4) reward motion along the facing direction (prevents backwards/back-pedal tricks)
    facing_term = facing_weight * torch.tanh(facing_speed / facing_temperature)
    # 5) sharp bonus for pointing the body along +x (fixes the ~25 deg heading offset)
    align_term = align_weight * torch.tanh(fwd_hx / align_temperature)
    # 6) keep the torso up
    upright_term = upright_weight * torch.tanh(up_z / upright_temperature)
    # 7) small constant survival bonus
    alive_term = alive_weight * torch.ones(n_envs, device=device, dtype=dtype)
    # 8) penalize sideways drift (wasted effort)
    lateral_term = -lateral_weight * torch.tanh(vy / lateral_temperature)
    # 9) penalize excessive spinning
    spin_term = -yaw_rate_weight * torch.clamp(yaw_rate - yaw_rate_threshold, min=0.0)
    # 10) regularization: action magnitude and joint velocities
    action_term = -action_cost_scale * torch.sum(actions * actions, dim=-1)
    joint_vel_term = -joint_vel_cost_scale * torch.mean(dof_vel * dof_vel, dim=-1)
    # 11) strong fall penalty (episode length was 950/1000 -> occasional falls)
    fall_amount = torch.clamp(fall_height - pos_z, min=0.0)
    fall_term = -fall_weight * fall_amount

    reward = (speed_linear + speed_sat + excess_speed + facing_term + align_term
              + upright_term + alive_term + lateral_term + spin_term
              + action_term + joint_vel_term + fall_term)

    components: Dict[str, torch.Tensor] = {
        "forward_speed": speed_linear + speed_sat,
        "excess_speed": excess_speed,
        "facing_speed": facing_term,
        "heading_align": align_term,
        "upright": upright_term,
        "alive": alive_term,
        "lateral_drift": lateral_term,
        "spin_penalty": spin_term,
        "action_cost": action_term,
        "joint_vel_cost": joint_vel_term,
        "fall_penalty": fall_term,
    }

    return reward, components
