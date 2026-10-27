@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dof_vel: torch.Tensor
                   ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # ------------------------------ hyper-parameters ------------------------------
    speed_cap: float = 12.0           # m/s, normalized speed term saturates here
    linear_speed_cap: float = 15.0    # m/s, hard cap of the raw linear speed bonus
    speed_temperature: float = 4.0    # m/s, soft saturation of the speed bonus
    quality_temperature: float = 6.0  # m/s, speed scale of the (upright x fast) term
    sprint_threshold: float = 5.0     # m/s, speed above which the sprint bonus starts
    sprint_cap: float = 12.0          # m/s, speed at which the sprint bonus is full
    lateral_temperature: float = 1.0  # m/s, scale of the lateral drift penalty
    yaw_temperature: float = 1.0      # rad/s, scale of the yaw-rate penalty

    speed_weight: float = 3.0         # normalized clipped-linear speed (main driver)
    linear_speed_weight: float = 0.5  # raw linear speed (keeps pushing for max speed)
    bonus_weight: float = 0.5         # tanh speed bonus
    quality_weight: float = 1.0       # uprightness x speed (couples posture to speed)
    sprint_weight: float = 1.0        # extra bonus only above sprint_threshold
    align_weight: float = 0.3         # heading alignment with +x
    upright_weight: float = 0.3       # torso uprightness
    lateral_weight: float = 0.1       # lateral drift penalty
    yaw_weight: float = 0.02          # spinning penalty
    action_cost_scale: float = 0.005
    joint_vel_cost_scale: float = 0.0005
    fall_weight: float = 1.0
    fall_height: float = 0.30         # m, torso height below which the ant has collapsed

    # ------------------------------ state extraction ------------------------------
    quat = root_states[:, 3:7]        # (x, y, z, w)
    vel = root_states[:, 7:10]        # world-frame linear velocity
    ang_vel = root_states[:, 10:13]   # world-frame angular velocity
    pos_z = root_states[:, 2]         # torso height

    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    # torso axes expressed in the world frame
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)   # body +x, world x
    fwd_y = 2.0 * (qx * qy + qw * qz)          # body +x, world y
    up_zz = 1.0 - 2.0 * (qx * qx + qy * qy)    # body +z, world z (uprightness)

    horiz = torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + 1.0e-8)
    fwd_hx = fwd_x / horiz                      # heading cosine with world +x

    vx = vel[:, 0]
    vy = vel[:, 1]
    wz = ang_vel[:, 2]

    # ------------------------------ reward components -----------------------------
    # 1) main objective: forward (+x) speed, clipped-linear -> constant gradient
    speed_norm = torch.clamp(vx / speed_cap, min=-0.5, max=1.0)
    forward_speed = speed_weight * speed_norm

    # 2) raw linear speed bonus (capped to stay bounded)
    linear_speed = linear_speed_weight * torch.clamp(vx, max=linear_speed_cap)

    # 3) soft saturating speed bonus
    speed_bonus = bonus_weight * torch.tanh(vx / speed_temperature)

    # 4) run quality: upright AND fast (multiplicative coupling, no dead gradient)
    upright_val = torch.clamp(up_zz, min=0.0, max=1.0)
    run_quality = quality_weight * upright_val * torch.tanh(vx / quality_temperature)

    # 5) sprint bonus: extra incentive once the ant is already fast
    sprint = sprint_weight * torch.clamp(
        (vx - sprint_threshold) / (sprint_cap - sprint_threshold), min=0.0, max=1.0)

    # 6) heading alignment with world +x
    heading_align = align_weight * torch.clamp(fwd_hx, min=0.0, max=1.0)

    # 7) torso uprightness
    upright = upright_weight * upright_val

    # 8) penalties
    lateral_drift = -lateral_weight * torch.tanh(vy / lateral_temperature)
    yaw_rate = -yaw_weight * torch.tanh(wz / yaw_temperature)
    action_cost = -action_cost_scale * torch.sum(actions * actions, dim=-1)
    joint_vel_cost = -joint_vel_cost_scale * torch.mean(dof_vel * dof_vel, dim=-1)
    fall_penalty = -fall_weight * torch.clamp(fall_height - pos_z, min=0.0)

    # ------------------------------ total reward ----------------------------------
    reward = (forward_speed + linear_speed + speed_bonus + run_quality + sprint
              + heading_align + upright + lateral_drift + yaw_rate
              + action_cost + joint_vel_cost + fall_penalty)

    components: Dict[str, torch.Tensor] = {
        "forward_speed": forward_speed,
        "linear_speed": linear_speed,
        "speed_bonus": speed_bonus,
        "run_quality": run_quality,
        "sprint_bonus": sprint,
        "heading_align": heading_align,
        "upright": upright,
        "lateral_drift": lateral_drift,
        "yaw_rate": yaw_rate,
        "action_cost": action_cost,
        "joint_vel_cost": joint_vel_cost,
        "fall_penalty": fall_penalty,
    }

    return reward, components
