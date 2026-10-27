@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ---------------- hyper-parameters (named constants, not inputs) ----------------
    speed_weight = 1.0        # weight of the linear forward-speed reward
    shaping_weight = 0.5      # weight of the saturating speed shaping
    shaping_temp = 1.0        # temperature (m/s) of the speed shaping
    body_speed_weight = 0.25  # weight of the body-frame forward speed bonus
    upright_weight = 0.5      # weight of the "stay upright" reward
    height_weight = 0.3       # weight of the torso-height preference
    height_target = 0.5       # preferred torso height (m)
    height_temp = 0.3         # tolerance (m) of the height preference
    ctrl_coef = 0.01          # action (energy) penalty coefficient
    dofvel_coef = 1.0e-4      # joint-velocity penalty coefficient
    alive_bonus = 0.1         # constant bonus for staying alive

    # ---------------- state extraction ----------------
    lin_vel = root_states[:, 7:10]      # world-frame linear velocity
    quat = root_states[:, 3:7]           # torso orientation (w, x, y, z)
    height = root_states[:, 2]           # world-frame torso height

    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]

    vel_x = lin_vel[:, 0]                # forward (world +x) velocity
    vel_y = lin_vel[:, 1]
    fwd_vel = torch.clamp(vel_x, min=0.0)   # never reward running backwards

    # world-frame z component of the torso up axis (local +z assumed up)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright = torch.clamp(up_z, min=0.0)

    # world-frame forward axis of the torso (local +x assumed forward)
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_norm = torch.sqrt(fwd_x * fwd_x + fwd_y * fwd_y + 1.0e-6)
    body_speed = torch.clamp((vel_x * fwd_x + vel_y * fwd_y) / fwd_norm, min=0.0)

    # ---------------- reward components ----------------
    forward_speed = speed_weight * fwd_vel
    speed_shaping = shaping_weight * torch.tanh(fwd_vel / shaping_temp)
    body_forward = body_speed_weight * torch.tanh(body_speed / shaping_temp)
    upright_reward = upright_weight * upright
    height_reward = height_weight * torch.exp(-((height - height_target) ** 2) / (height_temp ** 2))
    action_penalty = -ctrl_coef * torch.sum(actions * actions, dim=1)
    joint_vel_penalty = -dofvel_coef * torch.sum(dof_vel * dof_vel, dim=1)
    alive = alive_bonus * torch.ones_like(fwd_vel)

    reward = (forward_speed + speed_shaping + body_forward + upright_reward
              + height_reward + action_penalty + joint_vel_penalty + alive)

    components = {
        "forward_speed": forward_speed,
        "speed_shaping": speed_shaping,
        "body_forward": body_forward,
        "upright": upright_reward,
        "height": height_reward,
        "action_penalty": action_penalty,
        "joint_vel_penalty": joint_vel_penalty,
        "alive": alive,
    }
    return reward, components
