@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                     up_vec: torch.Tensor,
                     actions: torch.Tensor,
                     dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward 4 - body-frame running speed with exponential shaping / normalisation.

      * the torso forward axis (body +x) is recovered from the root quaternion, so the
        reward measures true "running in the direction the ant faces"
      * speed term 1 - exp(-v / T) saturating speed reward (T = speed_temperature)
      * speed term 2 - distance covered along the body forward axis during the step
      * alive bonus (torso not lying on the ground) and upright bonus
      * control effort penalty
      * the whole reward is finally passed through exp(.) so it is bounded in (0, e^..]
        while staying monotone in the running speed (T = reward_temperature)
    """
    speed_temperature = 1.5
    reward_temperature = 1.0

    # body forward axis expressed in the world frame (quaternion xyzw)
    q = root_states[:, 3:7]
    qw = q[:, 3]
    qx = q[:, 0]
    qy = q[:, 1]
    qz = q[:, 2]
    f_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    f_y = 2.0 * (qw * qz + qx * qy)
    f_z = 2.0 * (qx * qz - qw * qy)

    vel = root_states[:, 7:10]
    forward_speed = f_x * vel[:, 0] + f_y * vel[:, 1] + f_z * vel[:, 2]

    # saturating speed reward
    speed_reward = 1.0 - torch.exp(-forward_speed / speed_temperature)
    # distance travelled along the body forward axis during this control step
    distance_reward = forward_speed * dt

    # upright bonus
    upright_reward = torch.clamp(up_vec[:, 2], min=0.0, max=1.0)
    # alive bonus: torso kept above the ground
    alive_reward = torch.clamp(root_states[:, 2] - 0.2, min=0.0, max=1.0)
    # control effort
    energy_penalty = -0.002 * torch.sum(torch.square(actions), dim=-1)

    raw_reward = (2.0 * speed_reward + 1.0 * distance_reward
                  + 0.5 * upright_reward + 0.5 * alive_reward + energy_penalty)
    total = torch.exp(raw_reward / reward_temperature)

    components: Dict[str, torch.Tensor] = {
        "forward_speed": 2.0 * speed_reward,
        "forward_distance": distance_reward,
        "upright": 0.5 * upright_reward,
        "alive": 0.5 * alive_reward,
        "energy": energy_penalty,
        "shaped_reward": total,
    }
    return total, components
