@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                     up_vec: torch.Tensor,
                     actions: torch.Tensor,
                     dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward 1 - dense "run forward" reward.

      * linear reward on the forward (world +x) linear velocity of the torso
      * bonus for keeping the torso upright (up_vec[:, 2] ~ cos(tilt))
      * bonus for keeping the torso at a healthy height (nominal ~0.5-0.8 m)
      * small penalty on control effort and joint speeds for smooth gaits

    root_states layout (Isaac Gym): pos[0:3], quat[3:7] (xyzw), lin_vel[7:10], ang_vel[10:13]
    """
    # forward running speed (m/s)
    forward_speed = root_states[:, 7]
    speed_reward = torch.clamp(forward_speed, min=-1.0, max=10.0)

    # stay upright
    upright_reward = torch.clamp(up_vec[:, 2], min=0.0, max=1.0)

    # keep the torso lifted off the ground
    height_reward = torch.clamp(root_states[:, 2] - 0.25, min=0.0, max=1.0)

    # control / joint velocity energy
    action_energy = torch.sum(torch.square(actions), dim=-1)
    joint_energy = torch.sum(torch.square(dof_vel), dim=-1)
    energy_penalty = -0.001 * action_energy - 0.00001 * joint_energy

    total = speed_reward + 0.5 * upright_reward + 0.5 * height_reward + energy_penalty

    components: Dict[str, torch.Tensor] = {
        "forward_speed": speed_reward,
        "upright": upright_reward,
        "torso_height": height_reward,
        "energy": energy_penalty,
    }
    return total, components
