@torch.jit.script
def compute_reward(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # This reward function aims to make the Ant run forward as fast as possible.
    # It primarily rewards forward linear velocity while also applying penalties
    # for undesirable movements (sideways/vertical velocity) and for not maintaining
    # a stable height, which are crucial for effective locomotion.

    # Extract components from obs_buf based on the compute_ant_observations function.
    # The `obs_buf` is structured as follows:
    # Index 0: torso_height (torso_position[:, up_axis_idx])
    # Index 1: vel_loc_x (forward linear velocity in local frame)
    # Index 2: vel_loc_y (sideways linear velocity in local frame)
    # Index 3: vel_loc_z (vertical linear velocity in local frame)
    # ... other observation components follow ...

    # 1. Reward for running forward
    # We want to maximize the forward velocity component in the agent's local frame.
    forward_velocity = obs_buf[:, 1]  # vel_loc_x
    forward_vel_reward_temp = 1.0  # Temperature parameter for scaling the forward velocity reward
    reward_forward_vel = forward_velocity * forward_vel_reward_temp

    # 2. Penalty for sideways velocity
    # To encourage straight running, penalize any significant sideways movement.
    vel_loc_y = obs_buf[:, 2]  # sideways linear velocity
    sideways_vel_cost_temp = 0.5  # Temperature parameter for scaling the sideways velocity penalty
    reward_sideways_vel = -torch.square(vel_loc_y) * sideways_vel_cost_temp

    # 3. Penalty for vertical velocity
    # To encourage stable, horizontal locomotion and discourage excessive jumping or sinking.
    vel_loc_z = obs_buf[:, 3]  # vertical linear velocity
    vertical_vel_cost_temp = 0.2  # Temperature parameter for scaling the vertical velocity penalty
    reward_vertical_vel = -torch.square(vel_loc_z) * vertical_vel_cost_temp

    # 4. Reward for maintaining torso height
    # A stable height is critical for locomotion. We reward staying close to a target height.
    torso_height = obs_buf[:, 0]
    torso_height_target = 0.8  # A reasonable target height for the Ant
    torso_height_cost_temp = 2.0  # Temperature parameter for scaling the height deviation penalty
    # Use an exponential decay for height reward: max at target, rapidly decreases with deviation.
    reward_torso_height = torch.exp(-torso_height_cost_temp * torch.square(torso_height - torso_height_target))

    # Calculate total reward
    total_reward = reward_forward_vel + reward_sideways_vel + reward_vertical_vel + reward_torso_height

    # Store individual reward components for introspection and debugging
    reward_components = {
        "forward_vel_reward": reward_forward_vel,
        "sideways_vel_cost": reward_sideways_vel,
        "vertical_vel_cost": reward_vertical_vel,
        "torso_height_reward": reward_torso_height,
    }

    return total_reward, reward_components
