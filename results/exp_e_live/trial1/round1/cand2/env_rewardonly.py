@torch.jit.script
def compute_reward(obs_buf: torch.Tensor,
                   root_states: torch.Tensor,
                   actions: torch.Tensor,
                   potentials: torch.Tensor,
                   prev_potentials: torch.Tensor
                   ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # ------------------------------------------------------------------
    # hyper parameters
    # ------------------------------------------------------------------
    w_forward: float = 1.0        # forward running speed   [m/s]
    w_world_x: float = 0.25       # progress along world +x [m/s]
    w_lateral: float = 0.10       # sideways drift penalty
    w_yaw: float = 0.05           # yaw-rate penalty
    w_vertical: float = 0.02      # vertical bounce penalty
    w_upright: float = 0.50       # tipping-over penalty
    tilt_threshold: float = 0.60  # up_proj below this  (~53 deg) is penalised
    w_action: float = 0.005       # action regularisation
    w_alive: float = 0.05         # small survival bonus
    w_target: float = 0.20        # potential based shaping (closing speed)

    # ------------------------------------------------------------------
    # observation parsing (see compute_ant_observations for the layout)
    #   [ height(1), vel_body(3), angvel_body(3), yaw(1), roll(1),
    #     angle_to_target(1), up_proj(1), heading_proj(1), dof_pos(8),
    #     dof_vel(8), contact_forces(24), actions(8) ]
    # ------------------------------------------------------------------
    vel_body = obs_buf[:, 1:4]        # linear velocity in the torso frame
    ang_body = obs_buf[:, 4:7]        # angular velocity in the torso frame
    up_proj = obs_buf[:, 10]          # torso up-axis . world up-axis

    fwd_vel = vel_body[:, 0]          # torso +x : forward
    lat_vel = vel_body[:, 1]          # torso +y : sideways
    vert_vel = vel_body[:, 2]         # torso +z : vertical
    yaw_rate = ang_body[:, 2]         # torso +z : rotation about the up axis

    vel_world = root_states[:, 7:10]  # linear velocity in the world frame
    world_x_vel = vel_world[:, 0]

    # ------------------------------------------------------------------
    # reward components
    # ------------------------------------------------------------------
    # forward speed [m/s]; moving backwards is punished twice as hard
    forward_speed = fwd_vel - torch.relu(-fwd_vel)
    # progress along the global +x axis (keeps the run straight)
    world_x_speed = world_x_vel - torch.relu(-world_x_vel)

    lateral_pen = lat_vel * lat_vel
    yaw_pen = yaw_rate * yaw_rate
    vertical_pen = vert_vel * vert_vel
    tilt_pen = torch.relu(tilt_threshold - up_proj)
    action_pen = torch.sum(actions * actions, dim=-1)

    # closing speed towards the target (potential based shaping):
    # potentials = -||to_target||/dt  ->  (potentials - prev_potentials) > 0
    # when the ant gets closer to the target.  Clipped so it never pushes
    # the ant away from the direction it is running.
    target_progress = torch.relu(potentials - prev_potentials)

    reward = (w_forward * forward_speed
              + w_world_x * world_x_speed
              - w_lateral * lateral_pen
              - w_yaw * yaw_pen
              - w_vertical * vertical_pen
              - w_upright * tilt_pen
              - w_action * action_pen
              + w_alive
              + w_target * target_progress)

    components = {
        "forward_speed": w_forward * forward_speed,
        "world_x_speed": w_world_x * world_x_speed,
        "lateral_penalty": -w_lateral * lateral_pen,
        "yaw_rate_penalty": -w_yaw * yaw_pen,
        "vertical_penalty": -w_vertical * vertical_pen,
        "tilt_penalty": -w_upright * tilt_pen,
        "action_penalty": -w_action * action_pen,
        "alive_bonus": w_alive * torch.ones_like(fwd_vel),
        "target_progress": w_target * target_progress,
    }

    return reward, components
