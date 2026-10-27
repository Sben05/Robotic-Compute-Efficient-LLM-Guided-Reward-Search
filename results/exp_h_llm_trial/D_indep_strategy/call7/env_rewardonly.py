@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, obs_buf: torch.Tensor,
                   up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ------------------- hyper-parameters -------------------
    target_speed: float = 2.0      # desired forward speed [m/s]
    temp_speed: float = 0.5        # temperature of the exponential speed shaping
    w_speed: float = 1.0           # weight of the exponential speed shaping
    w_vx: float = 0.15             # weight of the linear forward-velocity bonus
    w_vt: float = 0.10             # weight of the progress-towards-target bonus
    w_head: float = 0.60           # weight of the heading-alignment term
    w_align: float = 0.30          # weight of the motion-direction alignment term
    w_up: float = 0.60             # weight of the upright/stability term
    alive_bonus: float = 0.20      # constant bonus for staying alive and upright

    pos = root_states[:, 0:3]
    vel = root_states[:, 7:10]

    # ---------------- forward (world +x) velocity ----------------
    v_x = vel[:, 0]
    v_x_pos = torch.clamp(v_x, min=0.0)

    # ---------------- horizontal velocity / target direction ----------------
    zeros = torch.zeros_like(vel[:, 0])
    v_horiz = torch.stack((vel[:, 0], vel[:, 1], zeros), dim=-1)
    v_horiz_norm = torch.norm(v_horiz, dim=-1, keepdim=True).clamp(min=1e-6)
    v_dir = v_horiz / v_horiz_norm

    to_target = targets - pos
    tgt_horiz = torch.stack((to_target[:, 0], to_target[:, 1], zeros), dim=-1)
    tgt_dir = tgt_horiz / torch.norm(tgt_horiz, dim=-1, keepdim=True).clamp(min=1e-6)

    v_t = torch.sum(v_horiz * tgt_dir, dim=-1)
    v_t_pos = torch.clamp(v_t, min=0.0)

    # ------- 1) exponential shaping on the gap to the target speed -------
    speed_gap = torch.clamp(target_speed - v_x_pos, min=0.0, max=target_speed)
    r_speed = torch.exp(-speed_gap / temp_speed)

    # ------- 2) linear bonus: keeps rewarding larger forward speeds -------
    r_vx = v_x_pos

    # ------- 3) progress towards the target (forward direction) -------
    r_vt = v_t_pos

    # ------- 4) heading alignment (env-provided torso heading vs. target) -------
    angle_to_target = obs_buf[:, 9]          # obs layout: yaw=7, roll=8, angle_to_target=9
    r_head = torch.clamp(torch.cos(angle_to_target), min=0.0, max=1.0)

    # ------- 5) alignment of the actual motion direction with the target -------
    r_align = torch.clamp(torch.sum(v_dir * tgt_dir, dim=-1), min=0.0, max=1.0)

    # ------- 6) uprightness (torso up axis vs. world up) -------
    r_up = torch.clamp(up_vec[:, 2], min=0.0, max=1.0)

    total = (w_speed * r_speed
             + w_vx * r_vx
             + w_vt * r_vt
             + w_head * r_head
             + w_align * r_align
             + w_up * r_up
             + alive_bonus * torch.ones_like(v_x))

    components: Dict[str, torch.Tensor] = {
        "speed": r_speed,
        "forward_vel": r_vx,
        "target_vel": r_vt,
        "heading": r_head,
        "align": r_align,
        "upright": r_up,
        "alive": alive_bonus * torch.ones_like(v_x),
    }
    return total, components
