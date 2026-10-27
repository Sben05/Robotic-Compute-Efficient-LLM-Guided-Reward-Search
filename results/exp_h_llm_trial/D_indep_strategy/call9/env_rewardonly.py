@torch.jit.script
def compute_reward(root_states: torch.Tensor,
                   actions: torch.Tensor,
                   inv_start_rot: torch.Tensor,
                   basis_vec0: torch.Tensor,
                   basis_vec1: torch.Tensor,
                   up_axis_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Posture-gated forward running reward for the Ant task.

    forward_speed : torso velocity along the horizontal direction the ant was
                    spawned facing (world frame), signed (running backwards
                    is penalised).
    uprightness   : z component of the torso up axis (1 = perfectly upright,
                    0 = on its side, <0 = upside down).
    gate          : smooth posture gate, ~1 while standing tall, ~0 when
                    tumbling -> a flipped/falling ant earns no speed reward.
    """

    num_envs = root_states.shape[0]
    device = root_states.device

    # ----------------------------- hyper parameters -----------------------------
    speed_coeff: float = 1.0      # weight on the (gated) forward velocity
    alive_coeff: float = 0.3      # small bonus for staying upright and alive
    posture_coeff: float = 0.1    # penalty for leaning over / tumbling
    ctrl_coeff: float = 0.0005    # tiny action regularisation
    gate_temp: float = 8.0        # temperature of the uprightness gate
    gate_bias: float = 0.4        # uprightness offset of the gate

    # ----------------------------- torso state ----------------------------------
    quat = root_states[:, 3:7]
    quat = quat / torch.sqrt(torch.sum(quat * quat, dim=-1, keepdim=True)).clamp(min=1e-6)
    lin_vel = root_states[:, 7:10]

    # body frame basis vectors -> (num_envs, 3)
    b0 = basis_vec0
    if b0.dim() == 1:
        b0 = b0.unsqueeze(0).expand(num_envs, 3)
    else:
        b0 = b0.expand(num_envs, 3)
    b1 = basis_vec1
    if b1.dim() == 1:
        b1 = b1.unsqueeze(0).expand(num_envs, 3)
    else:
        b1 = b1.expand(num_envs, 3)

    arange3 = torch.arange(3, device=device, dtype=torch.long)
    up_mask = (arange3 == up_axis_idx).to(root_states.dtype).unsqueeze(0)   # (1,3)
    idx1 = torch.full((num_envs, 1), up_axis_idx, device=device, dtype=torch.long)

    # ------------------------- uprightness (body up in world) --------------------
    # quaternion rotation: v' = v + w*t + qv x t   with   t = 2 * (qv x v)
    qv = quat[:, 1:4]
    t = 2.0 * torch.cross(qv, b0, dim=-1)
    up_world = b0 + quat[:, 0:1] * t + torch.cross(qv, t, dim=-1)
    up_proj = torch.gather(up_world, 1, idx1).squeeze(1)                   # (N,)

    # ------------- world frame direction the ant was spawned facing -------------
    inv_rot = inv_start_rot
    if inv_rot.dim() == 1:
        inv_rot = inv_rot.unsqueeze(0).expand(num_envs, 4)
    else:
        inv_rot = inv_rot.expand(num_envs, 4)
    start_quat = torch.cat((inv_rot[:, 0:1], -inv_rot[:, 1:4]), dim=-1)   # conjugate
    start_quat = start_quat / torch.sqrt(torch.sum(start_quat * start_quat, dim=-1, keepdim=True)).clamp(min=1e-6)
    sqv = start_quat[:, 1:4]
    t2 = 2.0 * torch.cross(sqv, b1, dim=-1)
    fwd = b1 + start_quat[:, 0:1] * t2 + torch.cross(sqv, t2, dim=-1)       # (N,3)

    # project the reference direction onto the horizontal plane and normalise
    fwd_up = torch.gather(fwd, 1, idx1).squeeze(1)
    fwd_h = fwd - up_mask * fwd_up.unsqueeze(1)
    fwd_h = fwd_h / torch.sqrt(torch.sum(fwd_h * fwd_h, dim=-1, keepdim=True)).clamp(min=1e-6)

    forward_speed = torch.sum(lin_vel * fwd_h, dim=-1)                      # (N,) m/s

    # ------------------------------- posture gate -------------------------------
    uprightness = torch.clamp(up_proj, min=0.0, max=1.0)
    gate = torch.sigmoid(gate_temp * (up_proj - gate_bias)) * uprightness    # ~1 upright, ~0 tumbling

    # ---------------------------- reward components -----------------------------
    speed_term = speed_coeff * gate * forward_speed
    alive_term = alive_coeff * gate
    posture_term = -posture_coeff * (1.0 - uprightness)
    ctrl_term = -ctrl_coeff * torch.sum(actions * actions, dim=-1)

    total = speed_term + alive_term + posture_term + ctrl_term

    rewards = {
        "forward_speed": speed_term,
        "upright_alive": alive_term,
        "posture_penalty": posture_term,
        "action_penalty": ctrl_term,
    }
    return total, rewards
