@torch.jit.script
def compute_reward(obs_buf: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity from obs_buf (index 1 is vel_loc's x-component)
    forward_vel = obs_buf[:, 1]
    # Extract uprightness projection (index 11)
    up_proj = obs_buf[:, 11]
    
    # Compute action penalty as sum of squared actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    
    # Reward weights
    forward_weight = 1.0
    upright_weight = 2.0
    action_penalty_weight = 0.05
    
    # Total reward components
    reward_forward = forward_weight * forward_vel
    reward_upright = upright_weight * up_proj
    reward_actions = -action_penalty_weight * action_penalty
    
    # Total reward
    total_reward = reward_forward + reward_upright + reward_actions
    
    # Return total reward and individual components
    reward_components = {
        "forward_velocity": reward_forward,
        "uprightness": reward_upright,
        "action_penalty": reward_actions
    }
    
    return total_reward, reward_components
