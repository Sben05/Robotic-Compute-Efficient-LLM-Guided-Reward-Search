@torch.jit.script
def compute_reward(root_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and rotation from root_states
    velocity = root_states[:, 7:10]
    rotation = root_states[:, 3:7]
    
    # Unpack quaternion components
    x = rotation[:, 0]
    y = rotation[:, 1]
    z = rotation[:, 2]
    w = rotation[:, 3]
    
    # Calculate forward vector components based on rotation
    forward_x = 1 - 2 * y * y - 2 * z * z
    forward_y = 2 * x * y + 2 * w * z
    forward_z = 2 * x * z - 2 * w * y
    
    # Compute forward velocity as dot product between global velocity and forward vector
    forward_velocity = (
        velocity[:, 0] * forward_x + 
        velocity[:, 1] * forward_y + 
        velocity[:, 2] * forward_z
    )
    
    # Use forward velocity as the reward
    reward = forward_velocity
    
    return reward, {'forward_velocity': reward}
