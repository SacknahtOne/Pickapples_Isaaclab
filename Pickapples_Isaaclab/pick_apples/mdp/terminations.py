from __future__ import annotations

import torch
from typing import List

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from leisaac.utils.robot_utils import is_so101_at_rest_pose


def apple_in_bowl(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg,
    bowl_cfg: SceneEntityCfg,
    distance_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if apple is close enough to bowl to be considered 'in' it.

    Args:
        env: The RL environment instance.
        apple_cfg: Configuration for the apple entity.
        bowl_cfg: Configuration for the bowl entity.
        distance_threshold: Distance threshold to consider apple "in" bowl.
    
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get apple and bowl objects
    apple: RigidObject = env.scene[apple_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    
    # Get positions
    apple_pos = apple.data.root_pos_w  # shape: (num_envs, 3)
    bowl_pos = bowl.data.root_pos_w    # shape: (num_envs, 3)
    
    # Calculate distance between apple and bowl
    distance = torch.norm(apple_pos - bowl_pos, dim=-1)  # shape: (num_envs,)
    
    # Success if apple is close enough to bowl
    success = distance < distance_threshold
    
    # Also check if robot is at rest (optional - can remove this)
    try:
        joint_pos = env.scene["robot"].data.joint_pos
        joint_names = env.scene["robot"].data.joint_names
        robot_at_rest = is_so101_at_rest_pose(joint_pos, joint_names)
        success = torch.logical_and(success, robot_at_rest)
    except:
        # If robot rest check fails, just use distance check
        pass

    if success.any():
        print("Apple successfully placed in bowl!")

    return success
