#!/usr/bin/env python3
import rospy
import numpy as np
from uuv_control_interfaces import DPControllerBase
from stable_baselines3 import PPO  # or SAC
import torch


class RLDynamicPositioningController(DPControllerBase):
    def __init__(self):
        super(RLDynamicPositioningController, self).__init__(self)

        self._is_init = True
        model_path = rospy.get_param("~model_path", "")
        if not model_path:
            raise rospy.ROSException("RL model path not set via ~model_path")

        rospy.loginfo(f"Loading RL model from {model_path}")
        self._model = PPO.load(model_path)  # or SAC.load(model_path)

        # Assume 6D tau output, bound within [-max_force, max_force]
        self._max_force = rospy.get_param("~max_force", 1000.0)

        rospy.loginfo("RL DP Controller initialized.")

    def update_controller(self):
        if not self._is_init or not self.odom_is_init:
            return

        # Create observation vector from current state
        obs = np.concatenate([
            self.error_pose_euler,      # 6D pose error
            self._errors['vel']         # 6D velocity error
        ])

        # Normalize obs if needed (or just use raw)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Predict action from RL model
        action, _ = self._model.predict(obs_tensor, deterministic=True)

        # Scale to force range
        tau = np.clip(action, -1.0, 1.0) * self._max_force

        self.publish_control_wrench(tau)
