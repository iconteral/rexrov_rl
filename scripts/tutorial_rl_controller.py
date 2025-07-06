#!/usr/bin/env python3
# Copyright (c) 2025 The UUV Simulator RL Example Authors.
# Licensed under the Apache License, Version 2.0

import rospy
import numpy as np
from uuv_control_interfaces import DPControllerBase

class SimpleRLController(DPControllerBase):
    """
    强化学习动力定位控制器示例。以状态-动作空间离散化的 Q-learning 伪实现为例。
    """

    def __init__(self):
        super(SimpleRLController, self).__init__(self)
        rospy.loginfo_throttle(2,"[RLController] 初始化控制器...")
        self.action_space = self._create_action_space()
        self.state_space = self._create_state_space()
        self.q_table = np.zeros((len(self.state_space), len(self.action_space)))
        self.epsilon = rospy.get_param("~epsilon", 0.1)
        self.alpha = rospy.get_param("~alpha", 0.01)
        self.gamma = rospy.get_param("~gamma", 0.9)
        self._last_state = None
        self._last_action = None
        self._is_init = True
        rospy.loginfo_throttle(2,"[RLController] 控制器初始化完成，动作空间%d，状态空间%d", len(self.action_space), len(self.state_space))

    def _create_state_space(self):
        levels = [-1, 0, 1]
        space = [s for s in np.array(np.meshgrid(*([levels]*6))).T.reshape(-1,6)]
        rospy.loginfo_throttle(2,"[RLController] 状态空间创建完成，共%d个状态", len(space))
        return space

    def _create_action_space(self):
        levels = [-1, 0, 1]
        space = [a for a in np.array(np.meshgrid(*([levels]*6))).T.reshape(-1,6)]
        rospy.loginfo_throttle(2,"[RLController] 动作空间创建完成，共%d个动作", len(space))
        return space

    def _state_to_index(self, state):
        idx = np.argmin(np.linalg.norm(np.array(self.state_space) - state, axis=1))
        rospy.loginfo_throttle(2,"[RLController] 状态 %s 映射为索引 %d", state, idx)
        return idx

    def _action_to_index(self, action):
        idx = np.argmin(np.linalg.norm(np.array(self.action_space) - action, axis=1))
        rospy.loginfo_throttle(2,"[RLController] 动作 %s 映射为索引 %d", action, idx)
        return idx

    def _reset_controller(self):
        super(SimpleRLController, self)._reset_controller()
        self._last_state = None
        self._last_action = None
        rospy.loginfo_throttle(2,"[RLController] 控制器已重置。")

    def update_controller(self):
        if not self._is_init or not self.odom_is_init:
            rospy.logwarn_throttle(5, "[RLController] 控制器未完成初始化或未接收到里程计消息。")
            return False

        # 1. 状态定义为当前位置误差
        state = np.round(self.error_pose_euler).astype(int)
        state_idx = self._state_to_index(state)

        # 2. epsilon-greedy策略选择动作
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(len(self.action_space))
            rospy.loginfo_throttle(2,"[RLController] 随机探索动作 %d", action_idx)
        else:
            action_idx = np.argmax(self.q_table[state_idx])
            rospy.loginfo_throttle(2,"[RLController] 利用Q表选择动作 %d", action_idx)
        action = self.action_space[action_idx]

        # 3. 奖励定义
        reward = -np.linalg.norm(self.error_pose_euler)
        rospy.loginfo_throttle(2,"[RLController] 当前奖励: %.3f", reward)

        # 4. Q-table 离线更新
        if self._last_state is not None and self._last_action is not None:
            last_state_idx = self._state_to_index(self._last_state)
            last_action_idx = self._action_to_index(self._last_action)
            best_next = np.max(self.q_table[state_idx])
            old_value = self.q_table[last_state_idx, last_action_idx]
            self.q_table[last_state_idx, last_action_idx] += \
                self.alpha * (reward + self.gamma * best_next - old_value)
            rospy.loginfo_throttle(2,"[RLController] Q表更新: [%d, %d] %.3f -> %.3f", last_state_idx, last_action_idx, old_value, self.q_table[last_state_idx, last_action_idx])

        # 5. 执行动作
        # print(action.astype(float))
        tau = action.astype(float) * rospy.get_param("~max_effort", 1000.0)
        tau = np.array([1000., 0., 0., 0., 0., 0.])
        rospy.loginfo_throttle(2, "[RLController] 输出tau: %s", tau)
        
        self.publish_control_wrench(tau)

        # 6. 记录状态与动作用于下次更新
        self._last_state = state
        self._last_action = action

if __name__ == '__main__':
    rospy.init_node('tutorial_rl_controller')
    rospy.loginfo("[RLController] 节点启动！")
    try:
        node = SimpleRLController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[RLController] 捕获到中断异常，节点即将退出。")
    rospy.loginfo("[RLController] 节点退出。")
