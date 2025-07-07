#!/usr/bin/env python3
import gym
from stable_baselines3 import PPO
from gym import spaces
import numpy as np
import rospy
from uuv_control_interfaces import DPControllerBase
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Empty

class ROVEnv(gym.Env):
    """Custom Gym environment for ROV control"""
    metadata = {'render.modes': ['human']}

    def __init__(self, uuv_name):
        super(ROVEnv, self).__init__()

        # Initialize the ROS node
        rospy.init_node('rov_ppo_env', anonymous=True)

        # The name of the UUV to be controlled
        self.uuv_name = uuv_name

        # The observation space will be the ROV's position, orientation (as a quaternion),
        # linear and angular velocities. This is a total of 13 values.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # The action space will be the 6-DOF wrench to be applied to the ROV.
        # We'll set a reasonable limit for the thruster forces.
        self.action_space = spaces.Box(low=-5000, high=5000, shape=(6,), dtype=np.float32)

        # Subscribe to the odometry topic to get the ROV's state
        self.odom_sub = rospy.Subscriber(f'/{self.uuv_name}/pose_gt', Odometry, self.odom_callback)

        # Publisher for the thruster commands
        self.wrench_pub = rospy.Publisher(f'/{self.uuv_name}/thruster_manager/input_stamped', WrenchStamped, queue_size=10)

        # The current state of the ROV
        self.current_state = None

        # The target position and orientation for the ROV
        self.target_position = np.array([0, 0, -20]) # Stay at a depth of 20 meters
        self.target_orientation = np.array([0, 0, 0, 1]) # No rotation

    def odom_callback(self, msg):
        """Callback function for the odometry subscriber"""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        lin_vel = msg.twist.twist.linear
        ang_vel = msg.twist.twist.angular

        self.current_state = np.array([
            pos.x, pos.y, pos.z,
            orient.x, orient.y, orient.z, orient.w,
            lin_vel.x, lin_vel.y, lin_vel.z,
            ang_vel.x, ang_vel.y, ang_vel.z
        ])

    def step(self, action):
        """Execute one time step within the environment"""
        # Publish the action as a wrench command
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = rospy.Time.now()
        wrench_msg.header.frame_id = f'{self.uuv_name}/base_link'
        wrench_msg.wrench.force.x = action[0]
        wrench_msg.wrench.force.y = action[1]
        wrench_msg.wrench.force.z = action[2]
        wrench_msg.wrench.torque.x = action[3]
        wrench_msg.wrench.torque.y = action[4]
        wrench_msg.wrench.torque.z = action[5]

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxSTEPxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.wrench_pub.publish(wrench_msg)

        # Wait for the next state to be received
        rospy.sleep(0.1)

        # Calculate the reward
        reward = self.calculate_reward()

        # Check if the episode is done
        done = self.is_done()

        return self.current_state, reward, done, {}

    def calculate_reward(self):
        """Calculate the reward for the current state"""
        # Position error
        pos_error = np.linalg.norm(self.current_state[:3] - self.target_position)

        # Orientation error (using the dot product of the quaternions)
        orient_error = 1 - np.abs(np.dot(self.current_state[3:7], self.target_orientation))

        # Velocity error
        vel_error = np.linalg.norm(self.current_state[7:])

        # The reward is the negative of the sum of the errors
        reward = - (pos_error + orient_error + vel_error)

        return reward

    def is_done(self):
        """Check if the episode has ended"""
        # The episode ends if the ROV goes too far from the target
        pos_error = np.linalg.norm(self.current_state[:3] - self.target_position)
        if pos_error > 10:
            return True
        return False

    def reset(self):
        """Reset the state of the environment to an initial state"""
        # Reset the simulation (this needs to be implemented in the ROS launch file)
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            reset_world()
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

        # Wait for the ROV to settle
        rospy.sleep(1)

        return self.current_state

    def render(self, mode='human', close=False):
        """Render the environment to the screen"""
        pass

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Create the custom Gym environment
env = ROVEnv(uuv_name='rexrov')

# Instantiate the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
model.learn(total_timesteps=1000)

# Save the trained model
model.save("rov_ppo_model")

# Close the environment
env.close()




00000000000000000000
#!/usr/bin/env python3

# File: train_ppo_rov.py
# Description: 训练一个 PPO 模型用于 ROV 姿态/位置稳定控制

import rospy
import gym
import numpy as np
from stable_baselines3 import PPO
from gym import spaces
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import time
import rospy.rostime
from stable_baselines3.common.callbacks import BaseCallback

class PrintProgressCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=1):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            mean_reward = np.mean(self.locals["rewards"])
            print(f"Step: {self.num_timesteps} - Recent reward mean: {mean_reward:.3f}")
        return True

class ROVStabilizationEnv(gym.Env):
    def __init__(self):
        super(ROVStabilizationEnv, self).__init__()

        rospy.init_node('rov_rl_env', anonymous=True)

        self._thruster_pub = rospy.Publisher('/rexrov/thruster_manager/input_stamped', WrenchStamped, queue_size=1)
        rospy.Subscriber('/rexrov/pose_gt', Odometry, self._odom_callback)

        rospy.wait_for_service('/gazebo/reset_simulation')
        self._reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self._odom = None
        self._target = np.zeros(6)  # 目标：原点 + 姿态全0

        # 连续动作空间：tau 控制
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # 观测空间：位置+姿态误差 + 线速度+角速度（12维）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def _odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        # 这里只取 XYZ 和 orientation 简化处理（可拓展为欧拉角）
        self._odom = np.array([
            pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z,
            vel.x, vel.y, vel.z,
            ang.x, ang.y, ang.z
        ])

    def _get_obs(self):
        while self._odom is None and not rospy.is_shutdown():
            # rospy.sleep(0.1)
            rospy.rostime.wallsleep(0.1)
        return self._odom

    def reset(self):
        """Reset the state of the environment to an initial state"""
        # Reset the simulation (this needs to be implemented in the ROS launch file)
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            reset_world()
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

        # Wait for the ROV to settle
        rospy.sleep(0.1)
        print("reset")
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        tau = action * 1000.0

        # 发布控制指令
        msg = WrenchStamped()
        msg.wrench.force.x = tau[0]
        msg.wrench.force.y = tau[1]
        msg.wrench.force.z = tau[2]
        msg.wrench.torque.x = tau[3]
        msg.wrench.torque.y = tau[4]
        msg.wrench.torque.z = tau[5]
        # print("publish ",msg)
        self._thruster_pub.publish(msg)

        # rospy.sleep(0.1)  # 控制频率
        rospy.rostime.wallsleep(0.1)
        print("STEP",action)
        obs = self._get_obs()
        reward = - np.linalg.norm(obs[:3]) - 0.1 * np.linalg.norm(obs[6:])
        done = False
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

if __name__ == '__main__':
    env = ROVStabilizationEnv()
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_rov_tensorboard")

    print("开始训练 PPO ROV 稳定控制器...")
    # model.learn(total_timesteps=10)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        tensorboard_log="~/ppo_rov_logs"
    )
    callback = PrintProgressCallback(print_freq=100)
    model.learn(total_timesteps=3_000_000, callback=callback)
    print("保存模型为 ppo_rov_controller.zip")
    model.save("~/ppo_rov_controller")
