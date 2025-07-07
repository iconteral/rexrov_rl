#!/usr/bin/env python3
#!/usr/bin/env python3
import gym
from stable_baselines3 import PPO
from gym import spaces
import numpy as np
import rospy
# from uuv_control_interfaces import DPControllerBase
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Empty
import os
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
        # self.action_space = spaces.Box(low=-5000, high=5000, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

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
        # print("GETTING ODOM ",pos)
        self.current_state = np.array([
            pos.x, pos.y, pos.z,
            orient.x, orient.y, orient.z, orient.w,
            lin_vel.x, lin_vel.y, lin_vel.z,
            ang_vel.x, ang_vel.y, ang_vel.z
        ])

    def step(self, action):
        """Execute one time step within the environment"""
        # Publish the action as a wrench command
        # wrench_msg = WrenchStamped()
        # wrench_msg.header.stamp = rospy.Time.now()
        # wrench_msg.header.frame_id = f'{self.uuv_name}/base_link'
        # wrench_msg.wrench.force.x = action[0]
        # wrench_msg.wrench.force.y = action[1]
        # wrench_msg.wrench.force.z = action[2]
        # wrench_msg.wrench.torque.x = action[3]
        # wrench_msg.wrench.torque.y = action[4]
        # wrench_msg.wrench.torque.z = action[5]
        action = np.clip(action, -1.0, 1.0)
        tau = action * 1000.0
        # 发布控制指令
        wrench_msg = WrenchStamped()
        # wrench_msg.wrench.force.x = 1000
        wrench_msg.wrench.force.x = tau[0]
        wrench_msg.wrench.force.y = tau[1]
        wrench_msg.wrench.force.z = tau[2]
        wrench_msg.wrench.torque.x = tau[3]
        wrench_msg.wrench.torque.y = tau[4]
        wrench_msg.wrench.torque.z = tau[5]
        # print("publish ",msg)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxSTEPxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # print(wrench_msg)
        self.wrench_pub.publish(wrench_msg)

        # Wait for the next state to be received
        rospy.sleep(0.1)

        # Calculate the rewardodom
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
        # print("pos_err==",pos_error)
        # print("pos_sta==",self.current_state[:3])
        if pos_error > 2:
            # print("DONE: ",pos_error)
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

if __name__ == '__main__':
    env = env = ROVEnv(uuv_name='rexrov')
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_rov_tensorboard")

    print("开始训练 PPO ROV 稳定控制器...")
    # model.learn(total_timesteps=10)
    home_dir = os.path.expanduser("~")
    tensorboard_log_path = os.path.join(home_dir, "ppo_rov_logs")
    model_save_path = os.path.join(home_dir, "ppo_rov_controller")

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        tensorboard_log=tensorboard_log_path # 使用绝对路径
        # tensorboard_log="~/ppo_rov_logs"
    )
    callback = PrintProgressCallback(print_freq=50)
    model.learn(total_timesteps=100000, callback=callback)
    print("保存模型为 ppo_rov_controller.zip")
    model.save(model_save_path) # 使用绝对路径
