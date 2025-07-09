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
import time
import threading
from stable_baselines3.common.callbacks import BaseCallback

class TerminalUI:
    """终端UI类，用于显示训练进度和ROV状态"""
    def __init__(self, steps=1000000):
        self.step_count = 0
        self.start_time = time.time()
        self.last_reward = 0.0
        self.rov_state = {
            'target_pos': np.array([0.0, 0.0, -20.0]),
            'current_pos': np.array([0.0, 0.0, 0.0]),
            'current_orient': np.array([0.0, 0.0, 0.0]),  # Roll, Pitch, Yaw in degrees
            'pos_error': 0.0,
            'orient_error': 0.0,
            'vel_error': 0.0,
            'action': np.zeros(6)
        }
        
        # 启动UI更新线程
        self.running = True
        self.ui_thread = threading.Thread(target=self._update_ui_loop)
        self.ui_thread.daemon = True
        self.ui_thread.start()
    
    def update_training_info(self, step_count, reward):
        """更新训练信息"""
        self.step_count = step_count
        self.last_reward = reward
    
    def update_rov_state(self, current_state, target_pos, target_orient, pos_error, orient_error, vel_error, action):
        """更新ROV状态信息"""
        self.rov_state['current_pos'] = current_state[:3]
        
        # 将四元数转换为欧拉角（度）
        quat = current_state[3:7]
        roll, pitch, yaw = self._quaternion_to_euler(quat)
        self.rov_state['current_orient'] = np.array([roll, pitch, yaw])
        
        self.rov_state['target_pos'] = target_pos
        self.rov_state['pos_error'] = pos_error
        self.rov_state['orient_error'] = orient_error
        self.rov_state['vel_error'] = vel_error
        self.rov_state['action'] = action
    
    def _quaternion_to_euler(self, quat):
        """将四元数转换为欧拉角（度）"""
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    
    def _update_ui_loop(self):
        """UI更新循环"""
        while self.running:
            self._clear_screen()
            self._display_ui()
            time.sleep(0.1)  # 10Hz更新频率
    
    def _clear_screen(self):
        """清屏"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _display_ui(self):
        """显示UI"""
        elapsed_time = time.time() - self.start_time
        fps = self.step_count / elapsed_time if elapsed_time > 0 else 0
        
        print("=" * 80)
        print("                    PPO ROV 控制器训练状态")
        print("=" * 80)
        
        # 训练进度信息
        print(f"  步数: {self.step_count:>6} | 训练时长: {elapsed_time:>6.0f}s | FPS: {fps:>6.0f}")
        print()
        
        # ROV状态信息
        print("  --- ROV 状态 ---")
        print(f"    目标位置 (X,Y,Z): {self.rov_state['target_pos'][0]:>7.2f}, {self.rov_state['target_pos'][1]:>7.2f}, {self.rov_state['target_pos'][2]:>7.2f}")
        print(f"    当前位置 (X,Y,Z): {self.rov_state['current_pos'][0]:>7.2f}, {self.rov_state['current_pos'][1]:>7.2f}, {self.rov_state['current_pos'][2]:>7.2f}")
        print(f"    当前姿态 (R,P,Y): {self.rov_state['current_orient'][0]:>7.2f}, {self.rov_state['current_orient'][1]:>7.2f}, {self.rov_state['current_orient'][2]:>7.2f} (度)")
        print()
        
        # 误差与奖励信息
        print("  --- 误差与奖励 ---")
        print(f"    位置误差: {self.rov_state['pos_error']:>7.3f} m")
        print(f"    姿态误差: {self.rov_state['orient_error']:>7.3f}")
        print(f"    速度误差: {self.rov_state['vel_error']:>7.3f} m/s")
        print(f"    上一步奖励: {self.last_reward:>7.3f}")
        print()
        
        # Agent动作输出
        print("  --- Agent 动作输出 ---")
        print(f"    力 (X,Y,Z):   {self.rov_state['action'][0]:>9.2f}, {self.rov_state['action'][1]:>9.2f}, {self.rov_state['action'][2]:>9.2f}")
        print(f"    力矩 (R,P,Y): {self.rov_state['action'][3]:>9.2f}, {self.rov_state['action'][4]:>9.2f}, {self.rov_state['action'][5]:>9.2f}")
        print()
        
        # 进度条
        progress = min(self.step_count / 1000000.0, 1.0)  # 假设总步数为100000
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"  训练进度: [{bar}] {progress*100:.1f}%")
        print("=" * 80)
    
    def stop(self):
        """停止UI更新"""
        self.running = False


class PrintProgressCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=1, ui=None):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.ui = ui

    def _on_step(self) -> bool:
        if self.ui and self.num_timesteps % 50 == 0:  # 更新UI频率
            mean_reward = np.mean(self.locals.get("rewards", [0]))
            self.ui.update_training_info(self.num_timesteps, mean_reward)
        
        if self.num_timesteps % self.print_freq == 0:
            mean_reward = np.mean(self.locals.get("rewards", [0]))
            # 这里我们注释掉原来的打印，因为UI会显示信息
            # print(f"Step: {self.num_timesteps} - Recent reward mean: {mean_reward:.3f}")
        return True


class ROVEnv(gym.Env):
    """Custom Gym environment for ROV control"""
    metadata = {'render.modes': ['human']}

    def __init__(self, uuv_name, ui=None):
        super(ROVEnv, self).__init__()

        # Initialize the ROS node
        rospy.init_node('rov_ppo_env', anonymous=True)

        # The name of the UUV to be controlled
        self.uuv_name = uuv_name
        
        # UI reference
        self.ui = ui

        # The observation space will be the ROV's position, orientation (as a quaternion),
        # linear and angular velocities. This is a total of 13 values.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # The action space will be the 6-DOF wrench to be applied to the ROV.
        # We'll set a reasonable limit for the thruster forces.
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([21]*6)
        # 创建一个从离散索引 [0, 20] 到实际力值 [-1000, 1000] 的映射
        #    (索引 - 10) * 100 => (0-10)*100 = -1000, (10-10)*100 = 0, (20-10)*100 = 1000
        self.action_map = np.array([-1000.,  -900.,  -800.,  -700.,  -600.,  -500.,  -400.,  -300.,
        -200.,  -100.,     0.,   100.,   200.,   300.,   400.,   500.,
         600.,   700.,   800.,   900.,  1000.])


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
        # action = np.clip(action, -1.0, 1.0)
        # tau = action * 1000.0
        print(action)
        tau = self.action_map[action]
        print(tau)
        # 发布控制指令
        wrench_msg = WrenchStamped()
        wrench_msg.wrench.force.x = tau[0]
        wrench_msg.wrench.force.y = tau[1]
        wrench_msg.wrench.force.z = tau[2]
        wrench_msg.wrench.torque.x = tau[3]
        wrench_msg.wrench.torque.y = tau[4]
        wrench_msg.wrench.torque.z = tau[5]
        
        self.wrench_pub.publish(wrench_msg)

        # Wait for the next state to be received
        rospy.sleep(0.1)

        # Calculate the reward
        reward = self.calculate_reward()
        
        # 更新UI状态
        if self.ui and self.current_state is not None:
            pos_error = np.linalg.norm(self.current_state[:3] - self.target_position)
            orient_error = 1 - np.abs(np.dot(self.current_state[3:7], self.target_orientation))
            vel_error = np.linalg.norm(self.current_state[7:])
            
            self.ui.update_rov_state(
                self.current_state, 
                self.target_position, 
                self.target_orientation,
                pos_error,
                orient_error,
                vel_error,
                tau
            )

        # Check if the episode is done
        done = self.is_done()

        return self.current_state, reward, done, {}

    def calculate_reward(self):
        """Calculate the reward for the current state"""
        if self.current_state is None:
            return 0.0
            
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
        if self.current_state is None:
            return False
            
        # The episode ends if the ROV goes too far from the target
        pos_error = np.linalg.norm(self.current_state[:3] - self.target_position)
        if pos_error > 15:  # 增加容错范围
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
    # 创建终端UI
    steps = 1000000
    terminal_ui = TerminalUI(steps=steps) 
    
    try:
        # 创建环境
        env = ROVEnv(uuv_name='rexrov', ui=terminal_ui)
        
        print("初始化PPO ROV 稳定控制器...")
        rospy.sleep(2)  # 等待ROS初始化
        
        # 设置模型保存路径
        home_dir = os.path.expanduser("~")
        tensorboard_log_path = os.path.join(home_dir, "ppo_rov_logs")
        model_save_path = os.path.join(home_dir, "ppo_rov_controller")

        # 创建PPO模型
        model = PPO(
            "MlpPolicy",
            env,
            # n_steps=2048,
            # batch_size=512,
            # learning_rate=3e-4,
            # gamma=0.99,
            verbose=0,  # 关闭详细输出，使用我们的UI
            tensorboard_log=tensorboard_log_path
        )
        
        # 创建回调函数
        callback = PrintProgressCallback(print_freq=1000, ui=terminal_ui)
        
        # 开始训练
        print("开始训练...")
        rospy.sleep(1)

        model.learn(total_timesteps=steps, callback=callback)
        
        print("训练完成！保存模型...")
        model.save(model_save_path)
        print(f"模型已保存到: {model_save_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
    finally:
        # 停止UI
        terminal_ui.stop()
        print("程序结束")