#!/usr/bin/env python3
import rospy
import numpy as np
import sys, select, termios, tty, threading
from uuv_control_interfaces import DPControllerBase

key_mapping = {
    'w': (0,  1), 's': (0, -1),
    'a': (1, -1), 'd': (1,  1),
    'q': (2,  1), 'e': (2, -1),
    'j': (3, -1), 'l': (3,  1),
    'i': (4,  1), 'k': (4, -1),
    'u': (5, -1), 'o': (5,  1),
    'x': 'reset'
}

force_magnitude = 1000.0

class KeyboardDPController(DPControllerBase):
    def __init__(self):
        super(KeyboardDPController, self).__init__(self)
        self._tau = np.zeros(6)
        self._is_init = True
        # The current state of the ROV
        self.current_state = None
        self.odom_sub = rospy.Subscriber(f'/{self.uuv_name}/pose_gt', Odometry, self.odom_callback)
        rospy.loginfo("Keyboard DP Controller initialized")

    def _reset_controller(self):
        super(KeyboardDPController, self)._reset_controller()
        self._tau = np.zeros(6)

    def update_controller(self):
        if not self._is_init:
            return
        self.publish_control_wrench(self._tau)

    def process_key(self, key):
        if key in key_mapping:
            cmd = key_mapping[key]
            if cmd == 'reset':
                self._tau[:] = 0
                rospy.loginfo("Reset tau to zero.")
            else:
                axis, direction = cmd
                self._tau[axis] = force_magnitude * direction
                rospy.loginfo("Set tau[%d] to %s" % (axis, self._tau[axis]))
        else:
            rospy.loginfo("Unrecognized key: %s" % key)


def keyboard_thread(ctrl):
    settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                ctrl.process_key(key)
    except Exception as e:
        rospy.logwarn("Keyboard input error: %s", e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        rospy.loginfo("Keyboard thread exiting.")


if __name__ == '__main__':
    rospy.init_node('keyboard_dp_controller')
    ctrl = KeyboardDPController()

    thread = threading.Thread(target=keyboard_thread, args=(ctrl,))
    thread.daemon = True
    thread.start()

    rospy.loginfo("Running keyboard controller. Press Ctrl+C to stop.")
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    rospy.loginfo("Shutting down Keyboard DP Controller.")
