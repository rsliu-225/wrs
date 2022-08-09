import serial
import time
import motorcontrol.Communication as Communication
import pickle
import config

DISPLAY = False


class MotorNema23():
    def __init__(self):
        # self.step_per_turn = 200
        # self.rpm = 6000
        # self.step_per_second = (self.step_per_turn * self.rpm) / 60
        # self.wait = (1 / self.step_per_second) * 1000000
        self._comm = Communication.Communication("COM3", 9600, .5)
        # self.comm = Communication.Communication("/dev/ttyACM0", 9600, .5)
        self.hi()
        self.max_digit = 5
        self.limit = 180
        self.max_step = self.cal_counter(self.limit)
        try:
            self.state = pickle.load(open(config.MOTOR_STATE_PATH, 'rb'))
            print(self.state)
        except:
            self.goto_init()

    def open(self):
        self._comm.open()

    def is_open(self):
        return self._comm.is_open()

    def cal_counter(self, degree, reduce_ratio=80, ppr=400):
        return int(degree * reduce_ratio / 360 * ppr)

    def hi(self):
        self._comm.wait_for_arduino()
        print('Arduino is ready')

    def gen_cmd(self, step, dir=0, use_sensor=0):
        '''

        :param step:
        :param dir: 0, clockwise; 1, counter clockwise
        :param use_sensor:
        :return:
        '''
        return str(step).zfill(self.max_digit) + chr(self._comm.special_byte) + \
               str(dir).zfill(self.max_digit) + chr(self._comm.special_byte) + \
               str(use_sensor).zfill(self.max_digit)

    def send_commond(self, commond):
        waiting_for_reply = False
        while self._comm.is_open():
            if self._comm.in_waiting() == 0 and not waiting_for_reply:
                self._comm.send_to_arduino(commond)
                if DISPLAY:
                    print('-------sent from PC--------')
                    print('BYTES SENT -> ' + self._comm.bytes2str(commond))
                    print('STR ' + commond)
                waiting_for_reply = True

            if self._comm.in_waiting():
                data_recvd = self._comm.recv_from_arduino()
                if data_recvd[0] == 0:
                    if DISPLAY:
                        self._comm.display_data(data_recvd[1])
                if data_recvd[0] > 0:
                    if DISPLAY:
                        print(data_recvd)
                        self._comm.display_data(data_recvd[1])
                        print('Reply Received')
                    break
                time.sleep(0.3)
        return True

    def goto_init(self):
        try:
            self.state = pickle.load(open('./motor_state.pkl', 'rb'))
            print(self.state)
            if self.state[1] > 0:
                dir = 1
            else:
                dir = 0
        except:
            dir = input('Input rotation direction (0,1): ')
        commond = self.gen_cmd(self.max_step, dir, 1)
        flag = self.send_commond(commond)
        if flag:
            print('---------motion complete---------')
            pickle.dump([-self.max_step, 0, self.max_step], open('./motor_state.pkl', 'wb'))

    def goto_pos(self, pos):
        print('Current state:', self.state)
        step = pos - self.state[1]
        if pos > self.state[2] or pos < self.state[0]:
            print('Invalid input postion!')
        if step != 0:
            print('Move step:', step)
            if step > 0:
                dir = 0
            else:
                dir = 1
            commond = self.gen_cmd(abs(step), dir, 0)
            flag = self.send_commond(commond)
            if flag:
                print('---------motion complete---------')
                self.state[1] = self.state[1] + step
                print('Updated state:', self.state)
                pickle.dump(self.state, open('./motor_state.pkl', 'wb'))
        else:
            print('The motor is at the given position! ')

    def rot_degree(self, clockwise=1, rot_deg=10, use_sensor=0):
        step = self.cal_counter(rot_deg)
        # print(step)
        commond = self.gen_cmd(step, clockwise, use_sensor)
        flag = self.send_commond(commond)
        if flag:
            print('---------motion complete---------')


if __name__ == '__main__':
    motor = MotorNema23()
    # print(motor.cal_counter(180))
    # motor.goto_init()
    # time.sleep(1)

    # motor.goto_pos(-10000)
    # time.sleep(2)
    motor.rot_degree(clockwise=1, rot_deg=10)
    # motor.rot_degree(clockwise=1, rot_deg=1)
    # motor.goto_pos(-3000)
    # time.sleep(1)
    # motor.goto_pos(-9500)
    # time.sleep(2)

    # motor.goto_pos(10000)
    # time.sleep(1)
    # motor.goto_pos(1000)
    # time.sleep(1)
    # motor.goto_pos(10000)

    # motor.goto_pos(-11000)
    # time.sleep(1)
    # motor.goto_pos(-3000)
    # time.sleep(1)
    # motor.goto_pos(11000)
    # time.sleep(1)
    # # motor.goto_init()

    # motor.rot_degree(clockwise=1, rot_deg=10)
