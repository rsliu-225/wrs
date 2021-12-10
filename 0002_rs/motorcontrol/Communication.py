import serial
import serial.tools.list_ports
import time
import re


class Communication():
    def __init__(self, com, bps, timeout):
        self.port = com
        self.bps = bps
        self.timeout = timeout
        self.special_byte = 123
        self.start_marker = 124
        self.end_marker = 125
        try:
            self.ser = serial.Serial(self.port, self.bps)

        except Exception as e:
            print("Error:", e)

    # example
    # self.ser.write(chr(0x06).encode("utf-8")) # 十六制发送一个数据
    # print(self.ser.read().hex()) # 十六进制的读取读一个字节
    # print(self.ser.read()) # 读一个字节
    # print(self.ser.read(10).decode("gbk")) # 读十个字节
    # print(self.ser.readline().decode("gbk")) # 读一行
    # print(self.ser.readlines()) # 读取多行，返回列表，必须匹配超时（timeout)使用
    # print(self.ser.in_waiting) # 获取输入缓冲区的剩余字节数
    # print(self.ser.out_waiting) # 获取输出缓冲区的字节数
    # print(self.ser.readall()) # 读取全部字符。

    def print_name(self):
        print('-------------------------')
        print(self.ser.name)  # 设备名字
        print(self.ser.port)  # 读或者写端口
        print(self.ser.baudrate)  # 波特率
        print(self.ser.bytesize)  # 字节大小
        print(self.ser.parity)  # 校验位
        print(self.ser.stopbits)  # 停止位
        print(self.ser.timeout)  # 读超时设置
        print(self.ser.writeTimeout)  # 写超时
        print(self.ser.xonxoff)  # 软件流控
        print(self.ser.rtscts)  # 软件流控
        print(self.ser.dsrdtr)  # 硬件流控
        print(self.ser.interCharTimeout)  # 字符间隔超时
        print('-------------------------')

    def is_open(self):
        return self.ser.is_open

    def open(self):
        try:
            self.ser.open()
        except Exception as e:
            print("Error: ", e)

    def close(self):
        self.ser.close()
        if not self.is_open():
            print('---------Closed---------')

    def in_waiting(self):
        return self.ser.in_waiting

    @staticmethod
    def print_used_com():
        port_list = list(serial.tools.list_ports.comports())
        print(port_list)

    def read_size(self, size):
        return self.ser.read(size=size)

    def read(self):
        return self.ser.read()

    def readline(self):
        data = str(self.ser.readline())
        data = re.findall(r"b'*(.+)\\r\\n'", data)
        print(data)
        return data

    def send_inputdata(self, show_str):
        command = input(show_str)
        self.ser.write(command.encode())

    def write(self, data):
        print(data.encode())
        self.ser.write(data.encode())

    def send_to_arduino(self, send_str):
        txLen = chr(len(send_str))
        adj_send_str = self.encode_high_bytes(send_str)
        adj_send_str = chr(self.start_marker) + txLen + adj_send_str + chr(self.end_marker)
        self.write(adj_send_str)

    def recv_from_arduino(self):
        ck = ''
        x = 'z'
        byte_count = -1

        # wait for the start character
        while ord(x) != self.start_marker:
            x = self.ser.read()

        # save data until the end marker is found
        while ord(x) != self.end_marker:
            # print(x.decode())
            ck = ck + x.decode()
            x = self.ser.read()
            byte_count += 1

        # save the end marker byte
        ck = ck + x.decode()
        return_data = [ord(ck[1]), self.decode_high_bytes(ck)]
        # print('return_data ' + str(return_data))

        return return_data

    def encode_high_bytes(self, in_str):
        out_str = ''
        s = len(in_str)

        for n in range(0, s):
            x = ord(in_str[n])

            if x >= self.special_byte:
                out_str = out_str + chr(self.special_byte)
                out_str = out_str + chr(x - self.special_byte)
            else:
                out_str = out_str + chr(x)

        # print('encin_str', self.bytes2str(in_str))
        # print('encout_str', self.bytes2str(out_str))

        return out_str

    def decode_high_bytes(self, in_str):
        out_str = ''
        n = 0

        while n < len(in_str):
            if ord(in_str[n]) == self.special_byte:
                n += 1
                x = chr(self.special_byte + ord(in_str[n]))
            else:
                x = in_str[n]
            out_str = out_str + x
            n += 1

        # print('decin_str', self.bytes2str(in_str))
        # print('decout_str', self.bytes2str(out_str))

        return out_str

    def display_data(self, data):
        print('---------------------------')
        print('NUM BYTES SENT->   ' + str(ord(data[1])))
        print('DATA RECVD BYTES-> ' + self.bytes2str(data[2:-1]))
        print('DATA RECVD CHARS-> ' + data[2: -1])

    def bytes2str(self, data):
        byteString = ''
        n = len(data)
        for s in range(0, n):
            byteString = byteString + str(ord(data[s])) + '-'

        return byteString[:-1]

    def display_debug(self, debug_str):
        print('DEBUG MSG-> ' + debug_str[2: -1])

    def wait_for_arduino(self):
        # wait until the Arduino sends 'Arduino Ready' - allows time for Arduino reset
        # it also ensures that any bytes left over from a previous message are discarded

        msg = ''
        while msg.find('Arduino Ready') == -1:
            # then wait until an end marker is received from the Arduino to make sure it is ready to proceed
            x = 'z'
            while ord(x) != self.end_marker:  # gets the initial debugMessage
                x = self.ser.read()
                msg = msg + x.decode(encoding='ascii')

            self.display_debug(msg)


if __name__ == '__main__':
    comm = Communication("COM4", 9600, .5)
    tst = '15000' + chr(comm.special_byte) + '00000' + chr(comm.special_byte) + '00001'
    comm.wait_for_arduino()
    print('Arduino is ready')
    waiting_for_reply = False

    while comm.is_open():
        if comm.in_waiting() == 0 and not waiting_for_reply:
            comm.send_to_arduino(tst)
            print('-------sent from PC--------')
            print('BYTES SENT -> ' + comm.bytes2str(tst))
            print('TEST STR ' + tst)
            waiting_for_reply = True

        if comm.in_waiting():
            # comm.readline()
            data_recvd = comm.recv_from_arduino()
            if data_recvd[0] == 0:
                comm.display_data(data_recvd[1])

            if data_recvd[0] > 0:
                print(data_recvd)
                comm.display_data(data_recvd[1])
                print('Reply Received')
                waiting_for_reply = False
                break

            time.sleep(0.3)

    comm.close()
