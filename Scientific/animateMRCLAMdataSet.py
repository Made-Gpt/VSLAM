import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import cv2


class MRCLADataSlam:
    n_robots = 5  # 机器人数量

    def __init__(self,
                 Robot1_GroundTruth=np.empty([0, 0]), Robot1_Odometry=np.empty([0, 0]),
                 Robot1_Measurement=np.empty([0, 0]),
                 Robot2_GroundTruth=np.empty([0, 0]), Robot2_Odometry=np.empty([0, 0]),
                 Robot2_Measurement=np.empty([0, 0]),
                 Robot3_GroundTruth=np.empty([0, 0]), Robot3_Odometry=np.empty([0, 0]),
                 Robot3_Measurement=np.empty([0, 0]),
                 Robot4_GroundTruth=np.empty([0, 0]), Robot4_Odometry=np.empty([0, 0]),
                 Robot4_Measurement=np.empty([0, 0]),
                 Robot5_GroundTruth=np.empty([0, 0]), Robot5_Odometry=np.empty([0, 0]),
                 Robot5_Measurement=np.empty([0, 0])):
        self.Robot1_GroundTruth = Robot1_GroundTruth
        self.Robot2_GroundTruth = Robot2_GroundTruth
        self.Robot3_GroundTruth = Robot3_GroundTruth
        self.Robot4_GroundTruth = Robot4_GroundTruth
        self.Robot5_GroundTruth = Robot5_GroundTruth
        self.Robot1_Odometry = Robot1_Odometry
        self.Robot2_Odometry = Robot2_Odometry
        self.Robot3_Odometry = Robot3_Odometry
        self.Robot4_Odometry = Robot4_Odometry
        self.Robot5_Odometry = Robot5_Odometry
        self.Robot1_Odometry = Robot1_Measurement
        self.Robot2_Odometry = Robot2_Measurement
        self.Robot3_Odometry = Robot3_Measurement
        self.Robot4_Odometry = Robot4_Measurement
        self.Robot5_Odometry = Robot5_Measurement

    # 载入数据
    def load15_MRCLAM_DataSet(self, i):
        # 定义列表，读取数据，每轮循环重置
        time = []
        x = []
        y = []
        theta = []
        v = []
        w = []
        barcode_num = []
        r = []
        b = []

        # 读取真值
        # print('Reading robot ' + str(i) + ' ground_truth')  # 打印
        # 读取.dat文件
        with open('Robot' + str(i) + '_GroundTruth.dat', 'r') as f:
            for j in f.readlines():
                if '#' not in j:
                    k = j.strip('\t').split()
                    time.append(float(k[0]))
                    x.append(float(k[1]))
                    y.append(float(k[2]))
                    theta.append(float(k[3]))
        exec('self.Robot' + str(i) +
             '_GroundTruth = np.transpose(np.array([np.array(time), np.array(x), np.array(y), np.array(theta)], '
             'dtype = object))')
        # print(eval('self.Robot' + str(i) + '_GroundTruth'))
        '''
        print(f"the row number of {'Robot' + str(i) + '_GroundTruth'} is "
              f"{eval('self.Robot' + str(i) + '_GroundTruth.shape[0]')} "
              f"and the column number is {eval('self.Robot' + str(i) + '_GroundTruth.shape[1]')}")
              '''
        time = []

        # 读取里程计值
        # print('Reading robot ' + str(i) + ' odometry')  # 打印
        # 读取.dat文件
        with open('Robot' + str(i) + '_Odometry.dat', 'r') as f:
            for j in f.readlines():
                if '#' not in j:
                    k = j.strip('\t').split()
                    time.append(float(k[0]))
                    v.append(float(k[1]))
                    w.append(float(k[2]))
        exec('self.Robot' + str(i) +
             '_Odometry = np.transpose(np.array([np.array(time), np.array(v), np.array(w)], dtype = object))')
        # print(eval('self.Robot' + str(i) + '_Odometry'))
        '''
        print(f"the row number of {'Robot' + str(i) + '_Odometry'} is "
              f"{eval('self.Robot' + str(i) + '_Odometry.shape[0]')} "
              f"and the column number is {eval('self.Robot' + str(i) + '_Odometry.shape[1]')}")
              '''
        time = []

        # 读取测量计值
        # print('Reading robot ' + str(i) + ' measurements')  # 打印
        # 读取.dat文件
        with open('Robot' + str(i) + '_Measurement.dat', 'r') as f:
            for j in f.readlines():
                if '#' not in j:
                    k = j.strip('\t').split()
                    time.append(float(k[0]))
                    barcode_num.append(float(k[1]))
                    r.append(float(k[2]))
                    b.append(float(k[3]))
        exec('self.Robot' + str(i) +
             '_Measurement = np.transpose(np.array([np.array(time,), np.array(barcode_num), np.array(r), '
             'np.array(b)], dtype = object))')
        '''
        print(f"the row number of {'Robot' + str(i) + '_Measurement'} is "
              f"{eval('self.Robot' + str(i) + '_Measurement.shape[0]')} "
              f"and the column number is {eval('self.Robot' + str(i) + '_Measurement.shape[1]')}")
              '''

    # 载入数据
    def load17_MRCLAM_DataSet(self):
        print('Parsing Dataset')  # 打印

        print('Reading barcode numbers')  # 打印
        subject_num = []
        barcode_num = []
        # 读取.dat文件
        with open('Barcodes.dat', 'r') as f:
            for i in f.readlines():
                if '#' not in i:
                    j = i.strip('\t').split()  # 逐行读取Barcodes.dat中的数据
                    subject_num.append(float(j[0]))  # 将数据存入列表
                    barcode_num.append(float(j[1]))  # 将数据存入列表
        Barcodes = np.transpose(np.array([np.array(subject_num), np.array(barcode_num)]))  # 列表转数组 -- 数组拼接 -- 数组转置

        print('Reading landmark ground_truth')  # 打印
        subject_num = []
        x = []
        y = []
        x_sd = []
        y_sd = []
        # 读取.dat文件
        with open('Landmark_GroundTruth.dat', 'r') as f:
            for i in f.readlines():
                if '#' not in i:
                    j = i.strip('\t').split()  # 逐行读取Landmark_GroundTruth.dat中的数据
                    subject_num.append(j[0])  # 将数据存入列表
                    x.append(float(j[1]))  # 将数据存入列表
                    y.append(float(j[2]))  # 将数据存入列表
                    x_sd.append(float(j[3]))  # 将数据存入列表
                    y_sd.append(float(j[4]))  # 将数据存入列表
        Landmark_GroundTruth = np.transpose(np.array([np.array(subject_num), np.array(x), np.array(y),
                                                      np.array(x_sd), np.array(y_sd)]))  # 列表转数组 -- 数组拼接 -- 数组转置
        # 对机器人真实值、测量值、里程计值进行读取
        for i in range(1, self.n_robots + 1):
            self.load15_MRCLAM_DataSet(i)

        print('Parsing Complete')
        return Barcodes, Landmark_GroundTruth

    # 对数据进行采样
    def sample_MRCLAM_DataSet(self):
        sample_time = 0.02  # 采样周期(s) Option
        min_time = self.Robot1_GroundTruth[0][0]  # 起始时间
        max_time = self.Robot1_GroundTruth[-1][0]  # 结束时间

        # 取所有机器人的最早开始时间与最晚开始时间
        for i in range(2, self.n_robots):
            exec('min_time = min(min_time, self.Robot' + str(i) + '_GroundTruth[0][0])')
            exec('min_time = max(max_time, self.Robot' + str(i) + '_GroundTruth[-1][0])')

        # 开始后经过的时间
        for i in range(1, self.n_robots):
            exec('self.Robot' + str(i) + '_GroundTruth[:,0] = self.Robot' + str(i) + '_GroundTruth[:,0] - min_time')
            exec('self.Robot' + str(i) + '_Measurement[:,0] = self.Robot' + str(i) + '_Measurement[:,0] - min_time')
            exec('self.Robot' + str(i) + '_Odometry[:,0] = self.Robot' + str(i) + '_Odometry[:,0]- min_time')
        max_time = max_time - min_time
        time_steps = math.floor(max_time / sample_time) + 1  # 向下取整

        # 打印信息
        print('time ' + str(min_time) + ' is the first timestep (t=0[s])')
        print('sampling time is ' + str(sample_time) + '[s] (' + str(1 / sample_time) + '[Hz])')
        print('number of resulting timesteps is ' + str(time_steps))

        array_names = np.array(['Robot1_GroundTruth', 'Robot1_Odometry',
                                'Robot2_GroundTruth', 'Robot2_Odometry',
                                'Robot3_GroundTruth', 'Robot3_Odometry',
                                'Robot4_GroundTruth', 'Robot4_Odometry',
                                'Robot5_GroundTruth', 'Robot5_Odometry'])
        old_data = np.zeros([0, 0])
        for name in range(len(array_names)):  # 遍历
            print('sampling ' + array_names[name])  # 正在采样
            #
            loc_old = locals()
            exec('old_data = self.' + array_names[name] + '.copy()')  # 原始数据复制
            old_data = loc_old['old_data']
            #
            # print(f"the shape of old_data: {old_data.shape}\r")
            k = 0
            t = 0
            i = 0
            p = 0
            nr = np.size(old_data, 0)  # 行数
            nc = np.size(old_data, 1)  # 列数
            new_data = np.zeros([time_steps, nc])
            while t <= max_time:  # 未超时
                new_data[k][0] = t
                while old_data[i][0] <= t:
                    if i == nr:
                        break
                    i = i + 1
                if i == 0 or i == nr:
                    if 'Odo' in array_names[name]:
                        new_data[k][1:] = old_data[k][1:]
                    else:
                        new_data[k][1:] = 0
                else:
                    p = (t - old_data[i - 1][0]) / (old_data[i][0] - old_data[i - 1][0])
                    if nc == 8:  # i.e. ground truth data
                        sc = 3
                        new_data[k][1] = old_data[i][1]  # keep id number
                    else:
                        sc = 2
                    for c in range(sc, nc):
                        if nc == 8 and c >= 6:
                            d = old_data[i][c] - old_data[i - 1][c]
                            if d > math.pi:
                                d = d + 2 * math.pi
                            elif d < -math.pi:
                                d = d + 2 * math.pi
                            new_data[k][c] = p * d + old_data[i - 1][c]
                        else:
                            new_data[k][c] = p * (old_data[i][c] - old_data[i - 1][c]) + old_data[i - 1][c]
                k = k + 1
                t = t + sample_time
            #
            loc_new = locals()
            exec('self.' + array_names[name] + ' = new_data.copy()')  # 新数据复制
            new_data = loc_new['old_data']
            print(f"the shape of new_data: {new_data.shape}\r")
            #

        array_names = np.array(['Robot1_Measurement',
                                'Robot2_Measurement',
                                'Robot3_Measurement',
                                'Robot4_Measurement',
                                'Robot5_Measurement'])
        old_data = np.zeros([0, 0])
        for name in range(len(array_names)):  # 遍历
            print('prosessing ' + array_names[name])  # 正在采样
            #
            loc_old = locals()
            exec('old_data = self.' + array_names[name] + '.copy()')  # 原始数据复制
            old_data = loc_old['old_data']
            #
            new_data = old_data.copy()
        for i in range(len(old_data)):
            new_data[i][0] = math.floor(old_data[i][0] / sample_time + 0.5) * sample_time
        #
        loc_new = locals()
        exec('self.' + array_names[name] + ' = new_data.copy()')  # 新数据复制
        new_data = loc_new['old_data']
        #

        return time_steps, sample_time

    # 建图
    def animate_MRCLAM_DataSet(self, Barcodes, Landmark_GroundTruth, time_steps, sample_time):
        read_slam = MRCLADataSlam()  # 读入数据集

        start_time_step = 1
        end_time_step = time_steps
        time_steps_per_frame = 50
        pause_time_between_frames = 0.0122
        draw_measurements = 0

        n_landmarks = len(Landmark_GroundTruth[:, 0])

        # Plots and Figure Setup
        colour = [[1, 0, 0], [0, 0.75, 0], [0, 0, 1], [1, 0.50, 0.25], [1, 0.5, 1]]
        for i in range(5, 5 + n_landmarks):
            colour.append([0.3, 0.3, 0.3])
        colour = np.array(colour)

        figHandle = plt.figure('Name', 'Dataset GroundTruth', 'Renderer', 'OpenGL')
        plt.setp(plt.gcf(), 'Position', [1300, 1, 630, 950])
        plotHandles_robot_gt = np.zeros([self.n_robots, 1])
        plotHandles_landmark_gt = np.zeros([n_landmarks, 1])

        # ?
        r_robot = 0.165
        d_robot = 2 * r_robot
        r_landmark = 0.055
        d_landmark = 2 * r_landmark

        # initial set up
        Robot = np.array([self.Robot1_GroundTruth[:, 1:3], self.Robot2_GroundTruth[:, 1:3],
                          self.Robot3_GroundTruth[:, 1:3], self.Robot4_GroundTruth[:, 1:3],
                          self.Robot5_GroundTruth[:, 1:3]])
        n_measurements = np.empty([0, 0])
        for i in range(self.n_robots):
            x = Robot[0, (i + 1) * 3 - 2]  # 机器人x 坐标
            y = Robot[0, (i + 1) * 3 - 1]  # 机器人y 坐标
            z = Robot[0, (i + 1) * 3]  # 机器人z 坐标
            x1 = d_robot * math.cos(z) + x
            y1 = d_robot * math.sin(z) + y
            p1 = x - r_robot
            p2 = y - r_robot
            plotHandles_robot_gt[i] = plt.Rectangle('Position', [p1, p2, d_robot, d_robot], 'Curvature', [1, 1],
                                                    'FaceColor', colour[i, :], 'LineWidth', 1)
            plt.plot([x, x1], [y, y1], 'k')
            exec(['n_measurements(i) = length(Robot' + str(i) + '_Measurement)'])
        for i in range(n_landmarks):
            exec('x = Landmark_GroundTruth[' + str(i) + ', 2]')
            exec('y = Landmark_GroundTruth[' + str(i) + ', 3]')
            p1 = x - r_landmark
            p2 = y - r_landmark
            plotHandles_landmark_gt[i] = plt.Rectangle('Position', [p1, p2, d_landmark, d_landmark], 'Curvature', [1, 1],
                                                       'FaceColor', colour[i + 5, :],'LineWidth', 1)

        squire = plt.axis
        equal = plt.axis
        plt.axis([-2, 6, -6, 7])
        plt.setp(plt.gca(), 'xTick', np.array(range(-10, 2, 10).T))

        # Going through data
        measurement_time_index = np.ones([self.n_robots, 1]) # index of last measurement processed
        barcode = 0
        tempIndex = [0]
        for i in range(self.n_robots):
            exec("tempIndex = np.where(self.Robot" + str(i) + "_Measurement(:, 0) >= start_time_step * sample_time, 0, 'first')")
            if tempIndex != []:
                measurement_time_index[i] = tempIndex
            else:
                measurement_time_index[i] = n_measurements[i] + 1

        for k in range(start_time_step, end_time_step):
            t = k * sample_time

            if k % self.n_robots == 0:
                now_gca = plt.gca()
                # del plt.findobj('type', 'line')

            for i in range(self.n_robots):
                x[i] = Robot[k, i * 3 - 2]
                y[i] = Robot[k, i * 3 - 1]
                z[i] = Robot[k, i * 3]

            if k % time_steps_per_frame == 0:
                x1 = d_robot * math.cos(z[i]) + x[i]
                y1 = d_robot * math.cos(z[i]) + y[i]
                p1 = x(i) - r_robot
                p2 = y(i) - r_robot
                plt.setp(plotHandles_robot_gt[i], 'Position', [p1, 1, 630, 950])
                plt.plot([x[i], x1], [y[i], y1], 'k')

            if draw_measurements:
                measure_id = 0
                measure_r = 0
                measure_b = 0
                while n_measurements[i] >= measurement_time_index[i] and eval("self.Robot" + str(i) + "_Measurement["
                                                                            + str(measurement_time_index[i]) + ", 0] <= t"):
                    exec("measure_id = self.Robot" + str + "_Measurement[" + str(measurement_time_index[i]) + ", 1]")
                    exec("measure_r = self.Robot" + str + "_Measurement[" + str(measurement_time_index[i]) + ", 2]")
                    exec("measure_b = self.Robot" + str + "_Measurement[" + str(measurement_time_index[i]) + ", 3]")
                    landmark_index = np.where(Barcodes[:, 1] == measure_id)
                    if landmark_index == []:
                        x1 = x[i] + measure_r * math.cos(measure_b + z(i))
                        y1 = y[i] + measure_r * math.sin(measure_b + y(i))
                        plt.plot([x[i], x1], [y[i], y1], color=colour[i, :], linewidth=1)
                    else:
                        robot_index = np.where(Barcodes[0:4, 1] == measure_id)
                        if robot_index == []:
                            x1 = x[i] + measure_r * math.cos(measure_b + z(i))
                            y1 = y[i] + measure_r * math.sin(measure_b + y(i))
                            plt.plot([x[i], x1], [y[i], y1], color=colour[i, :], linewidth=1)
                    measurement_time_index[i] = measurement_time_index[i] + 1

        if k % time_steps_per_frame == 0:
            # del plt.findobj('type', 'txt')
            texttime = "k = " + str(k, "%5d") + " t = " + str(t, "%5.2f") + "[s]"
            plt.text(1.5, 6.5, texttime)
            time.sleep(pause_time_between_frames)  # 暂停
        else:
            if draw_measurements:
                time.sleep(0.001)


if __name__ == "__main__":
    run_slam = MRCLADataSlam()  # 实例化对象,便于对类变量进行操作
    Barcodes, Landmark_GroundTruth = run_slam.load17_MRCLAM_DataSet()
    time_steps, sample_time = run_slam.sample_MRCLAM_DataSet()
    run_slam.animate_MRCLAM_DataSet(Barcodes, Landmark_GroundTruth, time_steps, sample_time)
