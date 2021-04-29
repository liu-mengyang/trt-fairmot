import psutil
import GPUtil
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
# import keyboard
#import seaborn as sns

class Monitor:
    """
    a computer performance monitor
    """
    def __init__(self, csv_out_file="default"):
        """
        init the monitor's members
        :param csv_out_file: the output .csv file's name without (.csv)
        """
        self.Gpus = GPUtil.getGPUs()
        self.index = 0
        self.index_lst = []
        self.cpu_info_lst = []
        self.mem_info_lst = []
        self.gpu_ulti_info_lst = []
        self.gpu_mem_info_lst = []
        self.csv_out_file = csv_out_file

    def obtain_cpu_info(self):
        """
        get the cpu ultilization every second
        update self.current_cpu: a float value
        :return: void
        """
        cpu_lst = psutil.cpu_percent(percpu=True)
        cpu = 0
        num = 0
        for percpu in cpu_lst:
            num += 1
            cpu += percpu
        cpu /= num
        print("cpu utilization ratio: ", cpu)
        self.cpu_info_lst.append(cpu)

    def obtain_mem_info(self):
        """
        get the memory ultilization every second
        update self.current_mem: a float value
        :return: void
        """
        memory = psutil.virtual_memory()
        mem = memory.percent
        print("memory size utilization ratio: ", mem)
        self.mem_info_lst.append(mem)

    def obtain_gpu_info(self):
        """
        get the gpu ultilization every second
        update self.current_gpu_lst: a gpu_info list [[gpu_id, gpu_ulti, gpu_mem], [...]]
        :return: void
        """
        self.Gpus = GPUtil.getGPUs()
        gpu_lst = []
        av_ulti = 0
        av_mem = 0
        num = 0
        for gpu in self.Gpus:
            gid = gpu.id
            if gid == 0:
                ulti = gpu.load*100
                mem = gpu.memoryUtil*100
                gpu_lst.append([gid, ulti*100, mem*100])
                print("gpu id:", gid)
                print("gpu ultilization ratio: ", ulti)
                print("gpu size ultilization ratio: ", mem)
                num += 1
                av_ulti += ulti
                av_mem += mem
        num = max(1, num)
        av_ulti /= num
        av_mem /= num
        self.gpu_ulti_info_lst.append(av_ulti)
        self.gpu_mem_info_lst.append(av_mem)

    def print_in_dim(self, inv):
        """
        print the result of all ultilization info on the screen in an interval
        :param inv: print interval
        :return: void
        """
        plt.ion()
        while True:
            # if keyboard.is_pressed('q'):
            #     print('Quit!')
            #     break
            try:
                self.index_lst.append(self.index)
                self.index += 1

                print("*******************************************")
                # get cpu info
                self.obtain_cpu_info()

                # get memory info
                self.obtain_mem_info()

                # get gpu info
                self.obtain_gpu_info()

                print("*******************************************")
                # time stop

                # self.plot_realtime()
                time.sleep(inv)
            except KeyboardInterrupt:
                monitor.write_with_csv()
                monitor.plot_all_info()
                raise
        #plt.show()

    def plot_realtime(self):
        # data = {"cpu": self.cpu_info_lst,
        #         "mem": self.mem_info_lst,
        #         "gpu_ulti": self.gpu_ulti_info_lst,
        #         "gpu_mem": self.gpu_mem_info_lst}
        # df = pd.DataFrame(data)
        # df.plt()
        # plt.clf()
        plt.plot(self.index_lst, self.cpu_info_lst, 'b', label='cpu')
        plt.plot(self.index_lst, self.mem_info_lst, 'orange', label='memory')
        plt.plot(self.index_lst, self.gpu_ulti_info_lst, 'g', label='gpu ulti')
        plt.plot(self.index_lst, self.gpu_mem_info_lst, 'r', label='gpu mem')
        plt.legend()
        plt.pause(0.05)
        plt.ioff()



    def write_with_csv(self):
        """
        write the content with csv type in an appointed file

        :return: void
        """
        # create the file
        f = open(self.csv_out_file+".csv", 'w', encoding='utf-8')

        #  create the writer
        csv_writer = csv.writer(f)

        # write the title
        csv_writer.writerow(["cpu", "mem", "gpu_ulti", "gpu_mem"])

        # write the content
        info_lst_len = len(self.cpu_info_lst)
        i = 0
        while i < info_lst_len:
            csv_writer.writerow([format(self.cpu_info_lst[i], ".2f"), format(self.mem_info_lst[i], ".2f"),
                                 format(self.gpu_ulti_info_lst[i], ".2f"), format(self.gpu_mem_info_lst[i], ".2f")])
            i += 1
        # close the file
        f.close()

    def plot_all_info(self):
        df = pd.read_csv(self.csv_out_file + ".csv")
        df.plot()
        # plt.show()
        plt.savefig("output.png")


if __name__=='__main__':
    interval = 0.1
    monitor = Monitor()
    monitor.print_in_dim(interval)
    # monitor.write_with_csv()
    # monitor.plot_all_info()

