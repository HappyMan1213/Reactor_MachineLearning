import os
import subprocess
import time

#————————————————————————————前处理函数————————————————————————————#
#编写.i替换函数
def replace_data(Demo_File, data1_file):
    # 获取输入目录下的文件
    files = os.listdir(Demo_File)
    print(files)
    Step1 = 0
    # 遍历每个文件并处理
    for file in files:
        # 仅处理.i文件
        if file.endswith('.i'):
            Step1 = Step1 + 1
            Step2 = 1
            
            #需要替换的参数
            #key_1 饱和蒸汽流量
            key_1 = "4440201  0.  0.  0.034  0." 
            #key_2 饱和蒸汽流量 
            key_2 = "4440202  10.0 0.0 0.034 0.0"

            # 读取写入修改数据文件
            with open(data1_file, 'r') as f:
                data1 = f.read()
            folder_path = 'F:\CCFL_LSTM\Data_Generate\Demo_Input'  # 指定文件夹路径
            file_path = os.path.join(folder_path, file)  # 构建完整的文件路径
            # 打开输入文件
            with open(file_path , 'r', encoding='utf-8') as f:
                demo = f.read()

            # 遍历数据文件的每一行
            for line in data1.splitlines():
                #检测是否停止    
                if line == "END":
                    Step2 = 1
                    break
                # 替换输入文件中的特定数据
                else:
                    value_1 = "4440201  0.  0.  " + line + "  0."
                    demo_1 = demo.replace(key_1,value_1)
                    value_2 = "4440202  10.0 0.0 "+ line + " 0.0"
                    demo_1 = demo_1.replace(key_2,value_2)
                    # 创建一个新的输出文件，写入替换后的内容
                    input_file_path = 'F:\CCFL_LSTM\Data_Generate\Input_File\CCFL' + str(Step1) +"_"+str(Step2) + ".i"
                    with open(input_file_path, 'w', encoding='utf-8') as f:
                        f.write(demo_1)
                    Step2 = Step2 + 1
        else:
                    print('ERROR_PROCESS')
    print("All Input_Files has been processed")

#————————————————————————————执行函数————————————————————————————#

#编写进程关闭函数
def kill_process(process_name):
    command = f"taskkill /F /IM {process_name}"
    subprocess.run(command, shell=True)

#编写cmd键入函数
def run_cmd(cmd, folder_path):
    subprocess.Popen(cmd, cwd=folder_path, shell=True)


#编写批量删除文件函数
def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def batch_relap_files(relap_path , Restart_File_path):
    bat = 'relap5 -i F:\CCFL_LSTM\Data_Generate\Input_File\CCFL1_1.i -o F:\CCFL_LSTM\Data_Generate\A\CCFL1_1.o -r F:\CCFL_LSTM\Data_Generate\Restart_File\CCFL1_1.r'
    key = '1_1'
    process_name = "relap5.exe"

    for i in range(1,7):
        for j in range(1,7):
            target = str(i)+'_'+str(j)
            command = bat.replace(key,target)
            
            #执行relap
            run_cmd(command, relap_path)
            time.sleep(5)
            
            #调用关闭进程把relap5程序kill
            kill_process(process_name)
            time.sleep(1)
            delete_files_in_folder(Restart_File_path)


#编写批处理文件    
def read_all_filenames(folder_path):
    filenames = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            filenames.append(file_name)
    return filenames



# 指定输入目录和数据文件
Demo_File_path = 'F:\CCFL_LSTM\Data_Generate\Demo_Input'
Data1_File_path = 'F:\CCFL_LSTM\Data_Generate\Data_File\Data1.txt'

Input_File_path = 'F:\CCFL_LSTM\Data_Generate\Input_File'
Restart_File_path = 'F:\CCFL_LSTM\Data_Generate\Restart_File'

relap_path = "F:\CCFL_LSTM\Data_Generate"

# 批量生成输入文件
replace_data(Demo_File_path, Data1_File_path)
# 批量运行文件
batch_relap_files(relap_path ,Restart_File_path)
