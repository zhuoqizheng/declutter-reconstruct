import glob
import os
import shutil
import datetime
import tabulate
import numpy as np

import parameters as par
import minibatch

import FFT_MQ as fft

'''SIMPLIFIED main.py for single action layout prediction'''

def make_folder(location, folder_name):
    if not os.path.exists(location + '/' + folder_name):
        os.mkdir(location + '/' + folder_name)


def check_int(name_folder):
    if not os.path.exists(name_folder):
        os.mkdir(name_folder)
    maps = os.listdir(name_folder)
    while True:
        index = 0
        for image in maps:
            print(str(index) + '   ' + image)
            index += 1
        try:
            val = int(input('insert the number you want to select\n'))
            if 0 <= val < index:
                return maps[val]
        except ValueError:
            print('invalid input')


def main():
    # ----------------PARAMETERS OBJECTS------------------------
    # loading parameters from parameters.py
    parameters_object = par.ParameterObj()
    # loading path object with all path and name interesting for code
    paths = par.PathObj()
    # taking all the folders inside the path INPUT/IMGs
    list_dir = '/home/zzq/Documents/declutter-reconstruct/code/data/INPUT'
    # asking the user what folder want to use
    paths.name_folder_input = check_int(list_dir)

    paths.path_folder_input = list_dir + '/' + paths.name_folder_input

    if paths.name_folder_input == 'Bormann' or paths.name_folder_input == 'Bormann_furnitures':
        parameters_object.bormann = True
    else:
        parameters_object.bormann = False


    # saving the output folder where the output is saved
    make_folder('/home/zzq/Documents/declutter-reconstruct/code/data/OUTPUT', 'SINGLEMAP')
    paths.path_folder_output = '/home/zzq/Documents/declutter-reconstruct/code/data/OUTPUT/SINGLEMAP'

    # asking what map to use
    paths.metric_map_name = check_int(paths.path_folder_input)
    paths.metric_map_path = os.path.join(paths.path_folder_input + '/' + paths.metric_map_name)
    running_time = str(datetime.datetime.now())[:-7].replace(' ', '@')

    paths.path_log_folder = os.path.join(paths.path_folder_output, paths.metric_map_name)
    make_folder(paths.path_folder_output, paths.metric_map_name)
    make_folder(paths.path_log_folder, running_time)
    paths.filepath = paths.path_log_folder + '/' + running_time + '/'
    # copying the parameters file
    shutil.copy('/home/zzq/Documents/declutter-reconstruct/code/parameters.py', paths.path_log_folder + '/' + running_time + '/parameters.py')
    # ----------------------------------------------------------------
    # orebro
    make_folder(paths.path_log_folder, running_time + '/OREBRO')
    paths.path_orebro = paths.path_log_folder + '/' + running_time + '/OREBRO'
    paths.orebro_img = paths.filepath + 'OREBRO_' + str(parameters_object.filter_level) + '.png'
    fft.main(paths.metric_map_path, paths.path_orebro, parameters_object.filter_level, parameters_object)
    # ----------------------------------------------------------------
    # evaluation
    paths.gt_color = '/home/zzq/Documents/declutter-reconstruct/code/data/INPUT/gt_colored/' + paths.name_folder_input + '/' + paths.metric_map_name
    # ----------------------------------------------------------------
    # starting main
    print('map name ', paths.metric_map_name)
    minibatch.start_main(par, parameters_object, paths)

    # -------------------------------ENDING EXECUTION AND EVALUATION TIME------------------------------------


if __name__ == '__main__':
    main()
