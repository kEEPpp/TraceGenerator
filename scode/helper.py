import copy
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from AutoModelerLoader import AutoModelLoader
from AutoModelerCleaner import AutoModelCleaner
from AutoModelerCluster import AutoModelCluster
from AutoModelerHough import AutoModelHough
from AutoModelerSegmenter import AutoModelSegmenter
from AutoModelerAngleDetector import AutoModelAngleDetector

with open(os.path.join('..', 'config.json')) as f:
    config = json.load(f)

def reset_index(trace, step_list, order_value='identifier', return_recipe_step_boundary=True):
    full_trace_list = []
    for i in trace[order_value].unique():
        d = trace[trace[order_value] == i].reset_index(drop=True)
        full_trace_list.append(d)

    # step_list = cleaned_data.RECIPE_STEP.unique()
    col_name = trace.columns.to_list()
    group_df = trace.groupby(order_value)
    recipe_step_index_list = []

    for step_number in step_list:
        len_max = max((df['RECIPE_STEP'] == step_number).sum() for _, df in group_df)

        recipe_step_index_list.append(len_max)
        for key, f in enumerate(full_trace_list):
            step_length = (f['RECIPE_STEP'] == step_number).sum()
            if step_length < len_max:
                if step_length == 0:
                    new_index = np.cumsum(recipe_step_index_list)[-1]  # 원래는 -2인데 뭐가 맞지..?
                else:
                    new_index = f[f['RECIPE_STEP'] == step_number].index.values[-1] + 1
                for i in range(len_max - step_length):
                    #if (key == 8 or key == 7) and step_number == '4':
                        #print(f'key:{key}\nstep_length:{step_length}\nlen_max:{len_max}\nnew_index:{new_index}\ni:{i}\n')
                        #print(display(full_trace_list[key]))
                    try:
                        full_trace_list[key] = pd.DataFrame(np.insert(full_trace_list[key].values,
                                                                    new_index + i,
                                                                    values=[np.nan],
                                                                    axis=0),
                                                            columns=col_name)
                        full_trace_list[key].loc[new_index + i, 'RECIPE_STEP'] = step_number
                    except:
                        pass

    for key, f in enumerate(full_trace_list):
        full_trace_list[key] = f.dropna(subset=[order_value])

    accumulated_index = np.cumsum(recipe_step_index_list)
    accumulated_index[-1] = accumulated_index[-1] - 1
    if return_recipe_step_boundary:
        return full_trace_list, accumulated_index
    else:
        return full_trace_list

def show_index_dist(cleaned_data, global_stat, recipe_step):
    cluster = AutoModelCluster(cleaned_data, global_stat, recipe_step=recipe_step)
    target_data = cluster.get_clustered_target_data()

    segmenter = AutoModelSegmenter(config, global_stat, target_data)

    norm_wafer, norm_segmented_wafer = segmenter.get_data()
    detector = AutoModelAngleDetector(norm_wafer, norm_segmented_wafer)
    detector.point_calculation('index', True)

def find_angle(seg):
    reg_stable = seg.copy()
    x = reg_stable.index.values.reshape(-1, 1)
    y = reg_stable.PARAMETER_VALUE.values.reshape(-1, 1)

    reg = LinearRegression(fit_intercept=True).fit(x, y)
    degree = np.degrees(np.arctan(reg.coef_[0]))[0]

    return degree

def angle_type_check(cleaned_data, global_stat, result, recipe_step):
    cluster = AutoModelCluster(cleaned_data, global_stat, recipe_step=recipe_step)
    target_data = cluster.get_clustered_target_data()

    segmenter = AutoModelSegmenter(config, global_stat, target_data, version=3)

    norm_wafer, norm_segmented_wafer = segmenter.get_data()
    #return norm_segmented_wafer
    #detector = AutoModelAngleDetector(norm_wafer, norm_segmented_wafer)

    angle_list = []
    for i in norm_segmented_wafer:
        angle_list.append(find_angle(i))
    angle_list = np.array(angle_list)

    n_angle_stable_boolean = abs(angle_list) < 10
    n_angle_up_boolean = angle_list >= 10
    n_angle_down_boolean = angle_list <= -10

    n_angle_stable = angle_list[n_angle_stable_boolean]
    n_angle_up = angle_list[n_angle_up_boolean]
    n_angle_down = angle_list[n_angle_down_boolean]
    #return angle_list
    plt.figure(figsize=(6, 3))

    plt.hist(([n_angle_stable, n_angle_up, n_angle_down]),
             edgecolor='black',
             bins=30,
             stacked=True,
             color=['red', 'orange', 'blue'],
             label=[f'stable({n_angle_stable_boolean.sum()})', f'up({n_angle_up_boolean.sum()})',
                    f'down({n_angle_down_boolean.sum()})'])
    plt.title(f"regression - normalization - angle")

    plt.legend()
    plt.show()

    #print(f"reciep step{recipe_step} type: {result[recipe_step]['type']}")


def cleaning_process_check(reset_raw_data, reset_cleaned_data, boundary_raw, boundary_clean):
    set_raw = set(pd.concat(reset_raw_data).identifier)
    set_clean = set(pd.concat(reset_cleaned_data).identifier)
    diff = set_raw.difference(set_clean)

    if diff:
        print(f"{list(diff)} are removed")
    else:
        print('all raw data is used')
    fig, ax = plt.subplots(1, 2, figsize=(24, 6))

    ax[0].set_title(f'raw data: # of {len(reset_raw_data)}')
    for i in reset_raw_data:
        if set(i.identifier).intersection(diff):
            label = i.identifier.unique()[0]
            r = random.random()
            g = random.random()
            b = random.random()
            color = (r, g, b)
            ax[0].plot(i.PARAMETER_VALUE, label=label, color=color, linewidth=3, linestyle='--')
        else:
            ax[0].plot(i.PARAMETER_VALUE, label=None)

    ax[1].set_title(f'cleaned data # of {len(reset_cleaned_data)}')
    for i in reset_cleaned_data:
        ax[1].plot(i.PARAMETER_VALUE)

    for i, j in zip(boundary_raw, boundary_clean):
        ax[0].axvline(i, linestyle='--', alpha=0.3, color='black')
        ax[1].axvline(j, linestyle='--', alpha=0.3, color='black')

    ax[0].legend()
    plt.show()

def prime_trace_check(cleaned_data, global_stat, result, step_list, boundary_clean):
    col_name = cleaned_data.columns
    reset_index_new = np.insert(boundary_clean, 0, 0)
    reset_index = boundary_clean

    group_df = cleaned_data.groupby('identifier')
    prime_list = []
    index_dict = {}
    target_data_list = []
    for step, v in result.items():
        index_temp = {'start': -1,
                      'end': -1}
        # prime wafer extraction
        prime_wafer_identifier = v['prime']
        prime_wafer = group_df.get_group(prime_wafer_identifier)
        prime_wafer = prime_wafer.loc[prime_wafer.RECIPE_STEP == step, :]
        prime_list.append(prime_wafer)

        start, end = v['start_index'], v['end_index']
        index_temp['start'], index_temp['end'] = start, end
        index_dict[step] = index_temp

        # major cluster
        cluster = AutoModelCluster(cleaned_data, global_stat, recipe_step=step)
        target_data = cluster.get_clustered_target_data()
        target_data_list.append(set(target_data.identifier))

    prime_list = pd.concat(prime_list).reset_index(drop=True)

    ## 경계선
    clean_group = cleaned_data.groupby('identifier')
    major_cluster_list = []
    minor_cluster_list = []
    for key, step in enumerate(step_list):
        major_temp = []
        minor_temp = []
        for identifier, df in clean_group:
            df = df.loc[df.RECIPE_STEP == step, :]
            if identifier in target_data_list[key]:
                major_temp.append(df)
            else:
                minor_temp.append(df)

        major_temp = pd.concat(major_temp)
        if minor_temp:
            minor_temp = pd.concat(minor_temp)
        else:
            minor_temp = pd.DataFrame([], columns=col_name)

        major_cluster_list.append(major_temp)
        minor_cluster_list.append(minor_temp)

    sorted_list = []

    for step, _ in enumerate(step_list):
        df = major_cluster_list[step]  # 해당 단계의 데이터프레임 가져오기
        temp_sort = []
        for identifier, group_df in df.groupby('identifier'):  # identifier로 그룹화
            group_df.reset_index(drop=True)
            group_df.index = np.arange(reset_index_new[step], len(group_df) + reset_index_new[step])  # 인덱스 업데이트
            temp_sort.append(group_df)  # 업데이트된 데이터프레임을 sorted_list에 추가
        temp_sort = pd.concat(temp_sort)
        sorted_list.append(temp_sort)

    minor_sorted_list = []

    for step, _ in enumerate(step_list):
        df = minor_cluster_list[step]  # 해당 단계의 데이터프레임 가져오기
        temp_sort = []
        for identifier, group_df in df.groupby('identifier'):  # identifier로 그룹화
            group_df.reset_index(drop=True)
            group_df.index = np.arange(reset_index_new[step], len(group_df) + reset_index_new[step])  # 인덱스 업데이트
            temp_sort.append(group_df)  # 업데이트된 데이터프레임을 sorted_list에 추가
        if temp_sort:
            temp_sort = pd.concat(temp_sort)
        else:
            temp_sort = pd.DataFrame([], columns=col_name)
        minor_sorted_list.append(temp_sort)

    # prime index reset
    prime = []
    for k, step in enumerate(step_list):
        p = prime_list.loc[prime_list.RECIPE_STEP == step, :]
        p = p.reset_index(drop=True)
        p.index = np.arange(reset_index_new[k], len(p) + reset_index_new[k])
        prime.append(p)
    prime = pd.concat(prime)

    # draw figure
    plt.figure(figsize=(12, 6))
    label_added = False
    for recipe_df in sorted_list:
        df = recipe_df.groupby('identifier')
        for k, v in df:
            if not label_added:
                plt.plot(v.PARAMETER_VALUE, color='blue', alpha=0.1, label='major cluster')
                label_added = True
            else:
                plt.plot(v.PARAMETER_VALUE, color='blue', alpha=0.1, label=None)

    label_added = False
    for step, v in result.items():
        start, end = v['start_index'], v['end_index']
        start, end = int(start), int(end)
        # new_prime = prime.groupby('RECIPE_STEP')
        new_prime = prime.loc[prime.RECIPE_STEP == step].PARAMETER_VALUE

        if not label_added:
            #plt.plot(new_prime.iloc[:start], color='purple', linewidth=5)
            plt.plot(new_prime.iloc[start:end + 1], color='black', linewidth=10, label='auto spec', alpha=0.6, zorder=100)
            #plt.plot(new_prime.iloc[end + 1:], color='purple', linewidth=3, label='prime reference')
            label_added = True
        else:
            #plt.plot(new_prime.iloc[:start], color='purple', linewidth=3)
            plt.plot(new_prime.iloc[start:end + 1], color='black', linewidth=10, label=None, alpha=0.6)
            #plt.plot(new_prime.iloc[end + 1:], color='purple', linewidth=3, label=None)

    plt.plot(prime.PARAMETER_VALUE, color='purple', linewidth=3, label='prime reference')

    label_added = False
    for recipe_df in minor_sorted_list:
        df = recipe_df.groupby('identifier')
        for k, v in df:
            if not label_added:
                plt.plot(v.PARAMETER_VALUE, color='red', linewidth=1, label='minor cluster', alpha=0.4)
                label_added = True
            else:
                plt.plot(v.PARAMETER_VALUE, color='red', linewidth=1, label=None, alpha=0.4)

    for ver in reset_index:
        plt.axvline(ver, color='black', linestyle='--', alpha=0.3, linewidth=2)

    for step, v in result.items():
        print(
            f"recipe step: {step}\nspec start/end index: [{v['start_index']}, {v['end_index']}]\ntime: {v['type']}\nprime wafer: {v['prime']}\n")
    plt.legend()
    plt.show()

def clustering_result_check(cleaned_data, global_stat, reset_cleaned_data, boundary_clean, recipe_step):
    #ordered_wafer = loader.reset_index(cleaned_data, step_list)
    ordered_wafer = reset_cleaned_data
    ordered_wafer = pd.concat(ordered_wafer)
    cluster = AutoModelCluster(cleaned_data, global_stat, recipe_step=recipe_step)
    target_data = cluster.get_clustered_target_data()

    segmenter = AutoModelSegmenter(config, global_stat, target_data)
    hough_wafer, hough_seg = segmenter.get_data()
    hough_angle = AutoModelAngleDetector(hough_wafer, hough_seg)

    all_wafer = set(cleaned_data.identifier)
    cluster_wafer = set(target_data.identifier)

    major = list(all_wafer.intersection(cluster_wafer))
    minor = list(all_wafer.difference(cluster_wafer))
    if minor:
        print(f'non-major cluster list:{minor}')
    else:
        print(f'all data is used major cluster')

    plt.figure(figsize=(24, 8))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax1.set_title('cleaning data')
    group_data = ordered_wafer.groupby('identifier')
    for identifier, df in group_data:
        value = df.PARAMETER_VALUE
        ax1.plot(value)

    for b in boundary_clean:
        ax1.axvline(b, color='black', alpha=0.3, linewidth=2, linestyle='--')

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    group_data = cleaned_data.groupby('identifier')

    index_point = hough_angle.get_point()[0]

    ax2.set_title(f"step{recipe_step}\nspec start{index_point['start']}, end{index_point['end']}")
    for k, m in enumerate(major):
        value = group_data.get_group(m)
        value = value.loc[value.RECIPE_STEP == recipe_step, 'PARAMETER_VALUE'].reset_index(drop=True)
        label = 'major' if k == 0 else None
        ax2.plot(value, color='blue', label=label, alpha=0.6)

    for k, m in enumerate(minor):
        value = group_data.get_group(m)
        value = value.loc[value.RECIPE_STEP == recipe_step, 'PARAMETER_VALUE'].reset_index(drop=True)
        label = 'non-major' if k == 0 else None
        ax2.plot(value, color='r', marker='o', label=label)

    if index_point['start'] != -1:
        ax2.axvline(index_point['start'], color='purple', linestyle='--', alpha=0.8, label='segment')
        ax2.axvline(index_point['end'], color='purple', linestyle='--', alpha=0.8)
    print(f"start: {index_point['start']}\nend: {index_point['end']}\ntype: {index_point['type']}")
    plt.legend()
    plt.show()

def clustering_result_check_old(cleaned_data, global_stat, ordered_wafer, boundary, recipe_step):
    cluster = AutoModelCluster(cleaned_data, global_stat, recipe_step=recipe_step)
    target_data = cluster.get_clustered_target_data()

    segmenter = AutoModelSegmenter(config, global_stat, target_data)
    hough_wafer, hough_seg = segmenter.get_data()
    hough_angle = AutoModelAngleDetector(hough_wafer, hough_seg)

    all_wafer = set(cleaned_data.identifier)
    cluster_wafer = set(target_data.identifier)

    major = list(all_wafer.intersection(cluster_wafer))
    minor = list(all_wafer.difference(cluster_wafer))
    if minor:
        print(f'non-major cluster list:{minor}')
    else:
        print(f'all data is used major cluster')

    plt.figure(figsize=(24, 8))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax1.set_title('full recipe step figure')
    group_data = ordered_wafer.groupby('identifier')
    for k, m in enumerate(major):
        value = group_data.get_group(m).PARAMETER_VALUE
        ax1.plot(value)

    for b in boundary:
        ax1.axvline(b, color='black', alpha=0.3, linewidth=2)

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    group_data = cleaned_data.groupby('identifier')

    index_point = hough_angle.get_point()[0]

    ax2.set_title(f"step{recipe_step}\nspec start{index_point['start']}, end{index_point['end']}")
    for k, m in enumerate(major):
        value = group_data.get_group(m)
        value = value.loc[value.RECIPE_STEP == recipe_step, 'PARAMETER_VALUE'].reset_index(drop=True)
        label = 'major' if k == 0 else None
        ax2.plot(value, color='blue', label=label, alpha=0.6)

    for k, m in enumerate(minor):
        value = group_data.get_group(m)
        value = value.loc[value.RECIPE_STEP == recipe_step, 'PARAMETER_VALUE'].reset_index(drop=True)
        label = 'non-major' if k == 0 else None
        ax2.plot(value, color='r', marker='o', label=label)

    if index_point['start'] != -1:
        ax2.axvline(index_point['start'], color='purple', linestyle='--', alpha=0.8)
        ax2.axvline(index_point['end'], color='purple', linestyle='--', alpha=0.8)
    print(f"start: {index_point['start']}\nend: {index_point['end']}\ntype: {index_point['type']}")
    plt.legend()
    plt.show()

def show_all_trace_each_step(path, data, recipe_step: 'str'):
    p = path
    cleaned_data, global_stat = AutoModelCleaner(config, data).get_cleaning_data()
    print(p.split(os.sep)[-1].split('.')[0])
    print(f'recipe step:{recipe_step}')

    cluster = AutoModelCluster(cleaned_data, global_stat, recipe_step=recipe_step)
    target_data = cluster.get_clustered_target_data()

    segmenter = AutoModelSegmenter(config, global_stat, target_data, version=3)

    # norm_reset = copy.deepcopy(reset)
    # for key, r in enumerate(reset):
    #     norm_reset[key].loc[:, 'PARAMETER_VALUE'] = ((r.loc[:, 'PARAMETER_VALUE'].copy() - global_stat['global_min']) /global_stat['global_range']) * 100

    wafer_list = []
    for identifier in target_data.identifier.unique():
        temp_wafer = target_data[target_data.identifier == identifier].reset_index(drop=True)
        wafer_list.append(temp_wafer)

    wafer_list_norm = []
    for identifier in target_data.identifier.unique():
        temp_wafer = target_data[target_data.identifier == identifier].reset_index(drop=True)
        temp_wafer.loc[:, 'PARAMETER_VALUE'] = ((temp_wafer.loc[:, 'PARAMETER_VALUE'].copy() - global_stat['global_min']) / global_stat['global_range']) * 100
        wafer_list_norm.append(temp_wafer)


    name = p.split(os.sep)[-1].split('.')[0]
    #show_chart(target_data, step, segmenter, boun, reset, name)
    show_chart2(wafer_list, wafer_list_norm, recipe_step, global_stat, name)

def show_all_trace(path):
    p = path
    df = pd.read_csv(p)
    loader = AutoModelLoader('LOT_ID', 'WAFER_ID', 'PROCESS', 'PROCESS_STEP', 'RECIPE', 'RECIPE_STEP', 'PARAMETER_NAME',
                             'PARAMETER_VALUE', 'TIME', df)
    data = loader.data_load()
    step_list = list(np.sort(np.unique(data.RECIPE_STEP.values).astype(int)).astype('str'))
    reset, boun = loader.reset_index(step_list, return_recipe_step_boundary=True)
    cleaned_data, global_stat = AutoModelCleaner(config, data).get_cleaning_data()
    print(p.split(os.sep)[-1].split('.')[0])

    for step in step_list:
        cluster = AutoModelCluster(cleaned_data, global_stat, recipe_step=step)
        target_data = cluster.get_clustered_target_data()

        segmenter = AutoModelSegmenter(config, global_stat, target_data)

        norm_reset = copy.deepcopy(reset)
        for key, r in enumerate(reset):
            norm_reset[key].loc[:, 'PARAMETER_VALUE'] = ((r.loc[:, 'PARAMETER_VALUE'].copy() - global_stat[
                'global_min']) / global_stat['global_range']) * 100

        wafer_list = []
        for identifier in target_data.identifier.unique():
            temp_wafer = target_data[target_data.identifier == identifier].reset_index(drop=True)
            wafer_list.append(temp_wafer)

        wafer_list_norm = []
        for identifier in target_data.identifier.unique():
            temp_wafer = target_data[target_data.identifier == identifier].reset_index(drop=True)
            temp_wafer.loc[:, 'PARAMETER_VALUE'] = ((temp_wafer.loc[:, 'PARAMETER_VALUE'].copy() - global_stat[
                'global_min']) / global_stat['global_range']) * 100
            wafer_list_norm.append(temp_wafer)

        name = p.split(os.sep)[-1].split('.')[0]
        show_chart(target_data, global_stat, step, segmenter, norm_reset, boun, reset, name)
        show_chart2(wafer_list, wafer_list_norm, step, global_stat, name)

def calculate_hough(hough, ax):
    row_min = hough.row_min
    mat = hough.trace_to_matrix(hough.row_min)
    h = hough.hough_transformation(mat)
    element_rho, element_theta = hough.local_maxima(h)
    segment = hough.segmentation(mat, element_rho, element_theta)

    m = -np.cos(element_theta) / np.sin(element_theta)
    b = element_rho / np.sin(element_theta)

    row = np.where(mat == 1)[0]
    col = np.where(mat == 1)[1]

    row = row[np.argsort(col)]
    col = col[np.argsort(col)]

    detected_line = []
    x_axis = []
    for j in range(len(m)):
        for r, c in zip(row, col):
            y = m[j] * c + b[j]
            detected_line.append(y)
            x_axis.append(c)
    detected_line = np.array(detected_line)

    y_upper = detected_line + row_min * 0.05
    y_lower = detected_line - row_min * 0.05
    y_upper = np.ceil(y_upper)
    y_lower = np.floor(y_lower)

    ax.imshow(mat, cmap='gray')
    ax.imshow(mat, cmap='gray')
    ax.set_title(f"row_min={row_min}\ny={m[0]:.2f}x+{b[0]:.2f}\nangle:{np.degrees(np.arctan(m[0])):.2f}")
    ax.plot(x_axis, detected_line, color='r', marker='o')
    ax.plot(segment, detected_line[segment], color='purple', marker='o')

    return ax

def show_chart2(wafer_list, norm_wafer, step, global_stat, name):
    new_norm_wafer = pd.concat(norm_wafer).PARAMETER_VALUE
    y_max = new_norm_wafer.max()
    y_min = new_norm_wafer.min()
    y_max = y_max * 1.05
    y_min = y_min * 0.8
    if y_min == 0:
        y_min = -1

    figure_number = 1
    for key, wafer in enumerate(wafer_list):
        hough = AutoModelHough(config, wafer, global_stat, version=3)

        new_r = hough.get_data()

        if key%4 == 0:
            fig, ax = plt.subplots(2,4, figsize=(20,12))
        for_index = key%4
        ax[0, for_index].set_title(f'{name}\n{wafer.identifier.unique()[0]}\nnumber: {key+1}/{len(wafer_list)}\nstep: {step}')
        ax[0, for_index].set_ylim(y_min, y_max)
        ax[0, for_index].plot(norm_wafer[key].PARAMETER_VALUE, marker='o')
        ax[0, for_index].plot(norm_wafer[key].loc[new_r['start']:new_r['end'], 'PARAMETER_VALUE'], marker='o', color='purple', label=f"start:{new_r['start']}\nend:{new_r['end']}")
        calculate_hough(hough, ax[1,for_index])

        if for_index%4 == 4-1:
        #     dir_path = os.path.join('hough_result', f'{name}_rs{step}')
        #     if not os.path.exists(dir_path):
        #         os.makedirs(dir_path)

        #     plt.savefig(os.path.join(dir_path, f'{name}_rs{step}_num{figure_number}.png'))
        #     figure_number += 1
            plt.show()

def show_chart(target_data, step, segmenter, boun, reset, name=None):
    hough_wafer, hough_seg = segmenter.get_data()

    hough_angle = AutoModelAngleDetector(hough_wafer, hough_seg)

    fig, ax = plt.subplots(1, 2, figsize=(24, 6))
    #ax[0].set_title(f'norm step {step}')
    ax[0].set_title(f'un-norm step {step}')
    #ax[2].set_title(f'norm step {step}')
    ax[1].set_title(f'un-norm step {step}')

    # for r in norm_reset:
    #     r = r[r.RECIPE_STEP == step]
    #     ax[0].plot(r.PARAMETER_VALUE, marker='o')

    for i in target_data.identifier.unique():
        # print(target_data[target_data.identifier == i].PARAMETER_VALUE.reset_index(drop=True))
        ax[0].plot(target_data[target_data.identifier == i].PARAMETER_VALUE.reset_index(drop=True))

    # for r in norm_reset:
    #     ax[2].plot(r.PARAMETER_VALUE)
    # for bb in boun:
    #     ax[2].axvline(bb)

    for c in reset:
        ax[1].plot(c.PARAMETER_VALUE)
    for bb in boun:
        ax[1].axvline(bb)

    plt.show()

    # fig, ax = plt.subplots(1,2, figsize=(16,4))
    plt.figure(figsize=(8, 4))

    index_point = hough_angle.get_point()[0]
    plt.ylim([-1, 101])
    plt.title(f"{name} new version step{step}\nstart{index_point['start']}, end{index_point['end']}")
    if index_point['start'] == -1:
        new_target, _ = segmenter.get_data(is_ignore_undefined_count=True)
        for www in new_target:
            plt.plot(www.PARAMETER_VALUE)
    else:
        for ww in hough_wafer:
            plt.plot(ww.PARAMETER_VALUE)
            plt.plot(ww.loc[index_point['start']:index_point['end'], 'PARAMETER_VALUE'], color='purple', linewidth='3')
    plt.show()
