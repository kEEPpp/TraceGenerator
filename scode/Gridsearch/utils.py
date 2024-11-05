import sys
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from functools import partial
from tqdm import tqdm
import warnings

from glob import glob
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))

from bayesian_github.bayesian_changepoint_detection.priors import const_prior
from bayesian_github.bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
import bayesian_github.bayesian_changepoint_detection.online_likelihoods as online_ll
from bayesian_github.bayesian_changepoint_detection.hazard_functions import constant_hazard

warnings.filterwarnings('ignore')

def bayesian_cpd(example, past=3, thres=0.1):
    example = norm(example)
    hazard_function = partial(constant_hazard, len(example) * 0.5)
    #hazard_function = partial(constant_hazard, 30)
    likelihood = online_ll.StudentT(alpha=0.1, beta=0.1, kappa=0.2, mu=example[0])  # kappa ==> 민감도?
    R, maxes = online_changepoint_detection(example, hazard_function, likelihood)

    new_R = R[past, past:-1]
    cpd = np.where(new_R > thres)[0]
    cpd = np.delete(cpd, np.where(np.diff(cpd) == 1))
    cpd = np.append(cpd, len(example))

    return cpd

def bayesian_detection(example):
    example = norm(example)
    hazard_function = partial(constant_hazard, len(example) * 0.5)
    likelihood = online_ll.StudentT(alpha=1, beta=5, kappa=0.2, mu=0)  # kappa ==> 민감도?
    R, maxes = online_changepoint_detection(example, hazard_function, likelihood)

    return R, maxes

def max_index(maxes, past):
    cpd = np.where(maxes == past)[0]-past

    return cpd

def find_cpd_max_index_method(example, past):
    R, maxes = bayesian_detection(example)
    cpd = max_index(maxes, past)

    return cpd

def summation_method(R, maxes, past, new_past=0):
    new_array = np.sum(R[:past], axis=0)
    new_R = np.concatenate([new_array.reshape(1, -1), R], axis=0)
    new_R[0, :past] = 0
    for t in range(len(R)):
        maxes[t] = new_R[:, t].argmax()

    cpd = max_index(maxes, new_past)

    return cpd

def find_cpd_summation_method(example, past, new_past=0):
    R, maxes = bayesian_detection(example)
    cpd = summation_method(R, maxes, past, new_past)

    return cpd

def extract_segmented_part(cpd, example):
    segment = []
    for key in range(len(cpd) - 1):
        segmented_part = example[cpd[key]:cpd[key + 1]]
        if len(segmented_part) == 1:
            np.append(segment[-1], segmented_part)
        else:
            segment.append(segmented_part)

    return segment

def expand_change_point_detection(segment, cpd, past=3, thres=0.2):
    full_cpd = []
    for example in segment:
        if len(example) <= past:
            continue
        new_example = norm(example)

        # online result
        cpd_temp = bayesian_cpd(new_example)

        if 0 in cpd_temp:
            cpd_temp = np.delete(cpd_temp, np.where(cpd_temp == 0))

        full_cpd.append(cpd_temp)

    for key, value in enumerate(full_cpd):
        cpd = np.append(cpd, cpd[key] + value)

    return cpd

def find_new_change_point(example):
    cpd = bayesian_cpd(example)
    segment = extract_segmented_part(cpd, example)
    new_cpd = expand_change_point_detection(segment, cpd)

    return new_cpd

def draw_change_point(cpd, example, ax):
    #plt.figure(figsize=(12,6))
    ax.plot(example)
    for i in cpd:
        ax.axvline(i, color='green', linewidth=3, linestyle='--', alpha=0.8)
    return ax
    #plt.show()

def cos_sim(a,b):
    cos = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if math.isnan(cos):
        return 0
    return cos

def dtw_distance(x, y, dist):
    m = len(x)
    n = len(y)
    
    # DTW 행렬 초기화
    dtw = np.zeros((m+1, n+1))
    
    # 첫 번째 행 초기화 (경로 시 제외하기 위해)
    dtw[0, 1:] = np.inf
    
    # 첫 번째 열 초기화 (경로 시 제외하기 위해)
    dtw[1:, 0] = np.inf
    
    # DTW 행렬 계산
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = dist(x[i-1], y[j-1])  # 두 데이터 포인트 간의 거리 또는 유사성 계산
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    # DTW 거리 반환
    return dtw[m, n]

def distance(a, b):
    return abs(a - b)

def insert_identifier(df):
    if 'identifier' in df.columns:
            pass
    else:
        df.insert(0,
                    'identifier',
                    df['LOT_ID'].astype('str') + '_' + df['WAFER_ID'].astype('str'))
    
    return df

def load_data(path):
    data = []
    for num, p in enumerate(path):
        temp = []
        df = pd.read_csv(p, dtype={'WAFER_ID':object})
        df = insert_identifier(df)                

        for w in df.WAFER_ID.unique():
            wafer = df[df.WAFER_ID == w] 
            temp.append(wafer)
        data.append(temp)

    return data

def f_z_score(data): # z-score 함수 생성
    mean = np.mean(data) #평균
    std = np.std(data)   #표준편차
    z_scores = [(y-mean)/std for y in data] #z-score
    return z_scores

def norm(example, norm=2):
    if norm == 3:
        return f_z_score(example)

    norm_factor = 10 ** norm

    max_val = example.max()
    min_val = example.min()

    new_example = norm_factor * (example - min_val) / (max_val - min_val)

    return new_example


def grid_search_each_case(example, label, param_grid):
    case_list = []

    case = {'past': list(),
            'threshold': list(),
            'norm': list(),
            'kappa': list(),
            'alpha': list(),
            'beta': list(),
            'mu': list(),
            'cos': list()}

    for norm_factor in param_grid['normalize']:
        #example = d[0].PARAMETER_VALUE.values
        example = norm(example, norm_factor)

        # case_num = path[num].split(os.sep)[-1].split('.')[0].split('_')[-1]
        hazard_function = partial(constant_hazard, 250)
        for kappa in param_grid['kappa']:
            for alpha in param_grid['alpha']:
                for beta in param_grid['beta']:
                    for mu in param_grid['mu']:
                        likelihood = online_ll.StudentT(alpha=alpha, beta=beta, kappa=kappa, mu=mu)
                        R, maxes = online_changepoint_detection(example, hazard_function, likelihood)

                        for nw in param_grid['past']:
                            for thres in param_grid['threshold']:
                                new_R = R[nw, nw:-1]
                                cpd = np.where(new_R > thres)[0]
                                if 0 in cpd:
                                    cpd = np.delete(cpd, np.where(cpd == 0))
                                cpd -= 1  # 베이지안은 -1 해줘야함

                                cpd_array = np.zeros(shape=len(example))
                                label_array = np.zeros(shape=len(example))

                                cpd_array[cpd] = 1
                                label_array[label] = 1

                                cos_list = cos_sim(cpd_array, label_array)
                                # dtw_list = dtw_distance(cpd, label_list[num], distance)

                                case['past'].append(nw)
                                case['threshold'].append(thres)
                                case['norm'].append(norm_factor)
                                case['alpha'].append(alpha)
                                case['beta'].append(beta)
                                case['kappa'].append(kappa)
                                case['mu'].append(mu)
                                case['cos'].append(cos_list)

                                # case['normalize'].append(norm_factor)
    #case_list.append(case)

    return case


def grid_search(data, label_list, param_grid):
    case_list = []
    for num, d in tqdm(enumerate(data)):
        case = {'past': list(),
                'threshold': list(),
                'norm': list(),
                'kappa': list(),
                'alpha': list(),
                'beta': list(),
                'mu': list(),
                'cos': list()}

        for norm_factor in param_grid['normalize']:
            example = d[0].PARAMETER_VALUE.values
            example = norm(example, norm_factor)

            #case_num = path[num].split(os.sep)[-1].split('.')[0].split('_')[-1]
            hazard_function = partial(constant_hazard, 250)
            for kappa in param_grid['kappa']:
                for alpha in param_grid['alpha']:
                    for beta in param_grid['beta']:
                        for mu in param_grid['mu']:
                            likelihood = online_ll.StudentT(alpha=alpha, beta=beta, kappa=kappa, mu=mu)
                            R, maxes = online_changepoint_detection(example, hazard_function, likelihood)

                            for nw in param_grid['past']:
                                for thres in param_grid['threshold']:
                                    new_R = R[nw, nw:-1]
                                    cpd = np.where(new_R > thres)[0]
                                    if 0 in cpd:
                                        cpd = np.delete(cpd, np.where(cpd == 0))
                                    cpd -= 1  # 베이지안은 -1 해줘야함

                                    cpd_array = np.zeros(shape=len(example))
                                    label_array = np.zeros(shape=len(example))
                                    
                                    cpd_array[cpd] = 1
                                    label_array[label_list[num]] = 1

                                    cos_list = cos_sim(cpd_array, label_array)
                                    # dtw_list = dtw_distance(cpd, label_list[num], distance)

                                    case['past'].append(nw)
                                    case['threshold'].append(thres)
                                    case['norm'].append(norm_factor)
                                    case['alpha'].append(alpha)
                                    case['beta'].append(beta)
                                    case['kappa'].append(kappa)
                                    case['mu'].append(mu)
                                    case['cos'].append(cos_list)

                                    # case['normalize'].append(norm_factor)
        case_list.append(case)

    return case_list

def draw_online_example(data, path, num_row=4):
    for num, d in enumerate(data):
        example = d[0].PARAMETER_VALUE.values
        fn = path[num].split(os.sep)[-1].split('.')[0].split('_')[-1]
        example = norm(example, 2)
        hazard_function = partial(constant_hazard, 250)
        likelihood = online_ll.StudentT(alpha=0.1, beta=0.1, kappa=0.1, mu=0)
        R, maxes = online_changepoint_detection(example, hazard_function, likelihood)

        nw = 3
        thres = 0.2
        new_R = R[nw, nw:-1]
        cpd = np.where(new_R > thres)[0]

        if num % num_row == 0:
            fig, ax = plt.subplots(1, num_row, figsize=(32, 6))
        for_index = num % num_row
        # plt.figure(figsize=(12,6))

        ax[for_index].set_title(fn)
        ax[for_index].plot(example)
        for label in cpd:
            ax[for_index].axvline(label, color='red', linewidth=3, linestyle='--', alpha=0.8)

        if for_index % num_row == num_row - 1:
            plt.show()

def load_label_list():
    label_list = []

    #label1 = [98, 209, 240, 459, 494, 513]
    #label4 = [2, 25, 30, 34, 36, 40, 45, 48, 50, 54, 57, 60, 62, 66, 69, 73, 76]
    label6 = [30, 51, 150, 181, 202, 421, 430, 438, 446, 454, 462, 470, 478, 486, 494, 502, 510, 518, 526, 536, 542, 571, 577, 585, 616, 637, 666, 727, 771, 792]
    label7 = [21, 142, 153, 275]
    label8 = [21, 134, 155, 186, 207, 238, 259, 266, 293, 312, 323]
    label13 = [2, 81, 146, 150, 167, 172, 192, 197, 208, 213, 258, 263, 293, 298, 344, 349]
    label14 = [3, 63]
    label15 = [12, 17, 113, 118, 157, 178, 208]
    label16 = [11, 13]
    label18 = [35, 41]
    label19 = []
    label21 = []
    label22 = [12, 26, 29, 401] # add more line
    label23 = [26, 44, 67, 71] # add more line
    label24 = [25, 41]
    label25 = [19, 31, 401] # add more line
    label26 = []
    label32 = [14, 81, 95, 151, 166, 232, 261, 281, 421, 434, 462, 482]
    #label33 = [221, 231, 251, 256, 401, 421]
    label34 = [15, 1101, 1111, 1141, 1151, 1201, 1211] # add more line
    label35 = [171, 182, 244, 254]
    label36 = [41, 93, 103, 123, 135, 145, 155, 166, 174, 285, 297, 328, 337]
    label37 = [30, 93, 103, 124]

    label_list = [label6, label7, label8, label13, label14, label15, label16, label18, label19, label21, label22, label23, label24, label25, label26, label32, label34, label35, label36, label37]

    for k, l in enumerate(label_list):
        label_list[k] = list(np.array(l)-1)

    return label_list

def load_prior1_list():
    prior1 = []

    #label1 = [98, 209, 240, 459, 494, 513]
    #label4 = [2, 25, 30, 34, 36, 40, 45, 48, 50, 54, 57, 60, 62, 66, 69, 73, 76]
    label6 = [30, 51, 150, 181, 202, 421, 542, 571, 577, 585, 616, 637, 666, 727, 771, 792]
    label7 = [21, 142, 153, 275]
    label8 = [21, 134, 155, 186, 207, 238, 259, 293, 312, 323]
    label13 = [2, 81, 146, 150, 167, 172, 192, 197, 208, 213, 258, 263, 293, 298, 344, 349]
    label14 = []
    label15 = [12, 17, 113, 118]
    label16 = [11, 13]
    label18 = [35, 41]
    label19 = []
    label21 = []
    label22 = [29] # add more line
    label23 = [26, 44, 67, 71] # add more line
    label24 = [25, 41]
    label25 = [19, 31, 401] # add more line
    label26 = []
    label32 = [14, 81, 95, 151, 166, 232, 261, 281, 421, 434, 462, 482]
    #label33 = [221, 231, 251, 256, 401, 421]
    label34 = [15, 1101, 1111, 1141, 1151, 1201, 1211] # add more line
    label35 = [171, 182, 244, 254]
    label36 = [41, 93, 103, 123, 145, 155, 174, 285, 297, 328]
    label37 = [30, 93, 103, 124]

    prior1 = [label6, label7, label8, label13, label14, label15, label16, label18, label19, label21, label22, label23, label24, label25, label26, label32, label34, label35, label36, label37]

    for k, l in enumerate(prior1):
        prior1[k] = list(np.array(l)-1)

    return prior1

def load_prioir2_list():
    prior2 = []

    # label1 = [98, 209, 240, 459, 494, 513]
    # label4 = [2, 25, 30, 34, 36, 40, 45, 48, 50, 54, 57, 60, 62, 66, 69, 73, 76]
    label6 = [430, 438, 446, 454, 462, 470, 478, 486, 494, 502, 510, 518, 526, 536]
    label7 = []
    label8 = [266]
    label13 = []
    label14 = [3, 63]
    label15 = [157, 178, 208]
    label16 = []
    label18 = []
    label19 = []
    label21 = []
    label22 = [12, 26, 401]  # add more line
    label23 = []  # add more line
    label24 = []
    label25 = []  # add more line
    label26 = []
    label32 = []
    # label33 = [221, 231, 251, 256, 401, 421]
    label34 = []  # add more line
    label35 = []
    label36 = [135, 166, 337]
    label37 = []

    prior2 = [label6, label7, label8, label13, label14, label15, label16, label18, label19, label21, label22, label23,
              label24, label25, label26, label32, label34, label35, label36, label37]

    for k, l in enumerate(prior2):
        prior2[k] = list(np.array(l) - 1)

    return prior2