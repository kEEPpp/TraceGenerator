import copy
import inspect
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


function_list = ['constant',
                 'impulse_like',
                 'linear_transition',
                 'rectangular_pulse',
                 'step_like',
                 'trapezoidal',
                 'triangular']

class TraceModule:
    def spike(self, maximum, minimum):
        return np.random.uniform(minimum, maximum)

    def gaussian_noise(self, signal_length, mean=0, std=1):
        return np.random.randn(signal_length) * std + mean

    def linear_transition(self, start_value, end_value, time1, time2, length):
        transition_length = time2 - time1 + 1
        start = np.ones(time1) * start_value
        t = np.arange(0, transition_length)
        m = (end_value - start_value) / (transition_length - 1)
        transition = start_value + m * t
        end = np.ones(length - time2 - 1) * end_value
        signal = np.concatenate([start, transition, end])

        return signal

    def constant(self, start_value, length):
        constant = np.ones(length) * start_value

        return constant

    def step_like(self, start_value, end_value, time1, length):
        time1 += 1
        start = np.ones(time1) * start_value
        end = np.ones(length - time1) * end_value

        signal = np.concatenate([start, end])

        return signal

    def exponential_form(self, start_value, end_value, time1, b, length):
        time1 += 1
        start = np.ones(time1) * start_value
        t = np.arange(time1, length)
        # t = np.arange(1, length-time1) check
        return np.append(start, start_value + (end_value - start_value) * (1 - np.exp(-t / b)))

    def impulse_like(self, start_value, peak_value, end_value, time1, length):
        start = np.ones(time1) * start_value
        peak = np.ones(1) * peak_value
        end = np.ones(length - time1 - 1) * end_value

        return np.concatenate([start, peak, end])

    def rectangular_pulse(self, start_value, high_value, end_value, time1, time2, length):
        start = np.ones(time1) * start_value
        high = np.ones(time2 - time1 + 1) * high_value
        end = np.ones(length - time2 - 1) * end_value

        return np.concatenate([start, high, end])

    def trapezoidal(self, start_value, high_value, end_value, time1, time2, time3, time4, length):
        start = np.ones(time1) * start_value
        up_trend = self.linear_transition(start_value=start_value, end_value=high_value, time1=0, time2=time2 - time1,
                                          length=(time2 - time1 + 1))
        up_stable = np.ones(time3 - time2 - 1) * high_value
        down_trend = self.linear_transition(start_value=high_value, end_value=end_value, time1=0, time2=time4 - time3,
                                            length=(time4 - time3 + 1))
        end = np.ones(length - time4 - 1) * end_value

        return np.concatenate([start, up_trend, up_stable, down_trend, end])

    def triangular(self, start_value, peak_value, end_value, time1, time2, time3, length):
        start = np.ones(time1) * start_value
        up_trend = start_value + np.arange(time2 - time1) * (peak_value - start_value) / (time2 - time1)
        peak = np.array([peak_value])
        down_trend = peak_value - np.arange(1, time3 - time2 + 1) * (peak_value - end_value) / (time3 - time2)
        end = np.ones(length - time3 - 1) * end_value

        return np.concatenate([start, up_trend, peak, down_trend, end])

    def exponential_pulse1(self, start_value, high_value, end_value, time1, time2, b, c, length):
        form1 = self.exponential_form(start_value, high_value, time1, b, time2 + 1)
        t = np.arange(1, length - time2)
        form2 = end_value + (high_value - end_value) * (1 - np.exp(-(t - time1 + time2) / b)) * np.exp(-t / c)

        return np.concatenate([form1, form2])

    def cloud_pulse(self, start_value, std_value, length):
        return self.gaussian_noise(length, mean=start_value, std=std_value)

    def exp_pulse(self, start_value, end_value, time1, time2, b_coeff, length):
        start = np.ones(time1) * start_value
        t = np.arange(time1, time2 + 1)

        p, q = self._exp_pulse(start_value, end_value, time1, time2, b_coeff)
        mid_value = np.exp((t - p) / b_coeff) + q
        if start_value > end_value:
            mid_value = mid_value[::-1]

        end = np.ones(length - time2 - 1) * end_value

        return np.concatenate([start, mid_value, end])

    def _exp_pulse(self, start_value, end_value, time1, time2, b):
        larger_value = max(start_value, end_value)
        smaller_value = min(start_value, end_value)

        k = (time1 - time2) / b
        p = time1 - b * np.log((np.exp(k) * (smaller_value - larger_value)) / (np.exp(k) - 1))
        q = (larger_value * np.exp(k) - smaller_value) / (np.exp(k) - 1)

        return p, q

    def log_pulse(self, start_value, end_value, time1, time2, b_coeff, length):
        start = np.ones(time1) * start_value
        t = np.arange(time1, time2 + 1)

        p, q = self._log_pulse(start_value, end_value, time1, time2, b_coeff)
        mid_value = -np.exp(-(t - p) / b_coeff) + q

        if start_value > end_value:
            mid_value = mid_value[::-1]

        end = np.ones(length - time2 - 1) * end_value

        return np.concatenate([start, mid_value, end])

    def _log_pulse(self, start_value, end_value, time1, time2, b):
        larger_value = max(start_value, end_value)
        smaller_value = min(start_value, end_value)

        p = b * np.log((smaller_value - larger_value) / ((np.exp(-time2 / b) - np.exp(-time1 / b))))
        q = smaller_value + np.exp((p - time1) / b)

        return p, q


class TraceDataGeneration(TraceModule):
    def __init__(self, n, global_noise=0, param_list=None, trace=None, seed=26):
        self.n = n
        self.global_noise = global_noise
        #self.seed_everything(seed)

        # nonfocus
        if param_list is None:
            self.fault_flag = False
            self.param_list = {}
            for i in range(self.n):
                self.param_list[f'trace{i + 1}'] = {}

        # focus
        else:
            self.param_list = param_list
            self.fault_flag = True
            self.fault_param_list = {}
            for i in range(self.n):
                self.fault_param_list[f'trace{i + 1}'] = {}


            # calculate step information such as, start index, end index, length
            self.extract_start_end_index_legnth()

            # calculate recipe step information ex: [1,1,1,1,1,2,2,2,2,3,3]
            #self.calculate_recipe_step_index(param_list)

            # nonfocus trace
            self.trace = trace

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    def extract_start_end_index_legnth(self):
        step_indices = {}
        for i in range(len(self.param_list)):
            step_indices[f'trace{i + 1}'] = {}
            current_index = 0
            for key, value in self.param_list[f'trace{i + 1}'].items():
                #step_number = str(key.split('_')[0])
                step_number = key
                if step_number not in step_indices[f'trace{i + 1}']:
                    step_indices[f'trace{i + 1}'][step_number] = {}
                    step_indices[f'trace{i + 1}'][step_number]['start_index'] = current_index
                    start_index = current_index
                current_index += value['length']
                step_indices[f'trace{i + 1}'][step_number]['end_index'] = current_index
                end_index = current_index
                step_indices[f'trace{i + 1}'][step_number]['length'] = end_index - start_index

        self.trace_step_info = step_indices

    def extract_recipe_step_length(self):
        step_indices = {}
        for i in range(len(self.param_list)):
            step_indices[f'trace{i + 1}'] = {}
            current_index = 0
            for key, value in self.param_list[f'trace{i + 1}'].items():
                step_number = str(key.split('_')[0])
                #step_number = key
                if step_number not in step_indices[f'trace{i + 1}']:
                    step_indices[f'trace{i + 1}'][step_number] = {}
                    step_indices[f'trace{i + 1}'][step_number]['start_index'] = current_index
                    start_index = current_index
                current_index += value['length']
                step_indices[f'trace{i + 1}'][step_number]['end_index'] = current_index
                end_index = current_index
                step_indices[f'trace{i + 1}'][step_number]['length'] = end_index - start_index

        self.step_index = step_indices

    def calculate_recipe_step_index(self, param_list):
        all_recipe_step = {}
        for i in range(self.n):
            trace = f'trace{i + 1}'
            all_recipe_step[trace] = {}
            recipe_step_len = []
            for key, value in param_list[trace].items():
                step_number = str(key.split('_')[0])
                recipe_step_len.extend(np.ones(shape=value['length'], dtype=int) * int(step_number[4:]))
                all_recipe_step[trace]['step_info'] = recipe_step_len

        self.recipe_step_index = all_recipe_step

    def param(self, trace_type, recipe_step, **kwargs):
        step_name = recipe_step.replace(',', '_')
        step_name = 'step' + step_name
        # fault type and jitter check
        fault_type = kwargs.get('fault')
        fault_level = kwargs.get('level')
        jitter_bool = kwargs.get('jitter')

        # value, time, coeff parameter check
        is_values = [s for s in kwargs if 'value' in s]
        is_times = [s for s in kwargs if 'time' in s]
        is_coeff = [s for s in kwargs if 'coeff' in s]
        is_jitter = [s for s in kwargs if 'jit' in s]
        

        #print(is_times)
        # allocate value parameter
        param_dict = {}
        for v in is_values:
            param_dict[v] = kwargs[v]

        # allocate coeff parameter
        for v in is_coeff:
            param_dict[v] = kwargs[v]

        trace_list = []
        for i in range(self.n):

            # allocate time parameter ==> if jitter exists, the length value is modified
            if is_jitter or jitter_bool is not None:
                if jitter_bool is not None:
                    kwargs['jit_length'] = 0.1
                    kwargs['jit_transition'] = 0.1
                #print(kwargs)
                time_values = [kwargs[v] for v in is_times]
                new_length, new_time_values = self.jitter(kwargs['jitter_point'], kwargs['length'], *time_values)
                param_dict.update({'length': new_length, **dict(zip(is_times, new_time_values))})

            else:
                # allocate length parameter
                param_dict.update({v: kwargs[v] for v in is_times})
                param_dict['length'] = kwargs['length']
            # save nonfocus parameter information
            if not self.fault_flag:
                self.param_list[f'trace{i + 1}'].update({step_name: {'length': param_dict['length']}})
            else:
                self.fault_param_list[f'trace{i + 1}'].update({step_name: {'length': param_dict['length']}})


            # in case of generating focus data, check whether fault exist or not
            if fault_type is not None:
                # change trace to fault trace
                trace = self.change_trace_to_fault(fault_type, fault_level, param_dict, step_name, trace_type)
            else:
                # call defined trace module
                trace = getattr(super(), trace_type)
                trace = trace(**param_dict)

            #add noise
            #local_segment_range = np.max(trace) - np.min(trace)
            #if trace_type == 'constant':
            #   local_segment_range = param_dict['start_value']/5
            #noise = self.gaussian_noise(len(trace), mean=0, std=self.global_noise)
            #trace += noise

            trace_list.append(trace)
        return trace_list

    def change_trace_to_fault(self, fault_type, fault_level, param_dict, recipe_step, trace_type):
        #step_name = recipe_step.split(',')[0]
        step_name = recipe_step

        # extract recipe step part in nonfocus traces
        std = 0
        for num, values in enumerate(self.trace_step_info.items()):
            trace_name, step_info = values
            start_index, end_index = step_info[step_name]['start_index'], step_info[step_name]['end_index']

            std += np.std(self.trace[num][start_index:end_index], ddof=1)
        std = std / len(self.trace)

        trace = getattr(super(), trace_type)
        trace = trace(**param_dict)

        # fault type check
        if fault_type == 'spike':
            spike_level = std * fault_level
            peak_time = random.randint(0, int(param_dict['length'] * 0.9))

            trace[peak_time] += spike_level
            return trace

        elif fault_type == 'mean_shift':
            shift_level = std * fault_level
            return trace + shift_level

        elif fault_type == 'fluctuation':
            noise = self.gaussian_noise(len(trace), mean=0, std=std*fault_level)
            return trace + noise

        elif fault_type == 'blob':
            return trace + std * fault_level * np.sin(np.pi * np.arange(len(trace)) / (len(trace) - 1))

    def concatenate_traces(self, args):
        traces = []
        for n in range(self.n):
            new_trace = []
            for step in args:
                new_trace.append(step[n])
            new_trace = np.concatenate(new_trace)

            global_range = np.max(new_trace) - np.min(new_trace)

            noise = self.gaussian_noise(len(new_trace), mean=0, std=global_range * self.global_noise)
            new_trace += noise
            traces.append(new_trace)

        return traces

    # only return trace value
    def generate_trace(self, *args):
        #return args
        #data = [np.concatenate([wafer[n] for wafer in args], axis=0) for n in range(self.n)]
        data = self.concatenate_traces(args)
        return data

    # return dataframe format
    def make_trace_format(self, *args):
        # make recipe step indices only in nonfocus cases
        if not self.fault_flag:
            self.calculate_recipe_step_index(self.param_list)
            self.extract_recipe_step_length()
        else:
            self.calculate_recipe_step_index(self.fault_param_list)

        col = ['LOT_ID', 'WAFER_ID', 'PROCESS', 'PROCESS_STEP', 'RECIPE', 'RECIPE_STEP', 'PARAMETER_NAME',
               'PARAMETER_VALUE', 'TIME']
        data = []
        trace_values = self.concatenate_traces(args)
        for n in range(self.n):
            df = pd.DataFrame([], columns=col)
            #df = pd.concat(df)
            df['PARAMETER_VALUE'] = trace_values[n]
            df = df.reset_index(drop=True)
            df['LOT_ID'] = 'lot_a'
            if self.fault_flag:
                df['WAFER_ID'] = f'wafer{len(self.param_list)+1}'
            else:
                df['WAFER_ID'] = f'wafer{n + 1}' # focus data 항상 n+1 이 부분 변경
            df['PROCESS'] = 'process'
            df['PROCESS_STEP'] = 'process_step'
            df['RECIPE'] = 'recipe'
            df['RECIPE_STEP'] = np.array(self.recipe_step_index[f'trace{n + 1}']['step_info'], dtype=str)
            df['PARAMETER_NAME'] = 'parameter_name'
            if self.fault_flag:
                df['TYPE'] = 'focus'
            else:
                df['TYPE'] = 'nonfocus'
            data.append(df)

        data = pd.concat(data)
        data['TIME'] = pd.date_range("2024-01-01", periods=data.shape[0], freq="S")
        temp_list = []
        for wafer in data['WAFER_ID'].unique():
            temp_list.append(data[data['WAFER_ID'] == wafer])

        return temp_list

    """jiter mean time difference between recorded signal"""

    def jitter_value_calculation(self, maximum_point):
        value = np.random.normal(1)
        intervals = np.linspace(-3, 3, maximum_point*2+1)
        jitter_score = list(range(-maximum_point+1, 1)) + list(range(0, maximum_point))

        idx = np.searchsorted(intervals, value, side='right')
        # less than 3
        if idx == 0:
            return jitter_score[0] - 1
        # grater than 3
        elif idx == len(intervals):
            return jitter_score[-1] + 1
        else:
            return jitter_score[idx - 1]

    def jitter(self, maximum_point, length, *time):

        # length jitter factor
        length_jitter_factor = self.jitter_value_calculation(maximum_point)
        new_length = length + length_jitter_factor
        #print(f'length:{length} new_length:{new_length}')
        # time jitter factor
        if time:
            new_time = []
            for key, t in enumerate(time):
                time_jitter_factor = self.jitter_value_calculation(maximum_point)
                new_t = max(0, t + time_jitter_factor)
                new_time.append(new_t)

            new_time.sort()
            #print(f'time:{time} new_time:{new_time}')

            if max(new_time) >= new_length:
                new_length = max(new_time) + 1

            if len(set(new_time)) != len(time):
                return length, time

            return new_length, new_time

        else:
            return new_length, []


    def jitter_old(self, jit_length, jit_transition, length, *time):
        # assert max(time) < length, "length must be larger than time value"
        new_length = int(np.random.randn(1) * length * jit_length + length)
        if new_length > length:
            new_length += 1
        else:
            new_length = length
        if time:
            new_time = []
            for key, t in enumerate(time):
                time_difference = max(0, int(np.random.randn(1) * t * jit_transition + t))
                new_time.append(time_difference)

            # check if time1 > time2 ==> change two values
            new_time.sort()

            # check: length must larger than time value
            # print(f"jitter logic\nnew_time: {new_time}\nnew_length: {new_length}\nlength: {length}")
            if max(new_time) >= new_length:
                return length, time

            # check duplicated time
            if len(set(new_time)) != len(time):
                return length, time

            return new_length, new_time

        # if time parameter don't exist then only return jitter length value
        else:
            return new_length, []


class AutoGenerator_original(TraceModule):

    def __init__(self, n, max_value, min_value, function_list, seed=23):
        self.n = n
        self.max_value = max_value
        self.min_value = min_value
        self.function = function_list
        self.seed = seed
        self.seed_everything()

        self.global_range = (max_value - min_value)
        self.mean_profiles = self.random_mean(len(function_list))
        self.param = self.default_parameter_generation()

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)

    def generate_recipe_step_num(self, num_function):
        target_recipe_step = random.randint(1, num_function)
        if target_recipe_step == num_function:
            return list(np.ones(1) * num_function)
        
        result = [] 
        # 리스트의 길이가 1 이상이고, 원하는 합계가 양수인 경우에만 작동
        if target_recipe_step <= 0 or num_function <= 0:
            return result
        
        for _ in range(target_recipe_step - 1):
            # 남은 합계 중에서 랜덤한 숫자를 생성
            num = random.randint(1, num_function - (target_recipe_step - len(result)))
            # 리스트에 추가
            result.append(num)
            # 합계 업데이트
            num_function -= num
        
        # 마지막 숫자는 남은 합계
        result.append(num_function)
        
        return result

    def random_mean(self, step_len=3):
        mean_range = np.random.rand(step_len)
        mean_range = (mean_range - np.min(mean_range)) / (np.max(mean_range) - np.min(mean_range))
        mean_range = mean_range * (self.max_value - self.min_value) + self.min_value

        return mean_range

    def generate_length(self):
        # trace_len = samples = np.random.exponential(400, 1000)
        return max(len(self.function) * 3, np.random.geometric(0.004, 1)[0])

    def generate_boundary(self, trace_length):

        length_list = np.random.dirichlet(np.ones(len(self.function)))
        # length_list = length_list / np.sum(length_list)
        trace_section = np.array(length_list * trace_length, dtype=int)
        #print(trace_section)
        # time parameter 는 최대 길이가 4인데, trace section의 길이가 4 미만인 경우와 만나면 에러가 생김 추후 고쳐야함
        trace_section = np.array([i if i >= 7 else 8 for i in trace_section])
        return trace_section

    def generate_value_parameter(self, length, loc_factor, std_factor=0.1):
        '''
        mean profile 값을 기준으로 global range * 0.1 std를 갖는 정규분포에서 start, end, high value 등을 뽑아냄
        '''
        #param_value = np.random.normal(loc=self.mean_profiles[key], scale=self.global_range * std_factor, size=len(is_values))
        std = self.global_range * std_factor
        param_value = np.random.normal(loc=loc_factor, scale=std, size=length)

        return param_value

    def generate_time_parameter(self, key, is_times, trace_section):
        if len(is_times) == 1:
            time_target = np.random.uniform(low=0, high=1)
            time_target = [int(time_target * trace_section[key])]
            # print(time_target)

        elif len(is_times) > 1:
            #print(is_times)
            #print(trace_section[key])
            time_target = np.random.choice(np.arange(trace_section[key]), size=len(is_times), replace=False)
            time_target.sort()
            
            # time_list = np.random.dirichlet(np.ones(len(is_times)))
            # time_list = time_list / np.sum(time_list)
            # time_target = np.array(time_list * trace_section[key], dtype=int)
            # if len(set(time_target)) == len(time_target):
            #     pass
            # else:
            #     time_list = np.random.dirichlet(np.ones(len(is_times)))
            #     time_target = np.array(time_list * trace_section[key], dtype=int)
            # time_target.sort()
            

        # len(is_times) == 0
        else:
            time_target = -1

        return time_target
    
    def default_parameter_generation(self):
        # make parameter dictionary 
        param_list = self.generate_trace_parameter()

        # generate length
        trace_length = self.generate_length()
        # generate each length of each trace
        trace_section = self.generate_boundary(trace_length)
        trace_length = np.sum(trace_section)

        # value default parameter generation
        for key, param in enumerate(param_list):
            is_values = [v for v in param if 'value' in v]
            param_value = self.generate_value_parameter(len(is_values), self.mean_profiles[key])

            # allocate value parameter
            for v, p in zip(is_values, param_value):
                param_list[key][v] = p
        
        # time default parameter generation
        for key, param in enumerate(param_list):
            is_times = [t for t in param if 'time' in t]
            param_value = self.generate_time_parameter(key, is_times, trace_section)

            # allocate time parameter
            if not is_times:
                pass
            else:
                for v, p in zip(is_times, param_value):
                    param_list[key][v] = p

        # allocate length
        for key, param in enumerate(param_list):
            param_list[key]['length'] = trace_section[key]

        return param_list
    
    def generate_trace_parameter(self):
        param_list = []
        for f in self.function:
            trace_type = getattr(super(), f)
            param = inspect.signature(trace_type).parameters
            param = list(param.keys())
            value = np.ones(shape=len(param))
            param = dict(zip(param, value))
            param_list.append(param)
        
        return param_list       
    
    def generate_random_trace(self, return_param=False):        
        #param = copy.deepcopy(self.param)
        param_list = []
        traces = []
        for _ in range(self.n):
            step = []
            param = copy.deepcopy(self.param)
            for key, f in enumerate(self.function):
                func = getattr(self, f)
                
                # value parameter generation
                is_values = [v for v in self.param[key] if 'value' in v]
                for value in is_values:
                    # std factor random으로 줘야함
                    param_value = self.generate_value_parameter(1, self.param[key][value], std_factor=0.05)[0]
                    param[key][value] = param_value
                
                # jitter 
                is_times = [t for t in self.param[key] if 'time' in t]
                time_values = [self.param[key][v] for v in is_times]
                    
                new_length, new_time_values = self.jitter(3, param[key]['length'], *time_values)
                param[key].update({'length': new_length, **dict(zip(is_times, new_time_values))})

                trace_value = func(**param[key])
                step.append(trace_value)
            param_list.append(param)
            traces.append(step)
        if return_param:
            return self.make_trace_format(*traces), param_list
        return self.make_trace_format(*traces)
    
    def jitter(self, maximum_point, length, *time):
        # length jitter factor
        length_jitter_factor = self.jitter_value_calculation(maximum_point)
        new_length = length + length_jitter_factor
        #print(f'length:{length} new_length:{new_length}')
        # time jitter factor
        if time:
            new_time = []
            for key, t in enumerate(time):
                time_jitter_factor = self.jitter_value_calculation(maximum_point)
                new_t = max(0, t + time_jitter_factor)
                new_time.append(new_t)

            new_time.sort()
            #print(f'time:{time} new_time:{new_time}')

            if max(new_time) >= new_length:
                new_length = max(new_time) + 1

            if len(set(new_time)) != len(time):
                return length, time

            return new_length, new_time

        else:
            return new_length, []
    
    def jitter_value_calculation(self, maximum_point):
        value = np.random.normal(1)
        intervals = np.linspace(-3, 3, maximum_point*2+1)
        jitter_score = list(range(-maximum_point+1, 1)) + list(range(0, maximum_point))

        idx = np.searchsorted(intervals, value, side='right')
        # less than 3
        if idx == 0:
            return jitter_score[0] - 1
        # grater than 3
        elif idx == len(intervals):
            return jitter_score[-1] + 1
        else:
            return jitter_score[idx - 1]

    def random_parameter_generation(self):
        # generate length
        self.trace_length = self.generate_length()
        trace_length = self.trace_length
        # generate each length of each trace
        self.trace_section = self.generate_boundary(trace_length)
        
        trace_section = self.trace_section

        full_param = []

        for key, f in enumerate(self.function):
            trace_type = getattr(super(), f)
            param_list = inspect.signature(trace_type).parameters
            param_list = list(param_list.keys())
            #print(param_list)
            value = np.ones(shape=len(param_list)) * -1
            param_dict = dict(zip(param_list, value))
            #print(param_dict)

            is_values = [s for s in param_list if 'value' in s]
            is_times = [s for s in param_list if 'time' in s]
            # is_coff = [s for s in param_list if 'b' or 'c' in s]

            # value parameter gen
            param_value = self.generate_value_parameter(key, is_values)

            # allocate value parameter
            for v, p in zip(is_values, param_value):
                param_dict[v] = p

            # allocate length
            param_dict['length'] = trace_section[key]

            # time parameter gen
            time_target = self.generate_time_parameter(key, is_times, trace_section)

            # allocate time parameter
            if not is_times:
                pass
            else:
                for t, p in zip(is_times, time_target):
                    param_dict[t] = p

            full_param.append(param_dict)
            #print(param_dict)

        return full_param

    def add_noise(self, value):
        noise = np.random.normal(loc=0, scale=self.global_range * 0.03, size=len(value))
        value = value + noise
        return value

    def generate_trace(self):
        traces = []
        for n in range(self.n):
            step = []
            for key, f in enumerate(self.function):
                func = getattr(self, f)
                value = func(**self.param[key])
                noised_value = self.add_noise(value)
                step.append(noised_value)
            traces.append(step)
        
        #return traces
        return self.make_trace_format(*traces)
    
    def generate_random_multi_trace(self, para_num=5, return_param=False):        
        #param = copy.deepcopy(self.param)
        param_list = []
        traces = []
        multi_para = []

        for _ in range(para_num):
            for _ in range(self.n):
                step = []
                param = copy.deepcopy(self.param)
                for key, f in enumerate(self.function):
                    func = getattr(self, f)
                    
                    # value parameter generation
                    is_values = [v for v in self.param[key] if 'value' in v]
                    for value in is_values:
                        # std factor random으로 줘야함
                        param_value = self.generate_value_parameter(1, self.param[key][value], std_factor=0.05)[0]
                        param[key][value] = param_value
                    
                    # jitter 
                    is_times = [t for t in self.param[key] if 'time' in t]
                    time_values = [self.param[key][v] for v in is_times]
                        
                    new_length, new_time_values = self.jitter(3, param[key]['length'], *time_values)
                    param[key].update({'length': new_length, **dict(zip(is_times, new_time_values))})

                    trace_value = func(**param[key])
                    step.append(trace_value)
                param_list.append(param)
                traces.append(step)
            multi_para.append(traces)
        
        if return_param:
            return self.make_multi_trace_format(para_num, *multi_para), param_list
        return self.make_multi_trace_format(para_num, *multi_para)
       
    def make_trace_format(self, *args):
        # make recipe step indices only in nonfocus cases
        
        col = ['Tool/Chamber', 'Tool', 'chamber', 'Process_Recipe', 'Start_Time',
               'Carrier_ID', 'WAFER_ID', 'Slot']

        col = ['LOT_ID', 'WAFER_ID', 'PROCESS', 'PROCESS_STEP', 'RECIPE', 'RECIPE_STEP', 'PARAMETER_NAME',
               'PARAMETER_VALUE', 'TIME']
        
        data = []
        for num, trace in enumerate(args):
            df = pd.DataFrame([], columns=col)
            df.PARAMETER_VALUE = np.concatenate(trace) 
            df['LOT_ID'] = 'lot'
            df['WAFER_ID'] = f'wafer{num+1}'
            df['PROCESS'] = 'process'
            df['PROCESS_STEP'] = 'process_step'
            df['RECIPE'] = 'recipe'
            df['PARAMETER_NAME'] = 'parameter_name'
            # step_list = []
            # for step_num, step in enumerate(single):
            #     recipe_step = [str(step_num+1)] * len(step)
            #     step_list.extend(recipe_step)
            df['RECIPE_STEP'] = [str(step_num+1) for step_num, step in enumerate(trace) for _ in step]
            data.append(df)
        data = pd.concat(data)
        data['TIME'] = pd.date_range("2024-01-01", periods=data.shape[0], freq="S")

        return data

    def trace_checker(self, df, marker=False):
        #fname = p.split(os.sep)[-1].split('.')[0]
        step_list = list(np.sort(np.unique(df.RECIPE_STEP.values)))
        reset_raw_data, boundary_raw = self.reset_index(df, step_list, True)
        #print(boundary_raw)
        self.index_sorted_image(reset_raw_data, boundary_raw, marker)

    def reset_index(self, trace, step_list, return_recipe_step_boundary=False):
        full_trace_list = []
        for i in trace['WAFER_ID'].unique():
            d = trace[trace.WAFER_ID == i].reset_index(drop=True)
            full_trace_list.append(d)

        # step_list = cleaned_data.RECIPE_STEP.unique()
        col_name = trace.columns.to_list()
        group_df = trace.groupby('WAFER_ID')
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
                        full_trace_list[key] = pd.DataFrame(np.insert(full_trace_list[key].values,
                                                                    new_index + i,
                                                                    values=[np.nan],
                                                                    axis=0),
                                                            columns=col_name)
                        full_trace_list[key].loc[new_index + i, 'RECIPE_STEP'] = step_number

        for key, f in enumerate(full_trace_list):
            full_trace_list[key] = f.dropna(subset=['WAFER_ID'])

        accumulated_index = np.cumsum(recipe_step_index_list)
        accumulated_index[-1] = accumulated_index[-1] - 1
        if return_recipe_step_boundary:
            return full_trace_list, accumulated_index
        else:
            return full_trace_list

    def index_sorted_image(self, reset_raw_data, boundary_raw, marker, path=None):
        plt.figure(figsize=(12, 6))

        plt.title(f'# of {len(reset_raw_data)} wafer')
        if marker:
            mark = 'o'
        else:
            mark = None
        for wafer in reset_raw_data:
            plt.plot(wafer.PARAMETER_VALUE, label=None, marker=mark)
            
        for i in boundary_raw:
            plt.axvline(i, linestyle='--', alpha=0.3, color='black')
        if path is not None:
            plt.savefig(path)
        plt.show()

class AutoGenerator(TraceModule):

    def __init__(self, n, max_value, min_value, function_num, para_num, seed=23):
        self.n = n
        self.max_value = max_value
        self.min_value = min_value
        self.function_num = 3
        self.function = None
        self.para_num = para_num
        self.function_lists = [[random.choice(function_list) for _ in range(function_num)] for _ in range(para_num)]

        
        self.global_range = max_value - min_value

        self.seed = seed
        self.seed_everything()

        self.mean_profiles = self.random_mean(function_num)
        #self.param = self.default_parameter_generation()
        self.trace_length, self.trace_section = self.default_length_para_generation()

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)

    def generate_function(self):
        for _ in range(self.function_num):
            function.append(random.choice(function_list))
        return function

    def generate_recipe_step_num(self, num_function):
        target_recipe_step = random.randint(1, num_function)
        if target_recipe_step == num_function:
            return list(np.ones(1) * num_function)
        
        result = [] 
        # 리스트의 길이가 1 이상이고, 원하는 합계가 양수인 경우에만 작동
        if target_recipe_step <= 0 or num_function <= 0:
            return result
        
        for _ in range(target_recipe_step - 1):
            # 남은 합계 중에서 랜덤한 숫자를 생성
            num = random.randint(1, num_function - (target_recipe_step - len(result)))
            # 리스트에 추가
            result.append(num)
            # 합계 업데이트
            num_function -= num
        
        # 마지막 숫자는 남은 합계
        result.append(num_function)
        
        return result

    def random_mean(self, step_len=3):

        mean_range = np.random.rand(step_len)
        mean_range = (mean_range - np.min(mean_range)) / (np.max(mean_range) - np.min(mean_range))
        mean_range = mean_range * (self.max_value - self.min_value) + self.min_value

        return mean_range

    def generate_length(self):
        # trace_len = samples = np.random.exponential(400, 1000)
        return max(self.function_num * 3, np.random.geometric(0.004, 1)[0])

    def generate_boundary(self, trace_length):

        length_list = np.random.dirichlet(np.ones(self.function_num))
        length_list = length_list / np.sum(length_list)
        trace_section = np.array(length_list * trace_length, dtype=int)
        #print(trace_section)
        # time parameter 는 최대 길이가 4인데, trace section의 길이가 4 미만인 경우와 만나면 에러가 생김 추후 고쳐야함
        trace_section = np.array([i if i >= 7 else 8 for i in trace_section])
        
        return trace_section
        
    def generate_value_parameter(self, length, loc_factor, std_factor=0.1):
        '''
        mean profile 값을 기준으로 global range * 0.1 std를 갖는 정규분포에서 start, end, high value 등을 뽑아냄
        '''
        #param_value = np.random.normal(loc=self.mean_profiles[key], scale=self.global_range * std_factor, size=len(is_values))
        param_value = np.random.normal(loc=loc_factor, scale=self.global_range * std_factor, size=length)

        return param_value

    def generate_time_parameter(self, key, is_times, trace_section):
        if len(is_times) == 1:
            time_target = np.random.uniform(low=0, high=1)
            time_target = [int(time_target * trace_section[key])]
            # print(time_target)

        elif len(is_times) > 1:
            #print(is_times)
            #print(trace_section[key])
            time_target = np.random.choice(np.arange(trace_section[key]), size=len(is_times), replace=False)
            time_target.sort()
            
            # time_list = np.random.dirichlet(np.ones(len(is_times)))
            # time_list = time_list / np.sum(time_list)
            # time_target = np.array(time_list * trace_section[key], dtype=int)
            # if len(set(time_target)) == len(time_target):
            #     pass
            # else:
            #     time_list = np.random.dirichlet(np.ones(len(is_times)))
            #     time_target = np.array(time_list * trace_section[key], dtype=int)
            # time_target.sort()
            

        # len(is_times) == 0
        else:
            time_target = -1

        return time_target
    
    def default_length_para_generation(self):
        # generate length
        trace_length = self.generate_length()
        
        # generate each length of each trace
        trace_section = self.generate_boundary(trace_length)
        trace_length = np.sum(trace_section)

        return trace_length, trace_section

    def default_parameter_generation(self):
        #self.seed_everything()
        # make parameter dictionary 
        param_list = self.generate_trace_parameter()

        # value default parameter generation
        for key, param in enumerate(param_list):
            is_values = [v for v in param if 'value' in v]
            param_value = self.generate_value_parameter(len(is_values), self.mean_profiles[key])

            # allocate value parameter
            for v, p in zip(is_values, param_value):
                param_list[key][v] = p
        
        # time default parameter generation
        for key, param in enumerate(param_list):
            is_times = [t for t in param if 'time' in t]
            param_value = self.generate_time_parameter(key, is_times, self.trace_section)

            # allocate time parameter
            if not is_times:
                pass
            else:
                for v, p in zip(is_times, param_value):
                    param_list[key][v] = p

        # allocate length
        for key, param in enumerate(param_list):
            param_list[key]['length'] = self.trace_section[key]

        return param_list
    
    def generate_trace_parameter(self):
        param_list = []
        for k, f in enumerate(self.function):
            trace_type = getattr(super(), f)
            param = inspect.signature(trace_type).parameters
            param = list(param.keys())
            value = np.ones(shape=len(param))
            param = dict(zip(param, value))
            param_list.append(param)
            #param_list.append(param)
        
        return param_list       
    
    def generate_random_trace(self, return_param=False):        
        #param = copy.deepcopy(self.param)
        param_list = []
        traces = []
        parameter = []
        for _ in range(self.n):
            parameter = []
            for i in range(self.para_num):
                step = []
                #param = copy.deepcopy(self.param)
                self.function = self.function_lists[i]
                print(self.function)
                param = self.default_parameter_generation()
                # # jitter 
                # is_times = [t for t in param[key] if 'time' in t]
                # time_values = [param[key][v] for v in is_times]
                    
                # new_length, new_time_values = self.jitter(3, param[key]['length'], *time_values)
                # param[key].update({'length': new_length, **dict(zip(is_times, new_time_values))})

                for key, f in enumerate(self.function):
                    func = getattr(self, f)
                    
                    # value parameter generation
                    is_values = [v for v in param[key] if 'value' in v]
                    for value in is_values:
                        # std factor random으로 줘야함
                        param_value = self.generate_value_parameter(1, param[key][value], std_factor=0.05)[0]
                        param[key][value] = param_value
                    
                    trace_value = func(**param[key])
                    step.append(trace_value)
                step = np.concatenate(step)
                plt.plot(step)
                plt.show()
                parameter.append(step)
            parameter = np.concatenate(parameter)
            #parameter = parameter.transpose()
            parameter = parameter.reshape(self.para_num, -1)
            parameter = parameter.transpose()
            traces.append(parameter)
        #return traces
        if return_param:
            return self.make_trace_format(*traces), param_list
        return self.make_trace_format(*traces)
    
    def jitter(self, maximum_point, length, *time):
        # length jitter factor
        length_jitter_factor = self.jitter_value_calculation(maximum_point)
        new_length = length + length_jitter_factor
        #print(f'length:{length} new_length:{new_length}')
        # time jitter factor
        if time:
            new_time = []
            for key, t in enumerate(time):
                time_jitter_factor = self.jitter_value_calculation(maximum_point)
                new_t = max(0, t + time_jitter_factor)
                new_time.append(new_t)

            new_time.sort()
            #print(f'time:{time} new_time:{new_time}')

            if max(new_time) >= new_length:
                new_length = max(new_time) + 1

            if len(set(new_time)) != len(time):
                return length, time

            return new_length, new_time

        else:
            return new_length, []
    
    def jitter_value_calculation(self, maximum_point):
        value = np.random.normal(1)
        intervals = np.linspace(-3, 3, maximum_point*2+1)
        jitter_score = list(range(-maximum_point+1, 1)) + list(range(0, maximum_point))

        idx = np.searchsorted(intervals, value, side='right')
        # less than 3
        if idx == 0:
            return jitter_score[0] - 1
        # grater than 3
        elif idx == len(intervals):
            return jitter_score[-1] + 1
        else:
            return jitter_score[idx - 1]

    def random_parameter_generation(self):
        # generate length
        self.trace_length = self.generate_length()
        trace_length = self.trace_length
        # generate each length of each trace
        self.trace_section = self.generate_boundary(trace_length)
        
        trace_section = self.trace_section

        full_param = []

        for key, f in enumerate(self.function):
            trace_type = getattr(super(), f)
            param_list = inspect.signature(trace_type).parameters
            param_list = list(param_list.keys())
            #print(param_list)
            value = np.ones(shape=len(param_list)) * -1
            param_dict = dict(zip(param_list, value))
            #print(param_dict)

            is_values = [s for s in param_list if 'value' in s]
            is_times = [s for s in param_list if 'time' in s]
            # is_coff = [s for s in param_list if 'b' or 'c' in s]

            # value parameter gen
            param_value = self.generate_value_parameter(key, is_values)

            # allocate value parameter
            for v, p in zip(is_values, param_value):
                param_dict[v] = p

            # allocate length
            param_dict['length'] = trace_section[key]

            # time parameter gen
            time_target = self.generate_time_parameter(key, is_times, trace_section)

            # allocate time parameter
            if not is_times:
                pass
            else:
                for t, p in zip(is_times, time_target):
                    param_dict[t] = p

            full_param.append(param_dict)
            #print(param_dict)

        return full_param

    def add_noise(self, value):
        noise = np.random.normal(loc=0, scale=self.global_range * 0.03, size=len(value))
        value = value + noise
        return value

    def generate_trace(self):
        traces = []
        for n in range(self.n):
            step = []
            for key, f in enumerate(self.function):
                func = getattr(self, f)
                value = func(**self.param[key])
                noised_value = self.add_noise(value)
                step.append(noised_value)
            traces.append(step)
        
        #return traces
        return self.make_trace_format(*traces)
    
    def make_multi_trace_format(self, para_num, *args):
        """
        Format multi-parameter traces into a single DataFrame.
        Args:
            para_num: Number of parameters
            args: List of trace data for each parameter
        Returns:
            DataFrame containing all traces with multi-parameter information
        """
        # Define base columns
        col = ['LOT_ID', 'WAFER_ID', 'PROCESS', 'PROCESS_STEP', 'RECIPE', 
            'RECIPE_STEP', 'PARAMETER_NAME', 'PARAMETER_VALUE', 'TIME']
        
        data = []

        for wafer_idx, trace_group in enumerate(args):  # Iterate through wafers
            # Flatten and combine traces for this wafer
            wafer_df = pd.DataFrame({
                'PARAMETER_VALUE': np.concatenate([np.concatenate(trace) for trace in trace_group]),
                'LOT_ID': 'lot',
                'WAFER_ID': f'wafer{wafer_idx + 1}',
                'PROCESS': 'process',
                'PROCESS_STEP': 'process_step',
                'RECIPE': 'recipe',
                'PARAMETER_NAME': [f'para{i}' for i in range(para_num) for trace in trace_group[i] for _ in trace],
                'RECIPE_STEP': [str(step_num + 1) for step_num, trace in enumerate(trace_group[0]) for _ in trace]
            })

            # Add time information
            wafer_df['TIME'] = pd.date_range("2024-01-01", periods=wafer_df.shape[0], freq="S")
            data.append(wafer_df)

        return pd.concat(data).reset_index(drop=True)
    #def make_multi_trace_format(self, para_num, *args):
        # make recipe step indices only in nonfocus cases
        
        col = ['Tool/Chamber', 'Tool', 'chamber', 'Process_Recipe', 'Start_Time',
               'Carrier_ID', 'WAFER_ID', 'Slot']
        
        param_col = [f'para{i}' for i in range(para_num)]

        col = ['LOT_ID', 'WAFER_ID', 'PROCESS', 'PROCESS_STEP', 'RECIPE', 'RECIPE_STEP', 'PARAMETER_NAME',
               'PARAMETER_VALUE', 'TIME']
        #return args
        data = []
        for num, trace in enumerate(args):
            df = pd.DataFrame([], columns=col)
            df.PARAMETER_VALUE = np.concatenate(trace) 
            df['Tool/Chamber'] = 'EP00018/PM-3'
            df['Tool'] = 123
            df['WAFER_ID'] = f'wafer{num+1}'
            df['PROCESS'] = 'process'
            df['PROCESS_STEP'] = 'process_step'
            df['RECIPE'] = 'recipe'
            df['PARAMETER_NAME'] = 'parameter_name'
            # step_list = []
            # for step_num, step in enumerate(single):
            #     recipe_step = [str(step_num+1)] * len(step)
            #     step_list.extend(recipe_step)
            df['RECIPE_STEP'] = [str(step_num+1) for step_num, step in enumerate(trace) for _ in step]
            data.append(df)
        data = pd.concat(data)
        data['TIME'] = pd.date_range("2024-01-01", periods=data.shape[0], freq="S")

        return data
       
    def make_trace_format(self, chamber=1, *args):
        # make recipe step indices only in nonfocus cases
        
        col = ['Tool/Chamber', 'Tool', 'chamber', 'Process_Recipe', 'Start_Time',
               'Carrier_ID', 'WAFER_ID', 'Slot']

        para_col = [f"para{p+1}" for p in range(self.para_num)]

        col = col + para_col

        #col = ['LOT_ID', 'WAFER_ID', 'PROCESS', 'PROCESS_STEP', 'RECIPE', 'RECIPE_STEP', 'PARAMETER_NAME',
        #       'PARAMETER_VALUE', 'TIME']
   
        data = []
        for key, wafer in enumerate(args):
            df = pd.DataFrame([], columns=col)
            df[para_col] = wafer
            df['Tool/Chamber'] = f'EP00018/PM-{chamber}]'
            df['Tool'] = 'EP00018'
            df['chamber'] = f'PM-{chamber}'
            df['Process_Recipe'] = 'Process Recipe'
            df['Carrier_ID'] = ''
            df['WAFER_ID'] = ''
            df['Slot']  = ''
            # step_list = []
            # for step_num, step in enumerate(single):
            #     recipe_step = [str(step_num+1)] * len(step)
            #     step_list.extend(recipe_step)
            df['RECIPE_STEP'] = [str(step_num+1) for step_num, step in enumerate(trace) for _ in step]
            data.append(df)
        data = pd.concat(data)
        data['TIME'] = pd.date_range("2024-01-01", periods=data.shape[0], freq="S")

        return data

    def trace_checker(self, df, marker=False):
        #fname = p.split(os.sep)[-1].split('.')[0]
        step_list = list(np.sort(np.unique(df.RECIPE_STEP.values)))
        reset_raw_data, boundary_raw = self.reset_index(df, step_list, True)
        #print(boundary_raw)
        self.index_sorted_image(reset_raw_data, boundary_raw, marker)

    def reset_index(self, trace, step_list, return_recipe_step_boundary=False):
        full_trace_list = []
        for i in trace['WAFER_ID'].unique():
            d = trace[trace.WAFER_ID == i].reset_index(drop=True)
            full_trace_list.append(d)

        # step_list = cleaned_data.RECIPE_STEP.unique()
        col_name = trace.columns.to_list()
        group_df = trace.groupby('WAFER_ID')
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
                        full_trace_list[key] = pd.DataFrame(np.insert(full_trace_list[key].values,
                                                                    new_index + i,
                                                                    values=[np.nan],
                                                                    axis=0),
                                                            columns=col_name)
                        full_trace_list[key].loc[new_index + i, 'RECIPE_STEP'] = step_number

        for key, f in enumerate(full_trace_list):
            full_trace_list[key] = f.dropna(subset=['WAFER_ID'])

        accumulated_index = np.cumsum(recipe_step_index_list)
        accumulated_index[-1] = accumulated_index[-1] - 1
        if return_recipe_step_boundary:
            return full_trace_list, accumulated_index
        else:
            return full_trace_list

    def index_sorted_image(self, reset_raw_data, boundary_raw, marker, path=None):
        plt.figure(figsize=(12, 6))

        plt.title(f'# of {len(reset_raw_data)} wafer')
        if marker:
            mark = 'o'
        else:
            mark = None
        for wafer in reset_raw_data:
            plt.plot(wafer.PARAMETER_VALUE, label=None, marker=mark)
            
        for i in boundary_raw:
            plt.axvline(i, linestyle='--', alpha=0.3, color='black')
        if path is not None:
            plt.savefig(path)
        plt.show()