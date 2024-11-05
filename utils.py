import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def reset_index(trace, step_list, return_recipe_step_boundary=False):
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

def index_sorted_image(reset_raw_data, boundary_raw, fname, path=None):
    plt.figure(figsize=(12, 6))

    plt.title(f'{fname}: # of {len(reset_raw_data)} wafer')
    for wafer in reset_raw_data:
        if pd.unique(wafer.TYPE) == 'focus':
            plt.plot(wafer.PARAMETER_VALUE, color='red')
        else:
            plt.plot(wafer.PARAMETER_VALUE, label=None)

    for i in boundary_raw:
        plt.axvline(i, linestyle='--', alpha=0.3, color='black')
    if path is not None:
        plt.savefig(path)
    plt.show()