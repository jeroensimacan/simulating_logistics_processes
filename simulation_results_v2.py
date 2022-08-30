import os
from collections.abc import MutableMapping
from itertools import product

import numpy as np
import pandas as pd
import yaml
import pickle

pd.options.mode.chained_assignment = None

def load_max_queue_length():
    return pd.read_excel('max_lc.xlsx')


def gather_results_all_experiments(filepath):
    max_lc = load_max_queue_length()
    all_results = []
    all_results_5min = []
    queue_length_personnels = []
    queue_length_lcs = []
    elevator_information = []
    simulation_settings = []
    for directory in os.listdir(filepath):
        results, results_5min, queue_length_personnel, queue_length_lc = gather_results_experiments(os.path.join(filepath, directory), max_lc)
        all_results.append(results)
        all_results_5min.append(results_5min)
        queue_length_personnels.append(queue_length_personnel)
        queue_length_lcs.append(queue_length_lc)
        simulation_settings.append(read_settings(os.path.join(filepath, directory)))

        elevator_information.append(obtain_information_per_elevator(os.path.join(filepath, directory)))

    writer = pd.ExcelWriter('results_experiments.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    pd.concat(all_results).to_excel(writer, sheet_name='Overall statistics', index=None)
    pd.concat(all_results_5min).to_excel(writer, sheet_name='Waiting time per 5 min', index=None)
    pd.concat(queue_length_lcs).to_excel(writer, sheet_name='Queue length lc', index=None)
    pd.concat(queue_length_personnels).to_excel(writer, sheet_name='Queue length personnel', index=None)
    pd.concat(simulation_settings).to_excel(writer, sheet_name='Simulation settings', index=None)
    writer.save()

    pd.concat(elevator_information).to_csv('Elevator information.csv', index=None)

    # Close the Pandas Excel writer and output the Excel file.
    b = 0

def obtain_information_per_elevator(filepath):
    settings = read_settings(filepath)
    dfs = []
    for result in open_files(filepath):
        queue_length = determine_maximum_queue_length(result['elevator_groups'], result['run'])
        elevators_available = determine_elevators_available(result, settings)
        # personnel_arrival = determine_arrival_of_personnel(result)
        average_waiting_time_5min = determine_average_waiting_time(filepath, process='elevator', interval=300,
                                                                   start_time=0, end_time=86400)
        average_waiting_time_5min.rename(columns={'location': 'elevator_group'}, inplace=True)

        df = queue_length.merge(elevators_available, on=['rounded_time', 'elevator_group'])
        df = df.merge(average_waiting_time_5min, on=['rounded_time', 'elevator_group'])
        dfs.append(df)
    result = pd.concat(dfs)
    result['experiment_name'] = settings['experiment_name'].values[0]
    result['experiment_nr'] = settings['experiment_nr'].values[0]
    return result

def determine_maximum_queue_length_all(result):
    b = 0
    pass

def determine_maximum_queue_length(df, run, start_time=0, end_time=86400, freq=300):
    df['rounded_time'] = df['_time'] - df['_time'] % 300
    # get a df with all timestamps and locations, floors.

    rounded_time = np.arange(start_time, end_time + freq, freq)
    rounded_time, name, floor = pd.core.reshape.util.cartesian_product([rounded_time, df['name'].unique(), df['from_floor'].unique()])
    empty_df = pd.DataFrame(dict(rounded_time=rounded_time, name=name, floor=floor))
    empty_df['queue length lc'] = 0
    empty_df['queue length personnel'] = 0
    empty_df['run'] = run
    empty_df.set_index(['rounded_time', 'name', 'floor'], inplace=True)

    # group results per timestamp, location, floor and get max queue length for lc and personnel
    agg_df = df.groupby(by=['rounded_time', 'name', 'from_floor']).agg({'queue length lc': 'max', 'queue length personnel': 'max'})
    empty_df.update(agg_df)
    empty_df.reset_index(drop=False, inplace=True)
    empty_df.rename(columns={'name': 'elevator_group'}, inplace=True)
    return empty_df

def determine_elevators_available_all(result):
    b = 0
    pass

def determine_elevators_available(result, settings, start_time=0, end_time=86400, freq=300):
    b = 0
    mapping = {}
    for c in settings.columns:
        splitted = c.split('.')
        if splitted[0] != 'elevator_groups' or len(splitted) < 5:
            continue

        _, group_name, _, elevator_name, *_ = splitted

        if group_name in mapping and elevator_name in mapping[group_name]:
            continue
        elif group_name in mapping:
            mapping[group_name].append(elevator_name)
        else:
            mapping[group_name] = [elevator_name]

    mapping_df = pd.DataFrame.from_dict(mapping, orient='index').unstack().dropna().reset_index()
    mapping_df.columns = ['', 'elevator_group', 'aasCode']
    mapping_df.drop(columns='', inplace=True)
    available_elevators = mapping_df.groupby(by=['elevator_group']).agg({'aasCode': 'count'}).reset_index().rename(columns={'aasCode': 'available'})

    # filled df...
    rounded_time = np.arange(start_time, end_time + freq, freq)
    rounded_time, elevator_group = pd.core.reshape.util.cartesian_product([rounded_time, mapping_df['elevator_group'].unique()])
    empty_df = pd.DataFrame(dict(rounded_time=rounded_time, elevator_group=elevator_group))
    empty_df = empty_df.merge(available_elevators, on=['elevator_group'])

    df = result['individual_elevators']
    if 'BREAKDOWN' not in df['reportingStatus'].unique():
        return empty_df
    tmp_df = df.loc[df['reportingStatus'] == 'BREAKDOWN']
    tmp_df['start_time'] = tmp_df['_time'] - tmp_df['_time'] % freq
    tmp_df['end_time'] = tmp_df['start_time'] + (tmp_df['duration'] - tmp_df['duration'] % freq) + freq
    lmd = lambda row: np.arange(row['start_time'], row['end_time'], freq)
    tmp_df['rounded_time'] = tmp_df.apply(lmd, axis=1)
    tmp_df = tmp_df.explode(column='rounded_time')
    tmp_df = tmp_df.merge(mapping_df)

    agg_df = tmp_df.groupby(by=['elevator_group', 'rounded_time']).agg({'start_time': 'count'}).reset_index().rename(columns={'start_time': 'cnt'})
    agg_df = agg_df.merge(available_elevators, on='elevator_group')
    agg_df['available'] -= agg_df['cnt']
    agg_df.set_index(['elevator_group', 'rounded_time'], inplace=True)

    empty_df.set_index(['elevator_group', 'rounded_time'], inplace=True)
    empty_df.update(agg_df)
    empty_df.reset_index(inplace=True)
    empty_df.rename(columns={'available': 'elevators available'}, inplace=True)
    return empty_df



def determine_arrival_of_personnel(result):
    b = 0
    pass


def gather_results_experiments(filepath, max_lc):
    simulation_settings = read_settings(filepath)
    start_time = simulation_settings['start_time_simulation'].values[0]
    end_time = simulation_settings['end_time_simulation'].values[0]

    average_waiting_time_5min = determine_average_waiting_time(filepath, process='elevator', interval=300, start_time=start_time, end_time=end_time)
    average_waiting_time_day = determine_average_waiting_time(filepath, process='elevator', interval=86400, start_time=start_time, end_time=end_time)
    exceeded_waiting_time = determine_average_waiting_time_exceeded(filepath, process='elevator', interval=300, start_time=start_time, end_time=end_time)
    queue_length_lc = obtain_time_per_queue_length(filepath, start_time=start_time, end_time=end_time, queue_length_col='queue length lc')
    queue_length_personnel = obtain_time_per_queue_length(filepath, start_time=start_time, end_time=end_time, queue_length_col='queue length personnel')
    exceeded_queue_length = determine_queue_length_exceeded(queue_length_lc, max_lc)
    # max_queue_length_per_interval = determine_max_queue_length_per_interval(filepath)
    average_waiting_time_day = average_waiting_time_day.drop(columns=['rounded_time'])
    results = average_waiting_time_day.merge(exceeded_waiting_time, on='location').merge(exceeded_queue_length, on='location')
    results = pd.concat([simulation_settings, results], axis=1)
    results.ffill(inplace=True)

    queue_length_lc = pd.concat([queue_length_lc, simulation_settings], axis=1)
    queue_length_lc.ffill(inplace=True)

    queue_length_personnel = pd.concat([queue_length_personnel, simulation_settings], axis=1)
    queue_length_personnel.ffill(inplace=True)

    average_waiting_time_5min = pd.concat([average_waiting_time_5min, simulation_settings], axis=1)
    average_waiting_time_5min.ffill(inplace=True)
    return results, average_waiting_time_5min, queue_length_personnel, queue_length_lc



def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_settings(filepath):
    experiment_name, experiment_nr = os.path.split(filepath)[-1].split('_experiment_')
    with open(os.path.join(filepath, 'settings.yaml'), 'r') as f:
        settings = yaml.safe_load(f)
    settings = flatten_dict(settings)
    for k, v in settings.items():
        if isinstance(v, list):
            settings[k] = str(v)
    settings['experiment_name'] = experiment_name
    settings['experiment_nr'] = experiment_nr

    return pd.DataFrame(settings, index=[0])


def open_files(filepath):
    for file in os.listdir(filepath):
        filename = os.path.join(filepath, file)
        f, extension = os.path.splitext(file)
        if extension != '.pkl':
            continue
        run = f.rsplit('_', 1)[1]
        with open(filename, 'rb') as f:
            output = pickle.load(f)
        output['run'] = run
        yield output


def determine_average_waiting_time_exceeded(filepath, process='elevator', interval=300, start_time=0, end_time=86400):
    dfs = []
    for results in open_files(filepath):
        df = determine_average_waiting_time_per_interval(results['users'], process=process, interval=interval, start_time=start_time, end_time=end_time)

        filled_df = pd.DataFrame({'location': df['location'].unique(), 'duration': [0] * len(df['location'].unique())})
        filled_df.set_index('location', inplace=True)

        agg_df = df.loc[df['duration'] > 90].groupby(by='location').agg({'duration': 'count'})
        # agg_df = df
        agg_df['duration'] *= interval
        filled_df.update(agg_df)
        dfs.append(filled_df)
    result = pd.concat(dfs).groupby(by=['location']).agg({'duration': 'mean'}).reset_index()
    return result.rename(columns={'duration': 'duration waiting time exceeded'})

def determine_average_waiting_time(filepath, process='elevator', interval=300, start_time=0, end_time=86400):
    dfs = []
    for results in open_files(filepath):
        # do something with the output
        df = determine_average_waiting_time_per_interval(results['users'], process=process, interval=interval, start_time=start_time, end_time=end_time)
        dfs.append(df)
    result = pd.concat(dfs).groupby(by=['rounded_time', 'location']).agg({'duration': 'mean'}).reset_index()
    return result.rename(columns={'duration': 'average waiting time'})


def determine_average_waiting_time_per_interval(df, process, interval=300, start_time=0, end_time=86400):
    rounded_time = np.arange(start_time, end_time, interval)

    df = df.loc[(df['process'] == process) & (df['event'] == 'waiting')]
    df['rounded_time'] = (df['time'] // interval) * interval
    rounded_time, location = pd.core.reshape.util.cartesian_product([rounded_time, df['location'].unique()])
    empty_df = pd.DataFrame(dict(rounded_time=rounded_time, location=location))
    empty_df.set_index(['rounded_time', 'location'], inplace=True)
    empty_df['duration'] = 0

    # insert new rows where waiting time > interval...
    n = (df['duration'].max() // interval) + 2
    dfs = []
    df['end_time'] = df['time'] + df['duration']
    for i in np.arange(n):
        tmp_df = df.loc[df['duration'] - i * interval >= 0]
        if i > 0:
            tmp_df['rounded_time'] += i * interval
            tmp_df['duration'] = tmp_df['end_time'] - tmp_df['rounded_time']
        dfs.append(tmp_df)
    agg_df = pd.concat(dfs).groupby(by=['rounded_time', 'location']).agg({'duration': 'mean'})
    empty_df['duration'].update(agg_df['duration'])
    empty_df.reset_index(drop=False, inplace=True)
    return empty_df


def obtain_time_per_queue_length(filepath, start_time=0, end_time=86400, queue_length_col='queue length lc'):
    result = pd.concat([
        obtain_queue_lengths(result['elevator_groups'], start_time=start_time, end_time=end_time, queue_length_col=queue_length_col) for result in open_files(filepath)
    ])
    return result.groupby(by=['name', 'from_floor', 'queue length lc']).agg({'duration': 'mean'}).reset_index()


def obtain_queue_lengths(tmp_df, start_time=0, end_time=86400, queue_length_col='queue length lc'):
    duration = end_time - start_time
    tmp_df.sort_values(by=['name', 'from_floor', '_time'], inplace=True)
    tmp_df['end time'] = tmp_df['_time'].shift(-1)
    tmp_df['duration'] = tmp_df['end time'] - tmp_df['_time']
    tmp_df = tmp_df.loc[(tmp_df['name'] == tmp_df['name'].shift(-1)) & (
                tmp_df['from_floor'] == tmp_df['from_floor'].shift(-1))].reset_index(drop=True)

    agg_df = tmp_df.groupby(by=['name', 'from_floor', 'queue length lc']).agg({'duration': 'sum'})
    filled_df = pd.DataFrame(
        product(tmp_df['name'].unique(), tmp_df['from_floor'].unique(), np.arange(26, -1, -1)),
        columns=['name', 'from_floor', 'queue length lc'])
    filled_df.set_index(['name', 'from_floor', 'queue length lc'], inplace=True)
    filled_df['duration'] = 0
    filled_df.update(agg_df)
    filled_df.reset_index(drop=False, inplace=True)
    filled_df['duration'] = filled_df.groupby(by=['name', 'from_floor'])['duration'].transform(pd.Series.cumsum)
    filled_df.reset_index(drop=False, inplace=True)
    filled_df.loc[filled_df['queue length lc'] == 0, 'duration'] = duration
    return filled_df

def determine_queue_length_exceeded(max_queue_length, max_lc):
    result = max_queue_length.merge(max_lc, left_on=['name', 'from_floor', 'queue length lc'], right_on=['location', 'from_floor', 'max'])
    result = result[['location', 'from_floor', 'duration']]
    return result.rename(columns={'duration': 'duration queue length exceeded'})

if __name__ == '__main__':
    filepath = r'results_v2'
    gather_results_all_experiments(filepath)
