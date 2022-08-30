import numpy as np
import pandas as pd

settings = pd.read_excel(r"C:\Users\Jeroen\Documents\University\Master Data Science and Technology\Master Thesis\Schiphol Group\Python\Simulatie\elevator_simulation_v2\results_experiments.xlsx", sheet_name='Simulation settings')

result = {}
for c in settings.columns:
    if not c.startswith('elevator_group') or len(c.split('.')) < 4:
        continue
    _, group, _, elevator, *_ = c.split('.')
    if group in result.keys() and elevator in result[group]:
        continue
    elif group in result.keys():
        result[group].append(elevator)
    else:
        result[group] = [elevator]

# df with experiment name, experiment number, elevator group, max elevators

for key, value in result.items():
    settings[key] = len(value)

for idx, row in settings.iterrows():
    for key in result.keys():
        for value in result[key]:
            name = f'elevator_groups.{key}.elevators.{value}.move_time'
            if not np.isnan(row[name]):
                continue
            settings.loc[idx, key] -= 1
            print(name)

    b = 0

df = pd.read_csv(r"C:\Users\Jeroen\Documents\University\Master Data Science and Technology\Master Thesis\Schiphol Group\Python\Simulatie\elevator_simulation_v2\Elevator information.csv")
agg_df = df.groupby(by=['experiment_name', 'experiment_nr', 'elevator_group']).agg({'available': 'max'}).reset_index()
agg_df.rename(columns={'available': 'max el available'}, inplace=True)
df = df.merge(agg_df, on=['experiment_name', 'experiment_nr', 'elevator_group'])
df.loc[df['available'] == df['max el available'], 'breakdown'] = False
df.loc[df['available'] != df['max el available'], 'breakdown'] = True
df.loc[(df['average waiting time'] <= 90) & (~df['breakdown']), 'waiting time exceeded'] = False
df.loc[(df['average waiting time'] > 90) & (~df['breakdown']), 'waiting time exceeded'] = True
df.loc[(df['average waiting time'] <= 150) & (df['breakdown']), 'waiting time exceeded'] = False
df.loc[(df['average waiting time'] > 150) & (df['breakdown']), 'waiting time exceeded'] = True


agg_df = df.groupby(['experiment_name', 'experiment_nr', 'elevator_group', 'waiting time exceeded']).agg({'waiting time exceeded': 'count'})
agg_df['waiting time exceeded'] *= 300
agg_df = agg_df.rename(columns={'waiting time exceeded': 'time'}).reset_index()

relevant_df = settings[['experiment_name', 'experiment_nr', 'pax']]
result = agg_df.merge(relevant_df, on=['experiment_name', 'experiment_nr'])
result.to_excel('final_results.xlsx')
