
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

def load_data(filename):
    df = pd.read_excel(filename)
    df['Levertijd'] = pd.to_datetime(df['Levertijd'], format='%H:%M:%S').dt.round('1h')
    df['time'] = df['Levertijd'].dt.time
    df['hour'] = df['Levertijd'].dt.hour
    return df

def visualize_planning(df, save=False):
    if save:
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    b = 0
    agg_df = df.groupby(by=['hour']).agg({'Rolcontainers': 'sum'})
    agg_df.reset_index(drop=False, inplace=True)
    expected_df = pd.DataFrame(data={'hour': np.arange(24)})
    agg_df = agg_df.merge(expected_df, on='hour', how='outer')
    agg_df.fillna(0, inplace=True)
    agg_df.sort_values(by='hour', inplace=True)
    # sns.barplot(data=agg_df, x='hour', y='Rolcontainers', palette=colors_from_values(agg_df['Rolcontainers'], "flare"))
    sns.barplot(data=agg_df, x='hour', y='Rolcontainers', color='blue')

    plt.xlim((0, 23))
    plt.xlabel('Hour')
    plt.ylabel('Load carriers')

    plt.gcf().set_size_inches(w=6.25, h=3)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join('images', 'load_carriers.pgf'))
    else:
        plt.show()




if __name__ == '__main__':
    filename = os.path.join('data', 'truck_arrivals.xlsx')
    df = load_data(filename)
    visualize_planning(df, save=True)
