
"""
Simple elevator demo using events to implements a subscribe, broadcast pattern to let passengers know when 
they have reached there floor.  All the passengers getting off on the same floor are waiting on the 
same one event.

Programmer: Michael R. Gibbs
"""
import numpy as np
import pandas as pd
import simpy
import os
import pickle

import yaml
from scipy.stats import poisson
from simpy import AllOf

from goederenstraat_v2 import Goederenstraat_v2
from load_carrier import User
from truck import Truck

START_TIME_SIMULATION = 0
END_TIME_SIMULATION = 24 * 60 * 60
DAY_OF_WEEK = 'Monday'



def create_rc_planning(df, pax, pax_to_rc_mapping):
    pax_to_rc_mapping['rc needed'] = (pax_to_rc_mapping['bias'] + pax_to_rc_mapping['coef'] * pax) / pax_to_rc_mapping['articles per lc']
    df = df.merge(pax_to_rc_mapping[['Retail sector', 'rc needed']])
    df['demand v2'] = df['verdeelsleutel'] * df['rc needed']
    df['demand'] = df['demand v2'].astype(int)
    agg_df = df.groupby(by='Retail sector').agg({'demand': 'sum'}).reset_index().rename(columns={'demand': 'planned rc'})
    df = df.merge(agg_df, on=['Retail sector'])
    df['missing rc'] = df['planned rc'] - df['rc needed']
    return df


def create_truck_planning(df):
    arrival_times = pd.read_excel(os.path.join('data', 'truck_arrivals.xlsx'))
    trucks = []
    for business_partner in df['Business partner'].unique():
        capacity = 25

        eta = np.random.choice(arrival_times.loc[arrival_times['Business partner'] == business_partner, 'Delivery time'])
        eta += np.random.normal(0, 10*60)
        tmp_df = df.loc[df['Business partner'] == business_partner]
        truck = Truck(eta=eta, capacity=capacity, business_partner=business_partner)
        for idx, row in tmp_df.iterrows():
            for _ in range(row['demand']):
                if truck.load == truck.capacity:
                    trucks.append(truck)
                    eta = np.random.choice(arrival_times.loc[arrival_times['Business partner'] == business_partner, 'Delivery time'])
                    eta += np.random.normal(0, 10 * 60)
                    truck = Truck(eta=eta, capacity=capacity, business_partner=business_partner)
                load_carrier = User(user_type='Load carrier', current_floor=0, destination_floor=row['floor'], area=0.8,
                                    elevator=row['elevator'], x=row['x'], y=row['y'], z=row['floor'],
                                    branch=business_partner)
                truck.add_load_carrier(load_carrier)
        trucks.append(truck)
    return trucks


def process_truck(env, truck, goederenstraat):
    until_eta = truck.eta - env.now
    print(f'Processing truck with eta: {truck.eta} and load: {truck.load}')
    yield env.timeout(until_eta)

    global users
    while truck.load > 0:
        load_carrier = truck.remove_load_carrier(index=0)
        users.append(load_carrier)
        env.process(process_load_carrier_with_store(env, load_carrier, goederenstraat))


def gen_retail_personnel(env, goederenstraat):
    for i in range(20):
        employee = User(user_type='personnel', current_floor=0, destination_floor=1, branch='Retail and catering')
        env.process(move_lc_from_store_to_shop(env, employee, goederenstraat))
        yield env.timeout(60)


def use_elevator_generator(env, elevator_manager, load_carrier, current_floor, destination_floor):
    with elevator_manager['process'].request() as request:
        yield request
        yield env.process(
            use_elevator(elevator_manager['management_tool'], load_carrier, current_floor, destination_floor))


def use_elevator(goederenstraat, load_carrier, start_floor, end_floor):
    """
        process for using a elevator to move from one floor to another
    """
    yield goederenstraat.move_to(load_carrier, start_floor, end_floor)



def move_lc_from_store_to_shop(env, employee, goederenstraat):
    # process starts with picking up one or multiple load carriers
    current_location = None
    while True:
        load_carrier = yield goederenstraat.storage.get()
        t1 = env.timeout(60)
        load_carrier_events = [goederenstraat.storage.get(lambda lc: lc.elevator == load_carrier.elevator and lc.branch == load_carrier.branch) for _ in
                               range(1)]

        yield t1 | AllOf(env, load_carrier_events)

        # what is the position of the employee?
        location = obtain_security_name(load_carrier, goederenstraat)
        if current_location is None:
            pass
        elif location == current_location:
            pass
        else:
            t = goederenstraat.graph.graph.nodes.get(location)
            u = goederenstraat.graph.graph.nodes.get(current_location)
            distance, path = goederenstraat.graph.calculate_shortest_path(start_point=[t['x'], t['y'], t['z']], end_point=[u['x'], u['y'], u['z']])
            elevators = [p for p in path if p.startswith('l0')]
            elevator = elevators[0].split(' ')[0]
            elevator_management_tool = goederenstraat.elevators[elevator]
            yield env.process(use_elevator_generator(env, elevator_management_tool, employee, 0, 1))
            yield env.timeout(distance / (5 / 3.6))
            elevator = elevators[-1].split(' ')[0]
            elevator_management_tool = goederenstraat.elevators[elevator]
            yield env.process(use_elevator_generator(env, elevator_management_tool, employee, 1, 0))

        lcs = [load_carrier] + [lc.value for lc in load_carrier_events if lc.processed]
        for load_carrier in lcs:
            load_carrier.add_event(time=env.now, process='storage', event='finished', location='storage')

        elevator = match_elevator(lcs[0].elevator, goederenstraat)
        current_floor, destination_floor = lcs[0].current_floor, lcs[0].destination_floor
        elevator_management_tool = goederenstraat.elevators[elevator]

        results = [
            env.process(use_elevator_generator(env, elevator_management_tool, lc, current_floor, destination_floor)) for
            lc in [employee] + lcs
        ]

        yield AllOf(env, results)

        # insert walking time to shop and back here!
        load_carrier.add_event(env.now, 'distribution', 'waiting', location='terminal')
        with goederenstraat.distribution.request() as request:
            yield request
            yield env.process(goederenstraat.handle_distribution(employee, lcs))

        results = [
            env.process(use_elevator_generator(env, elevator_management_tool, lc, destination_floor, current_floor)) for
            lc in [employee] + lcs
        ]

        yield AllOf(env, results)
        current_location = location


def obtain_security_name(load_carrier, goederenstraat):
    security_names = []
    for n1, n2 in goederenstraat.graph.graph.edges(f'{load_carrier.elevator} 0'):
        security_names.append(n1) if 'security' in n1 else None
        security_names.append(n2) if 'security' in n2 else None
    if len(security_names) == 0:
        raise Exception(f'No security found for load_carrier: {load_carrier.id}')
    return security_names[0]


def process_load_carrier_with_store(env, load_carrier, goederenstraat):
    print(f'{env.now:.2f}: Load carrier {load_carrier.id} arrived at security')
    security_name = obtain_security_name(load_carrier, goederenstraat)
    load_carrier.add_event(time=env.now, event='waiting', process='security', location=security_name)
    with goederenstraat.security[security_name].request() as request:
        yield request
        yield env.process(goederenstraat.handle_security(load_carrier, security_name))
    print(f'{env.now:.2f}: Load carrier {load_carrier.id} processed at security')

    load_carrier.add_event(time=env.now, process='storage', event='started', location='storage')
    goederenstraat.storage.put(load_carrier)


def gen_load_carriers_from_csv(env, goederenstraat, shop_locations, pax, pax_to_rc_mapping):
    rc_needed = create_rc_planning(shop_locations, pax, pax_to_rc_mapping)
    trucks = create_truck_planning(rc_needed)
    trucks.sort(key=lambda x: x.eta)
    for truck in trucks:
        yield env.process(process_truck(env, truck, goederenstraat))


def match_elevator(elevator, goederenstraat):
    if not isinstance(elevator, str) and np.isnan(elevator):
        raise Exception('Unknown elevator!')
    for key in goederenstraat.elevators.keys():
        if elevator.lower() in key.lower():
            return key.lower()
    else:
        raise Exception('Unknown elevator!')


def gen_all_personnel(env, goederenstraat, simulation_settings={}):
    total_pax = simulation_settings['pax']
    day_of_week = simulation_settings['day_of_week']

    filepath = simulation_settings['filepaths']['input_pdfs_personnel']
    for filename in os.listdir(filepath):
        if not filename.endswith('.pkl'):
            continue

        with open(os.path.join(filepath, filename), 'rb') as f:
            settings = pickle.load(f)

        if len([node for node in goederenstraat.graph.graph.nodes() if settings['elevator'].lower() in node]) == 0:
            print(f'Warning, elevator {settings["elevator"]} is not found in nodes!')
            continue

        elevator = match_elevator(settings['elevator'], goederenstraat)
        if not elevator:
            continue

        branch_name = settings['branch_name']
        employee_factor = determine_employee_factor(simulation_settings, elevator, branch_name)
        nr_employees = employee_factor * (total_pax * settings['coef'] + settings['seasonality'].get(day_of_week) + settings['bias'])
        if employee_factor == 0 or nr_employees <= 0:
            continue
        cdf = pd.DataFrame(settings['data'])
        env.process(
            gen_personnel(env, nr_employees, goederenstraat, cdf, elevator=elevator, branch=settings['branch_name']))
        yield env.timeout(0.001)


def determine_employee_factor(simulation_settings, elevator, branch_name):
    employee_factor = 1
    if 'elevator_users' not in simulation_settings:
        print(f'Elevator users not found in the simulation settings')
        print('Setting the employee factor to 1.')
    elif elevator not in simulation_settings['elevator_users']:
        print(f'Elevator {elevator} not found in the user settings')
        print('Setting the employee factor to 1.')
    elif branch_name not in simulation_settings['elevator_users'][elevator]:
        print(f'Branch {branch_name} not found in the user settings for elevator: {elevator}')
        print('Setting the employee factor to 1.')
    else:
        employee_factor = simulation_settings['elevator_users'][elevator][branch_name]
    return employee_factor


def gen_timing_of_personnel(nr_employees, df):
    lambda_star = df['pdf'].max() * nr_employees
    df['acceptance_probability'] = df['pdf'] / df['pdf'].max()
    p = poisson(3600 / lambda_star)
    try:
        interval = p.rvs(size=int(lambda_star * 2 * (END_TIME_SIMULATION - START_TIME_SIMULATION) / 3600))
    except Exception as e:
        b = 0
    time = np.cumsum(interval)
    tmp_df = pd.DataFrame(data={'time': time})
    tmp_df['HOUR'] = tmp_df['time'] // 3600
    tmp_df = tmp_df.merge(df[['HOUR', 'acceptance_probability']], on='HOUR')
    tmp_df['random_value'] = np.random.random(size=tmp_df.shape[0])
    tmp_df = tmp_df.loc[tmp_df['random_value'] <= tmp_df['acceptance_probability']]
    if tmp_df.empty:
        interval = []
    elif tmp_df.shape[0] == 1:
        interval = [tmp_df['time'].values[0]]
    else:
        interval = [tmp_df['time'].values[0], *tmp_df['time'].diff().values[1:]]
    return interval

def determine_area(branch):
    if branch == 'Construction':
        area = np.random.uniform(low=0.35, high=2)
    elif branch == 'Cleaning and facilities':
        area = 1.
    else:
        area = 0.35
    return area


def determine_user_type(branch):
    if branch == 'Construction':
        user_type = 'Load carrier'
    else:
        user_type = 'personnel'
    return user_type


def gen_personnel(env, nr_employees, goederenstraat, df, elevator, branch):
    try:
        interval = gen_timing_of_personnel(nr_employees, df)
        for timeout in interval:
            yield env.timeout(timeout)
            area = determine_area(branch)
            user_type = determine_user_type(branch)
            if branch == 'Cleaning and facilities':
                available_floors = goederenstraat.elevators[elevator]['management_tool'].elevators[0].floors
                start_floor, end_floor = np.random.choice(available_floors, size=2, replace=False)
                for i in range(4):
                    env.process(process_user(env, goederenstraat, branch=branch, elevator=elevator, area=1., start_floor=start_floor, end_floor=end_floor, user_type='Load carrier'))
            env.process(process_user(env, goederenstraat, branch=branch, elevator=elevator, area=area, user_type=user_type))
    except Exception as e:
        b = 0


def process_user(env, goederenstraat, branch, elevator, area, start_floor=None, end_floor=None, user_type=''):
    if start_floor is None and end_floor is None:
        available_floors = goederenstraat.elevators[elevator]['management_tool'].elevators[0].floors
        start_floor, end_floor = np.random.choice(available_floors, size=2, replace=False)
    user = User(user_type=user_type, current_floor=start_floor, destination_floor=end_floor, branch=branch, area=area)

    global users
    users.append(user)

    elevator_management_tool = goederenstraat.elevators[elevator]
    yield env.process(use_elevator_generator(env, elevator_management_tool, user, user.current_floor, user.destination_floor))
    user.add_event(env.now, process='finished', event='finished', location='finished')


def load_simulation_settings(settings_filename, pax=None):
    with open(settings_filename, 'r') as f:
        simulation_settings = yaml.safe_load(f)
    if pax is not None:
        simulation_settings['pax'] = pax
    return simulation_settings


def find_or_create_output_dir(settings_filename, simulation_settings, output_dir):
    experiment_name = os.path.splitext(os.path.split(settings_filename)[-1])[0]
    # possible_path = os.path.join(output_dir, experiment_names)
    matching_paths = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)) and f.startswith(experiment_name)]
    for directory in matching_paths:
        settings_path = os.path.join(output_dir, directory, 'settings.yaml')
        old_settings = load_simulation_settings(settings_path)
        if old_settings == simulation_settings:
            output_dir = os.path.join(output_dir, directory)
            break
    else:
        if len(matching_paths) == 0:
            experiment_nr = 0
        else:
            experiment_nr = max([int(d.rsplit('_', 1)[1]) for d in matching_paths]) + 1
        output_dir = os.path.join(output_dir, f'{experiment_name}_experiment_{experiment_nr}')
        os.mkdir(output_dir)
        with open(os.path.join(output_dir, 'settings.yaml'), 'w') as f:
            yaml.dump(simulation_settings, f)
    return output_dir

def prepare_simulation_run(output_dir, nr_runs, settings_filename, ignore_irrelevant_personnel):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, f'{pax}_pax_{os.path.splitext(settings_filename)[0]}')
    if ignore_irrelevant_personnel:
        output_dir += '_ignore_irrelevant_personnel'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    elif len(os.listdir(output_dir)) < nr_runs:
        for filename in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, filename))
    return output_dir


def load_shop_locations(simulation_settings):
    return pd.read_excel(simulation_settings['filepaths']['shop_locations'])


def load_pax_to_rc_mapping(simulation_settings):
    return pd.read_excel(simulation_settings['filepaths']['pax_to_rc_mapping'])


def execute_simulation_run(pax=100_000, seed=1234, nr_runs=10, settings_filename='settings.yaml', output_dir='results_v2'):
    np.random.seed(seed)

    # initialize and load settings
    simulation_settings = load_simulation_settings(settings_filename, pax)
    output_dir = find_or_create_output_dir(settings_filename, simulation_settings, output_dir)

    if len([f for f in os.listdir(output_dir) if os.path.splitext(f)[1] == '.pkl']) >= nr_runs:
        return

    shop_locations = load_shop_locations(simulation_settings)
    pax_to_rc_mapping = load_pax_to_rc_mapping(simulation_settings)

    for run in range(nr_runs):
        global users
        users = []
        env = simpy.Environment(START_TIME_SIMULATION)

        goederenstraat = Goederenstraat_v2(env, simulation_settings)

        env.process(gen_all_personnel(env, goederenstraat=goederenstraat, simulation_settings=simulation_settings))
        env.process(
            gen_load_carriers_from_csv(env, goederenstraat, shop_locations, pax, pax_to_rc_mapping=pax_to_rc_mapping))
        env.process(gen_retail_personnel(env, goederenstraat))

        env.run(END_TIME_SIMULATION)

        user_history = pd.concat([user.get_history() for user in users])
        elevator_results = pd.concat([v['management_tool'].to_pandas() for v in goederenstraat.elevators.values()])
        elevator_results.reset_index(drop=True, inplace=True)
        individual_elevators = []
        for key in goederenstraat.elevators:
            for el in goederenstraat.elevators[key]['management_tool'].elevators:
                individual_elevators.append(el.to_pandas())
        individual_elevators = pd.concat(individual_elevators)

        output = {
            'individual_elevators': individual_elevators,
            'elevator_groups': elevator_results,
            'users': user_history
        }

        output_path = os.path.join(output_dir, f'output_run_{run}.pkl')

        with open(output_path, 'wb') as f:
            pickle.dump(output, f)


if __name__ == '__main__':
    settings_filepath = os.path.join('settings', 'regular_settings.yaml')
    # for settings_filename in os.listdir('settings'):
    #     settings_filepath = os.path.join('settings', settings_filename)
    for pax in [180000]: #, 200000, 240000, 264000, 278400, 28800, 300000]:
        execute_simulation_run(pax=pax, seed=1234, nr_runs=10, settings_filename=settings_filepath)

