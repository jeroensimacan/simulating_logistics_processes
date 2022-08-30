
import pandas as pd
import simpy
import numpy as np
import yaml

from elevator_manager import ElevatorManager
from schiphol_graph import SchipholGraph


class Goederenstraat_v2:

    def __init__(self, env, elevator_settings):
        self.env = env
        self.graph = SchipholGraph()
        self.security_v2 = simpy.Resource(env)
        self.distribution = simpy.Resource(env)
        self.elevators = self.create_elevators(elevator_settings['elevator_groups'])
        self.security = self.create_security()
        self.elevator = simpy.Resource(env, capacity=100_000)
        self.storage = simpy.FilterStore(env, capacity=100_000)
        b = 0

    def handle_security(self, load_carrier, security_name):
        # security times should be higher for incoming vs. outgoing load carriers
        security_time = np.random.randint(low=15, high=20)
        load_carrier.add_event(self.env.now, 'security', 'processing', location=security_name)
        yield self.env.timeout(security_time)
        load_carrier.add_event(self.env.now, 'security', 'finished', location=security_name)

    def handle_distribution(self, employee, load_carriers):
        start_point = self.graph.graph.nodes().get(load_carriers[0].elevator + ' 0')
        start_point = [start_point['x'], start_point['y'], start_point['z']]
        waypoints = [[lc.y, lc.x, lc.z] for lc in load_carriers]
        distance = self.graph.calculate_shortest_path_distance(start_point, start_point, waypoints)

        distribution_time = distance / (5 / 3.6)
        for load_carrier in load_carriers + [employee]:
            load_carrier.add_event(self.env.now, 'distribution', 'processing', location='terminal')
        yield self.env.timeout(distribution_time)
        for load_carrier in load_carriers + [employee]:
            load_carrier.add_event(self.env.now, 'distribution', 'finished', location='terminal')

    def create_elevators(self, elevator_settings):
        b = 0
        elevator_groups = {}
        for group_name, elevator_manager_settings in elevator_settings.items():
            em = ElevatorManager(self.env, elevator_manager_settings, name=group_name)
            elevator_groups[group_name] = {
                'management_tool': em,
                'process': simpy.Resource(self.env, capacity=100_000)
            }

        return elevator_groups

    def create_security(self):
        b = 0
        security_names = []
        for elevator in self.elevators:
            node_name = f'{elevator} 0'
            edges = self.graph.graph.edges(node_name)
            for n1, n2 in edges:
                if 'security' in n1:
                    security_names.append(n1)
                if 'security' in n2:
                    security_names.append(n2)
        security_names = list(set(security_names))
        security_groups = {
            name: simpy.Resource(self.env, capacity=1) for name in security_names
        }
        return security_groups


if __name__ == '__main__':
    with open('settings/settings.yaml', 'r') as f:
        elevator_settings = yaml.safe_load(f)
    Goederenstraat_v2(simpy.Environment(), elevator_settings)