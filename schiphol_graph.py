
import itertools
from itertools import product, permutations
from math import radians

import geojson
from matplotlib.patches import Rectangle, Polygon
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point

import os
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

# from create_graph.create_graph import load_basemap

def load_basemap(ax):
    filepath, _ = os.path.split(__file__)
    with open(os.path.join(filepath, 'data', 'basemap.geojson'), 'r') as f:
        test = geojson.load(f)

    for el in test[0]['geometry']['coordinates']:
        for e in el:
            # np.array(e)
            ax.add_patch(Polygon(np.array(e), closed=True, color='grey'))
    all_x = [f[0] for el in test[0]['geometry']['coordinates'] for e in el for f in e]
    all_y = [f[1] for el in test[0]['geometry']['coordinates'] for e in el for f in e]
    dx = max(all_x) - min(all_x)
    dy = max(all_y) - min(all_y)
    plt.xlim([min(all_x) - 0.0 * dx, max(all_x) + 0.0 * dx])
    plt.ylim([min(all_y) - 0.0 * dy, max(all_y) + 0.0 * dy])
    return ax


class SchipholGraph:

    def __init__(self):
        self.graph = self.create_graph()

    def visualize_floor(self, floor=2, df=None):
        fig, ax = plt.subplots()
        ax = load_basemap(ax)
        for e1, e2 in self.graph.edges():
            n1 = self.graph.nodes().get(e1)
            n2 = self.graph.nodes().get(e2)
            if n1['z'] != floor or n2['z'] != floor:
                continue
            plt.plot([n1['y'], n2['y']], [n1['x'], n2['x']], 'k', linewidth=5)

        for node, values in self.graph.nodes(data=True):
            if values['z'] != floor:
                continue
            if 'l0' in node.lower():
                ax.plot(values['y'], values['x'], 'yo', markersize=10)
            else:
                ax.plot(values['y'], values['x'], 'ro', markersize=10)

        if df is not None:
            tmp = df.loc[df['floor'] == floor]
            plt.plot(tmp['x'], tmp['y'], 'co')
            b = 0

        plt.axis('off')
        plt.tight_layout()
        # plt.savefig(os.path.join('images', f'abstraction_floor_{floor}.png'))
        plt.show()

    def create_graph(self):
        G = nx.Graph()
        coordinates = {
            'security_laadloskade': [52.3078660, 4.7647489, 0],
            'security_transportstaat': [52.3098812, 4.7642785, 0],
            'security_expeditiestraat': [52.3108456, 4.7606374, 0],

            'l089 0': [52.3073304, 4.7649418, 0],
            'l089 2': [52.3073304, 4.7649418, 2],

            'l086_and_l087 0': [52.308192, 4.764696, 0],
            'l086_and_l087 1': [52.308192, 4.764696, 1],
            'l086_and_l087 2': [52.308192, 4.764696, 2],

            'l013_and_l014 0': [52.309931, 4.764581, 0],
            'l013_and_l014 1': [52.309931, 4.764581, 1],
            'l013_and_l014 2': [52.309931, 4.764581, 2],

            'l056_and_l057 0': [52.310941, 4.760956, 0],
            'l056_and_l057 1': [52.310941, 4.760956, 1],
            'l056_and_l057 2': [52.310941, 4.760956, 2],

            '1G end': [52.30926809299719, 4.752831904456476, 1],
            '1HG': [52.31109883198455, 4.757280624196045, 1],
            '1H end': [52.313787426683206, 4.75697323511764, 1],
            '1F': [52.311138187939335, 4.761979854324515, 1],
            '1F end': [52.313046261353215, 4.761908499636125, 1],
            '1E': [52.310049326949134, 4.764747893989231, 1],
            '1E end': [52.3140497831187, 4.769075362076724, 1],
            '1D': [52.30875708969503, 4.765101945574253, 1],
            '1D middle': [52.30859117529977, 4.769011818655832, 1],
            '1D up middle': [52.3095068265551, 4.770148570452533, 1],
            '1D up end': [52.309413055911456, 4.773838502871982, 1],
            '1D down middle': [52.30752656900555, 4.769950089980093, 1],
            '1D down end': [52.30744382651151, 4.773739262635762, 1],
            '1C': [52.307104020553, 4.7645655037787655, 1],
            '1C end': [52.305194393877635, 4.766468254991557, 1],
            '1B': [52.30555895233896, 4.761083315437157, 1],
            '1B end': [52.303147563023394, 4.763496367409801, 1],

            '2HG': [52.31118825125989, 4.759553721950411, 2],

            '2F': [52.311138187939335, 4.761979854324515, 2],
            '2F end': [52.313046261353215, 4.761908499636125, 2],
            '2E': [52.310049326949134, 4.764747893989231, 2],
            '2E end': [52.3140497831187, 4.769075362076724, 2],
            '2D': [52.30875708969503, 4.765101945574253, 2],
            '2D middle': [52.30859117529977, 4.769011818655832, 2],
            '2D up middle': [52.3095068265551, 4.770148570452533, 2],
            '2D up end': [52.309413055911456, 4.773838502871982, 2],
            '2D down middle': [52.30752656900555, 4.769950089980093, 2],

            '2C': [52.307104020553, 4.7645655037787655, 2],

        }

        for key, value in coordinates.items():
            G.add_node(key, x=value[0], y=value[1], z=value[2])

        edges = np.array([
            ['security_laadloskade', 'l086_and_l087 0'],
            ['security_laadloskade', 'l089 0'],
            ['security_transportstaat', 'l013_and_l014 0'],
            ['security_expeditiestraat', 'l056_and_l057 0'],

            ['l089 0', 'l089 2'],
            ['l089 2', '2C'],

            ['l086_and_l087 0', 'l086_and_l087 1'],
            ['l086_and_l087 1', 'l086_and_l087 2'],
            ['l086_and_l087 1', '1D'],
            ['l086_and_l087 1', '1C'],
            ['l086_and_l087 2', '2D'],

            ['l013_and_l014 0', 'l013_and_l014 1'],
            ['l013_and_l014 1', 'l013_and_l014 2'],
            ['l013_and_l014 1', '1D'],
            ['l013_and_l014 1', '1E'],
            ['l013_and_l014 2', '2D'],
            ['l013_and_l014 2', '2E'],

            ['l056_and_l057 0', 'l056_and_l057 1'],
            ['l056_and_l057 1', 'l056_and_l057 2'],
            ['l056_and_l057 1', '1F'],
            ['l056_and_l057 1', '1HG'],
            ['l056_and_l057 2', '2F'],
            ['l056_and_l057 2', '2HG'],

            ['1G end', '1HG'],
            ['1HG', '1H end'],
            ['1HG', '1F'],
            ['1F', '1F end'],
            ['1F', '1E'],
            ['1E', '1E end'],
            ['1E', '1D'],
            ['1D', '1D middle'],
            ['1D middle', '1D up middle'],
            ['1D up middle', '1D up end'],
            ['1D middle', '1D down middle'],
            ['1D down middle', '1D down end'],
            ['1D', '1C'],
            ['1C', '1C end'],
            ['1C', '1B'],
            ['1B', '1B end'],
            ['2HG', '2F'],
            ['2F', '2F end'],
            ['2F', '2E'],
            ['2E', '2E end'],
            ['2E', '2D'],
            ['2D', '2D middle'],
            ['2D middle', '2D up middle'],
            ['2D up middle', '2D up end'],
            ['2D middle', '2D down middle'],
        ])

        for u, v in edges:
            distance = self.calculate_haversine([coordinates[u][:2]], [coordinates[v][:2]])[0][0]
            G.add_edge(u, v, weight=distance)

        return G

    def find_closest_edge(self, point):
        min_distance = np.inf
        min_edge = (0, 0)
        for edge in self.graph.edges():
            n1 = self.graph.nodes.get(edge[0])
            n2 = self.graph.nodes.get(edge[1])
            if n1['z'] != point.z and n2['z'] != point.z:
                continue
            l1 = LineString([[n1['x'], n1['y']], [n2['x'], n2['y']]])
            distance = l1.distance(point)
            if distance < min_distance:
                min_distance = distance
                min_edge = edge
        return min_edge

    def calculate_shortest_path(self, start_point, end_point):
        # map points to nodes
        self.calculate_distance(start_point, end_point)
        e1 = self.find_closest_edge(Point(start_point))
        e2 = self.find_closest_edge(Point(end_point))

        min_d = np.inf
        path = []
        for n1, n2 in product(e1, e2):
            d = nx.shortest_path_length(G=self.graph, source=n1, target=n2, weight='weight')
            if d < min_d:
                min_d = d
                path = nx.shortest_path(G=self.graph, source=n1, target=n2, weight='weight')
        return min_d, path

    def calculate_shortest_path_distance(self, start_point, end_point, way_points):
        if len(way_points) > 3:
            raise Warning('Trying to calculate shortest path for large instance!')

        all_points = np.vstack([start_point, way_points, end_point])
        # calculate distance matrix
        n = len(all_points)
        distances = np.zeros(shape=(n, n))
        for idx, (r1, r2) in enumerate(product(all_points, all_points)):
            d = self.calculate_distance(r1.tolist(), r2.tolist())
            distances[idx // n, idx % n] = d

        # what are the options?
        min_distance = np.inf
        for order in permutations(range(1, n-1)):
            all_order = [0, *order, n-1]
            distance = sum([distances[e1, e2] for e1, e2 in zip(all_order[:-1], all_order[1:])])
            min_distance = min(distance, min_distance)

        return min_distance

    def calculate_distance(self, start_point, end_point):
        e1 = self.find_closest_edge(Point(start_point))
        e2 = self.find_closest_edge(Point(end_point))

        # 4 shortest paths:
        min_distance = np.inf
        if e1 == e2:
            # calculate distance between start and end
            min_distance = self.calculate_haversine([start_point], [end_point])[0][0]
        else:
            nodes = self.graph.nodes(data=True)
            relevant_nodes = [[nodes[idx]['x'], nodes[idx]['y']] for idx in [*e1, *e2]]
            distances_e1 = self.calculate_haversine([start_point], relevant_nodes[:2])[0]
            distances_e2 = self.calculate_haversine([end_point], relevant_nodes[2:])[0]
            for idx, (i, j) in enumerate(product(e1, e2)):
                distance = (distances_e1[idx // 2] + distances_e2[idx % 2])
                distance += nx.shortest_path_length(self.graph, source=i, target=j, weight='weight')
                min_distance = min(min_distance, distance)
        return min_distance

    def calculate_haversine(self, X, Y):
        X = np.array(X)
        X = np.radians(X[:, :2])
        # X = [[radians(x1), radians(x2)] for x1, x2 in X]
        if Y is not None:
            Y = np.array(Y)
            Y = np.radians(Y[:, :2])
            # Y = [[radians(y1), radians(y2)] for y1, y2 in Y]
        return haversine_distances(X=X, Y=Y) * 6371000


if __name__ == '__main__':
    g = SchipholGraph()
    for floor in range(3):
        g.visualize_floor(floor=floor)
    start_point = [52.308192, 4.764696, 0]
    end_point = [52.308192, 4.764696, 0]
    waypoints = [
        [52.30926809299719, 4.752831904456476, 1],
        [52.30859117529977, 4.769011818655832, 1],
        [52.31118825125989, 4.759553721950411, 2]
    ]
    g.calculate_shortest_path_distance(start_point, end_point, waypoints)


