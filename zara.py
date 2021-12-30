import os.path
from concurrent.futures import ThreadPoolExecutor

import geopandas as gp
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely.geometry
from palettable.mycarta import Cube1_6
from tqdm import tqdm


def get_colors(n, cmap='viridis', start=0., stop=1., alpha=1., ):
    # todo: no need for a for loop here, cmaps accept a np.linspace
    colors = [cm.get_cmap(cmap)(x) for x in np.linspace(start, stop, n)]
    colors = [(r, g, b, alpha) for r, g, b, _ in colors]
    return colors


def get_edge_colors_by_attr(G, G_edges, attr, num_bins=None, cmap='viridis', start=0, stop=1, na_color='none'):
    if num_bins is None:
        num_bins = len(G.edges())
    bin_labels = range(num_bins)
    # invert values because greater distances should have smaller cmap values
    # rank edge lengths because there will be duplicates (we don't want to drop them)
    attr_values = pd.Series(-G_edges[attr].rank(method='first').values)
    # Quantile based discretization of distance values
    cats = pd.qcut(x=attr_values, q=num_bins, labels=bin_labels)
    # get cmap bin colors
    colors = get_colors(num_bins, cmap, start, stop)
    # map distances to colors
    edge_colors = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in cats]
    return edge_colors


# turn addresses to coordinates and automatically save them to a file
def geocode_poi(addresses):
    if not os.path.isfile('zara.geojson'):
        zara = gp.tools.geocode(addresses, provider='google', api_key=os.environ['GOOGLE_API_KEY'])
        zara.to_file('zara.geojson')
    else:
        zara = gp.read_file('zara.geojson')
    zara = zara.to_crs(crs)
    return zara


# load the graph from a bbox and convert to the appropriate crs. Automatically saves it to a file if it doesn't exist.
def load_graph(ymin, ymax, xmin, xmax, crs):
    if not os.path.isfile('zagreb.osm'):
        G = ox.graph_from_bbox(ymin, ymax, xmin, xmax, network_type='all', simplify=True)
        G = ox.project_graph(G, to_crs=crs)
        ox.save_graphml(G, filepath='zagreb.osm')
    else:
        G = ox.load_graphml('zagreb.osm')
    return G


# get nodes from points of interests (in a GeoDataFrame)
def get_nodes(G, pois):
    nodes = []
    for geometry in pois.geometry:
        node = ox.nearest_nodes(G, geometry.x, geometry.y)
        nodes.append(node)
    return nodes


# calculate and update the shortest route distance to the closest point of interest for a single node.
def calculate_single_node(G, G_nodes):
    def wrapper(node):
        if node in nodes:
            return
        min_route_length = None
        for target in nodes:
            try:
                # route = nx.shortest_path(G, node, target)
                route_length = nx.shortest_path_length(G, node, target)
                if min_route_length is None or route_length < min_route_length:
                    min_route_length = route_length
                    G_nodes.at[node, ATTR] = route_length
            except nx.NetworkXNoPath:
                continue

    return wrapper


# calculate and update the shortest route distance to the closest point of interest for all nodes.
def calculate_distances(G):
    G_nodes, _ = ox.graph_to_gdfs(G, nodes=True, edges=True)
    G_nodes[ATTR] = 0
    if not os.path.isfile('nodes.shp'):
        with ThreadPoolExecutor(7) as pool:
            list(tqdm(pool.map(calculate_single_node(G, G_nodes), G.nodes), total=len(G.nodes)))
        G_nodes.to_file('nodes.shp')
    else:
        G_nodes = gp.read_file('nodes.shp')
    return G_nodes


# update edge distances based on nodes connected to them.
def update_edge_distances(G, G_nodes):
    if not os.path.isfile('edges.shp'):
        for node in G.nodes:
            val = int(G_nodes.loc[G_nodes['osmid'] == node, ATTR])
            attrs = dict(map(lambda t: (t, {ATTR: val}), G.out_edges(node, keys=True)))
            nx.set_edge_attributes(G, attrs)
        G_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        # G_edges.to_file('edges.shp') fixme: can't store?
    else:
        G_edges = gp.read_file('edges.shp')
    return G_edges


# plot the graph
def plot_map(G, G_edges, pois, filepath='rezultat.svg', dpi=300, home=None):
    # nc = get_node_colors_by_attr(G, attr=ATTR, num_bins=20, )
    ec = get_edge_colors_by_attr(G, G_edges, attr=ATTR, num_bins=40, cmap=Cube1_6.mpl_colormap)
    fig, ax = ox.plot_graph(G, edge_color=ec, node_size=0, figsize=(11.75, 8.25),
                            bgcolor='#FFFFFF', show=False, close=False, edge_linewidth=0.5)
    pois.plot(ax=ax, marker="v", color='red', markersize=35)
    if home is not None:
        home = gp.GeoDataFrame({'address': ['Ulica Drage Stipca, 8'],
                                'geometry': [shapely.geometry.Point(15.903368169277508, 45.79833351915581)]})
        home.crs = crs
        home.plot(ax=ax, marker='*', color='orange', markersize=45)

    extent = ax.bbox.transformed(fig.dpi_scale_trans.inverted())
    # temporarily turn figure frame on to save with facecolor
    fig.set_frameon(True)
    fig.savefig(filepath, bbox_inches=extent, transparent=True, dpi=dpi)
    fig.set_frameon(False)  # and turn it back off again


def create_home(address, lat, lon, crs):
    return gp.GeoDataFrame({'address': [address], 'geometry': [shapely.geometry.Point(lon, lat)]}, crs=crs)


if __name__ == '__main__':
    addresses = ['Ilica, 30', 'Avenija Dubrovnik, 16',
                 'Slavonska avenija, 11', 'Vice Vukova, 6',
                 'Jankomir, 33']
    crs = 'EPSG:4326'
    ATTR = 'shortest_route_length_to_target'[:10]  # when stored in shp it's truncated to 10 chars

    # geocode points of interest
    zara = geocode_poi(addresses)
    # get map bounds and offset them slightly to avoid having points at the edges
    OFFSET = 0.01
    xmin, ymin, xmax, ymax = zara.total_bounds
    xmin -= OFFSET
    ymin -= OFFSET
    xmax += OFFSET
    ymax += OFFSET

    # if graph from bbox hasn't been downloaded before, download it.
    G = load_graph(ymin, ymax, xmin, xmax, crs)
    nodes = get_nodes(G, zara)

    G_nodes = calculate_distances(G)
    G_edges = update_edge_distances(G, G_nodes)

    # home = create_home('your address', 48.85842831388544, 2.2945108025251, crs)
    home = None
    plot_map(G, G_edges, zara, home=home)
