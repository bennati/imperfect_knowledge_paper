import pandas as pd
import geopandas as gpd
import json
import numpy as np
from collections import defaultdict
from random import sample
from tqdm import tqdm

import folium
from folium.features import GeoJsonTooltip
from geojson import Point, Polygon, Feature, FeatureCollection

from turfpy.measurement import boolean_point_in_polygon
import matplotlib.pyplot as plt


def subtract_polygons(districts, district_name, polygons):
    ret = districts[districts['BoroCD'] == district_name]
    for p in polygons:
        ret = gpd.overlay(ret, p, how='difference')
    return ret


def extract_geom(poly):
    return json.loads(poly.to_json())['features']


def scores_time_window(scores, df, spatial_param, temporal_param, time_window):
    # compute the bboxes of cells in the grid, according to the given parameters
    _, bboxes = define_equivalence_areas(df, spatial_param, temporal_param, return_bbox=True)
    # select only those that overlap with time window
    bboxes = bboxes[bboxes['time_box'].apply(lambda t: len(set(range(*t)).intersection(set(range(*time_window)))) > 0)]
    # select the scores that match with the parameters
    ret = scores.query(f"spatial_threshold == {spatial_param} & temporal_threshold == {temporal_param}"
                       ).drop_duplicates('cluster_id').set_index('cluster_id')
    if 'trajectory_id' in ret.columns:
        ret.drop('trajectory_id', axis=1, inplace=True)
    ret = pd.merge(bboxes, ret, left_index=True, right_index=True, how='inner')
    ret = ret.reset_index()
    # aggregate scores over the time, to have a 2D representation that can be plotted onto a map
    ret['index'] = ret['index'].apply(lambda x: "_".join(x.split("_")[:-1]))  # remove time component
    ret = ret.groupby(['lat_box', 'lng_box', 'index']).agg(
        {'k_anonymity': np.mean, 'l_diversity': np.mean, 't_closeness': np.mean}).reset_index().set_index('index')
    return ret


def scores_time_window_polygon(scores, df, polygons, temporal_param, time_window):
    # define eq areas
    if ('equivalence_area_start' not in df.columns) or ('equivalence_area_end' not in df.columns):
        print("Defining equivalence areas from polygons")
        df = define_equivalence_areas_polygons(df, polygons, temporal_param)
    df = df[~((df.start_polygon.isin(["NA"])) | (df.end_polygon.isin(["NA"])))]
    # select only data that overlaps with time window
    tqdm.pandas(desc="Select data inside time window")
    df = df[df.progress_apply(lambda r: len(set(range(r.start_time_unix, r.end_time_unix)
                                                ).intersection(set(range(*time_window)))) > 0, axis=1)]
    cluster_ids = set(df.equivalence_area_start).union(set(df.equivalence_area_end))
    print("Select scores")
    # select the scores that match with the parameters
    ret = scores.query(f"temporal_threshold == {temporal_param}"
                       ).drop_duplicates('cluster_id').set_index('cluster_id')
    if 'trajectory_id' in ret.columns:
        ret.drop('trajectory_id', axis=1, inplace=True)
    # keep only the clusters that match with the given time window
    ret = ret.loc[ret.index.isin(cluster_ids)]
    ret = ret.reset_index()
    # aggregate scores over the time, to have a 2D representation that can be plotted onto a map
    print("Compute stats")
    ret['cluster_id'] = ret['cluster_id'].apply(lambda x: "_".join(x.split("_")[:-1]))  # remove time component
    ret = ret.groupby(['cluster_id']).agg(
        {'k_anonymity': np.mean, 'l_diversity': np.mean, 't_closeness': np.mean, 'l_div_norm': np.mean}).reset_index().set_index('cluster_id')
    return ret


def plot_staircase(df, ax=None, fontsize=14):
    if ax is None:
        fig, ax = plt.subplots()
        plt.tight_layout()
    kanons = df['k_anonymity'].sort_values().reset_index(drop=True)
    kanons.plot(ax=ax, linewidth=3, color='blue')
    ldivs = df['l_diversity'].sort_values().reset_index(drop=True)
    ldivs.plot(ax=ax, linewidth=3, color='orange')
    plt.legend(loc='upper left', prop={"size": fontsize})
    ax.set_xticks(np.linspace(0, len(kanons), 5))
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    return ax


def plot_tclose(df, ax=None, fontsize=14):
    if ax is None:
        fig, ax = plt.subplots()
        plt.tight_layout()
    tcls = df['t_closeness'].sort_values().reset_index(drop=True)
    tcls.plot(ax=ax, linewidth=3, color='blue')
    ax.set_xticks(np.linspace(0, len(tcls), 5))
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='upper left', prop={"size": fontsize})
    return ax


def plot_choropleth(df, field='k_anonymity', cmap='YlGn'):
    geoJsonData = FeatureCollection([
        Feature(geometry=Polygon([[(min(cell.lng_box), max(cell.lat_box)),  # top left
                                   # bottom left
                                   (min(cell.lng_box), min(cell.lat_box)),
                                   # bottom right
                                   (max(cell.lng_box), min(cell.lat_box)),
                                   # top_right
                                   (max(cell.lng_box), max(cell.lat_box)),
                                   (min(cell.lng_box), max(
                                       cell.lat_box))  # top_left
                                   ]]
                                 ), name=cell.name, properties={'name': cell.name, 'value': cell[field]})
        for i, cell in df.iterrows()])
    tooltip = GeoJsonTooltip(
        fields=['value'],
        aliases=[field],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 1px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    m = folium.Map(location=[(min(df.lat_box.apply(min)) + max(df.lat_box.apply(max))) / 2,
                             (min(df.lng_box.apply(min)) + max(df.lng_box.apply(max))) / 2], zoom_start=9)
    g = folium.Choropleth(geo_data=geoJsonData, data=df.reset_index(),
                          columns=['index', field],
                          key_on='feature.name',
                          fill_color=cmap, fill_opacity=0.7, line_opacity=0.2, legend_name=field).add_to(m)
    folium.GeoJson(
        geoJsonData,
        style_function=lambda feature: {
            'fillColor': '#ffff00',
            'color': 'black',
            'weight': 0.2,
            'dashArray': '5, 5'
        },
        tooltip=tooltip).add_to(g)
    return m


def plot_choropleth_polygon(df, polygons, field='k_anonymity', cmap='YlGn'):
    df = df[~df.index.isin(["NA"])]
    geoJsonData = FeatureCollection(extract_geom(pd.merge(polygons, df.reset_index(),
                                                          left_on='BoroCD', right_on='cluster_id', how='inner')))

    tooltip = GeoJsonTooltip(
        fields=[field, 'BoroCD'],
        aliases=[field, 'cluster_id'],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 1px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    m = folium.Map(location=[40.75583913326111, -73.98246791099724], zoom_start=12)
    g = folium.Choropleth(geo_data=geoJsonData,
                          data=df.reset_index(),
                          columns=['cluster_id', field],
                          key_on='feature.properties.BoroCD',
                          fill_color=cmap, fill_opacity=0.7, line_opacity=0.2, legend_name=field).add_to(m)
    folium.GeoJson(
        geoJsonData,
        style_function=lambda feature: {
            'fillColor': '#ffff00',
            'color': 'black',
            'weight': 0.2,
            'dashArray': '5, 5'
        },
        tooltip=tooltip).add_to(g)
    return m


def body(e, df):
    clustered_data = define_equivalence_areas(df, e['spatial_threshold'], e['temporal_threshold'])
    ret = body_compute_metrics(clustered_data)
    for k, v in e.items():
        ret[k] = v
    return ret


def body_polygons(e, df, polygons):
    print("Defining equivalence areas from polygons")
    clustered_data = define_equivalence_areas_polygons(df, polygons, e['temporal_threshold'])
    ret = body_compute_metrics(clustered_data)
    for k, v in e.items():
        ret[k] = v
    return ret


def body_compute_metrics(clustered_data):
    print("Building clusters")
    cluster_to_trj = clustered_data.groupby('equivalence_area_start').agg({'trajectory_id': set})
    cluster_end_to_trj = clustered_data.groupby('equivalence_area_end').agg({'trajectory_id': set})
    trj_to_cluster = clustered_data.set_index('trajectory_id')['equivalence_area_end'].to_dict()
    tqdm.pandas(desc="Computing kanon and ldiv")
    ret = clustered_data.progress_apply(lambda r: compute_kanon_and_ldiv_from_dict(
        build_dict(r, cluster_to_trj=cluster_to_trj, cluster_end_to_trj=cluster_end_to_trj,
                   trj_to_cluster=trj_to_cluster)), axis=1)
    print("Computing t-closeness")
    tcl = compute_t_closeness(clustered_data)
    ret = pd.merge(pd.DataFrame(list(ret)), pd.Series(tcl).rename(
        't_closeness'), left_on='cluster_id', right_index=True)
    return ret


def add_noise_to_data(e, df):
    anon_data = df.copy()
    anon_data['start_latitude'] = anon_data['start_latitude'] + np.random.normal(0, e['spatial_noise'], len(anon_data))
    anon_data['start_longitude'] = anon_data['start_longitude'] + \
        np.random.normal(0, e['spatial_noise'], len(anon_data))
    anon_data['end_latitude'] = anon_data['end_latitude'] + np.random.normal(0, e['spatial_noise'], len(anon_data))
    anon_data['end_longitude'] = anon_data['end_longitude'] + np.random.normal(0, e['spatial_noise'], len(anon_data))
    anon_data['start_time'] = anon_data['start_time'] + \
        pd.to_timedelta(np.random.randint(0, e['temporal_noise'], len(anon_data)), unit='sec')
    anon_data['end_time'] = anon_data['end_time'] + \
        pd.to_timedelta(np.random.randint(0, e['temporal_noise'], len(anon_data)), unit='sec')
    return anon_data


def anon_body(e, df):
    return body(e, add_noise_to_data(e, df))


def define_equivalence_areas(df, spatial_delta, temporal_delta, return_bbox=False):
    ret = df.copy()
    lat_min = min(ret.start_latitude.min(), ret.end_latitude.min())
    lat_max = max(ret.start_latitude.max(), ret.end_latitude.max())
    lng_min = min(ret.start_longitude.min(), ret.end_longitude.min())
    lng_max = max(ret.start_longitude.max(), ret.end_longitude.max())
    time_min = min(ret.start_time_unix.min(), ret.end_time_unix.min())
    time_max = max(ret.start_time_unix.max(), ret.end_time_unix.max())
    ##
    lat_grid = list(np.arange(lat_min, lat_max, step=spatial_delta)) + [lat_max + 0.001]
    lng_grid = list(np.arange(lng_min, lng_max, step=spatial_delta)) + [lng_max + 0.001]
    time_grid = list(np.arange(time_min, time_max, step=temporal_delta)) + [time_max + 1]
    ##
    ret['start_latitude_grid'] = np.digitize(ret['start_latitude'], lat_grid)
    ret['end_latitude_grid'] = np.digitize(ret['end_latitude'], lat_grid)
    ret['start_longitude_grid'] = np.digitize(ret['start_longitude'], lng_grid)
    ret['end_longitude_grid'] = np.digitize(ret['end_longitude'], lng_grid)
    #
    ret['start_time_grid'] = np.digitize(ret['start_time_unix'], time_grid)
    ret['end_time_grid'] = np.digitize(ret['end_time_unix'], time_grid)
    # build area names
    ret['equivalence_area_start'] = ret.apply(lambda r: (str(r['start_latitude_grid']) + "_" +
                                                         str(r['start_longitude_grid']) + "_" +
                                                         str(r['start_time_grid'])), axis=1)
    ret['equivalence_area_end'] = ret.apply(lambda r: (str(r['end_latitude_grid']) + "_" +
                                                       str(r['end_longitude_grid']) + "_" +
                                                       str(r['end_time_grid'])), axis=1)
    if return_bbox:
        bboxes = defaultdict(list)
        for k in set(list(ret['equivalence_area_start']) + list(ret['equivalence_area_end'])):
            indices = [int(x) for x in k.split("_")]
            bboxes[k] = [(l[i - 1], l[i]) for i, l in zip(indices, [lat_grid, lng_grid, time_grid])]
        return ret, pd.DataFrame(bboxes).T.rename(columns={0: 'lat_box', 1: 'lng_box', 2: 'time_box'})
    return ret


def build_dict(r, cluster_to_trj, cluster_end_to_trj, trj_to_cluster):
    sess_id = r.trajectory_id
    start_cluster_id = r.equivalence_area_start
    output = defaultdict(dict)
    # all trajectories that start in the same cluster as T, including T
    trjs_in_cluster = cluster_to_trj.loc[start_cluster_id]['trajectory_id']
    for trj_id in trjs_in_cluster:
        end_cluster_id = trj_to_cluster[trj_id]  # find the cluster IDs where each of these trj ends
        trjs_in_end_cluster = cluster_end_to_trj.loc[end_cluster_id]['trajectory_id'].intersection(
            trjs_in_cluster)
        output[trj_id] = trjs_in_end_cluster
    return {'trajectory_id': sess_id, 'cluster_id': start_cluster_id, 'cluster': output}


def compute_kanon_and_ldiv_from_dict(clusters):
    return {'trajectory_id': clusters['trajectory_id'],
            'cluster_id': clusters['cluster_id'],
            'k_anonymity': compute_kanon(clusters['cluster']),
            'l_diversity': len(compute_ldiv(clusters['cluster'])),
            'l_div_norm': normalize_ldiv(clusters['cluster'])}


def compute_kanon(x):
    """
    x is a dictionary where the keys are trj ids of all trjs that start in the same cluster.
    We define k-anonymity as the number of trjs that start in the same cluster.
    """
    return len(x)


def compute_ldiv(x):
    """
    x is a dictionary where the keys are trj ids of all trjs that start in the same cluster.
    and the values contain the subset of these trjs which end in the same cluster as the key.

    L-diversity is the number of well-represented values associated to a trj, i.e. distinct end clusters.
    """
    neighbors = []
    trjs = list(x.keys())
    while len(trjs) > 0:
        j = trjs.pop()
        box_input_trj = x[j]
        neighbors.append(box_input_trj)
        trjs = set(trjs) - set(box_input_trj)  # mark the resulting trjs as processed
    return neighbors


def normalize_ldiv(clusters, thresh=0.1, method='relative'):
    """
    Compute a variation of l_diversity which has the long tail truncated.
    This mitigates situations where one destination is hugely popular and many are not, VS where a few destinations are comparably popular.

    Args:
    clusters: a dictionary representing the clusters
    thresh: the threshold where to cut the tail, no cutting is applied if None
    method: How to perform the cut. Valid values are:
    - 'individual', where one cluster is dropped if its size (fraction over all clusters) is less than 'thresh'
    - 'cumulative', where clusters are ordered by size and those are kept that cumulatively account for 'thresh'% of the trajectories.
    - 'relative', where clusters are dropped if their size is smaller than a fraction 'thresh' of the largest cluster
    """
    if len(clusters) == 0:
        return np.nan
    if thresh is not None:
        # compute the size of each destination cluster, i.e. how many traces lead there
        destinations = [(c, len(c)) for c in compute_ldiv(clusters)]
        if method == 'individual':
            # keep only the destination which individually account for a fraction greater than 'thresh'
            norm = sum([c[1] for c in destinations])
            top_d = [d[0] for d in destinations if d[1] / norm >= thresh]
        elif method == 'cumulative':
            # order destinations by size
            destinations = sorted(destinations, key=lambda t: t[1], reverse=True)
            # cumulative sizes
            cum_sizes = np.cumsum([c[1] for c in destinations])
            norm = max(cum_sizes)
            top_d = cum_sizes / norm <= thresh
            # keep one extra element, to go over the threshold
            last_element = np.where(top_d == False)[0][0]
            top_d[last_element] = True
            # keep only the destination which cumulatively account for a fraction less than 'thresh'
            top_d = [d[0] for d in np.array(destinations)[top_d]]
        elif method == 'relative':
            highest = max([c[1] for c in destinations])
            top_d = [d[0] for d in destinations if d[1] >= highest * thresh]
        else:
            raise ValueError(f"Provided method {method} is not supported")
        # filter the data to keep only these clusters
        top_d = [item for sublist in top_d for item in sublist]
        filtered_clusters = {k: clusters[k] for k in top_d}
        if len(filtered_clusters) == 0:
            return np.nan
    else:
        filtered_clusters = clusters
    return len(compute_ldiv(filtered_clusters))


from scipy.stats import entropy
import math


def compute_t_closeness(clusters, fct=entropy, how='outer'):
    """
    clusters is a dataframe with columns
    - 'trajectory_id' contains the trj ID, each row is supposed to identify one trj
    - 'equivalence_area_start' contains the ID of the equivalence class of the origin
    - 'equivalence_area_end' contains the ID of the destination cluster, i.e. the equivalence classes on the destinations

    Return a dictionary, where each trj is associated with the corresponding t-closeness
    """
    t_closeness_clusters = {}
    Q = clusters.equivalence_area_end.value_counts()  # data-wide distribution
    Q = pd.DataFrame(Q).rename(columns={'equivalence_area_end': 'Q'})
    for clid, g in clusters.groupby('equivalence_area_start'):  # every equivalence class
        P = g.equivalence_area_end.value_counts()  # distribution in equivalence class
        distrib = Q.merge(pd.DataFrame(P).rename(
            columns={'equivalence_area_end': 'P'}), left_index=True, right_index=True, how=how)
        distrib.fillna(0, inplace=True)
        t_closeness_clusters.update({clid: fct(pk=distrib.P, qk=distrib.Q)  # compute the Kullback-Leibler divergence
                                     })
    return {k: (np.inf if v == 0 else
                math.exp(v)) for k, v in t_closeness_clusters.items()}


def f(r, cluster_to_trj, cluster_end_to_trj, trj_to_cluster):
    return compute_kanon_and_ldiv_from_dict(build_dict(r[1], cluster_to_trj=cluster_to_trj,
                                                       cluster_end_to_trj=cluster_end_to_trj, trj_to_cluster=trj_to_cluster))


def get_containing_polygon_name(lat, lon, polygons):
    eq_area = [p for p in polygons if boolean_point_in_polygon(Feature(geometry=Point([lon, lat])), p)]
    if len(eq_area) > 1:  # take a random one
        eq_area = sample(eq_area, 1)
    return eq_area[0]['properties']['BoroCD'] if len(eq_area) > 0 else "NA"


def define_equivalence_areas_polygons(df, polygons, temporal_delta):
    ret = df.copy()
    time_min = min(ret.start_time_unix.min(), ret.end_time_unix.min())
    time_max = max(ret.start_time_unix.max(), ret.end_time_unix.max())
    ##
    time_grid = list(np.arange(time_min, time_max, step=temporal_delta)) + [time_max + 1]
    ##
    ret['start_time_grid'] = np.digitize(ret['start_time_unix'], time_grid)
    ret['end_time_grid'] = np.digitize(ret['end_time_unix'], time_grid)
    # build area names
    tqdm.pandas(desc="Match starting points with polygons")
    ret['start_polygon'] = ret.progress_apply(lambda r: get_containing_polygon_name(
        r.start_latitude, r.start_longitude, polygons), axis=1)
    tqdm.pandas(desc="Match ending points with polygons")
    ret['end_polygon'] = ret.progress_apply(lambda r: get_containing_polygon_name(
        r.end_latitude, r.end_longitude, polygons), axis=1)
    print("Build cluster names")
    ret['equivalence_area_start'] = ret.apply(lambda r: (str(r['start_polygon']) + "_" +
                                                         str(r['start_time_grid'])), axis=1)
    ret['equivalence_area_end'] = ret.apply(lambda r: (str(r['end_polygon']) + "_" +
                                                       str(r['end_time_grid'])), axis=1)
    return ret
