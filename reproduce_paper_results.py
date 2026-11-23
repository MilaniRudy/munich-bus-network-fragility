# -*- coding: utf-8 -*-
"""Compute the analysis for the Munich bus network case study."""

from pathlib import Path
from collections import Counter, defaultdict
from contextlib import contextmanager
import datetime as dt
import heapq
import io
import math
import os
import time
import warnings
import zipfile

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import requests

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

#%% Parameters selection

GTFS_SOURCE = "https://www.mvg.de/static/gtfs/google_transit.zip"
ANALYSIS_DATE = None
STRICT_ANALYSIS_DATE = True
TIME_WINDOW = ("08:00:00", "20:00:00")
SERVICE_KEY_MODE = "route_direction"
WAIT_BETA_MIN = 0.0
TOP_K = 10

RUN_LITERATURE_MEASURES = True
RUN_MAPS = True
RUN_WALKING_SENSITIVITY = False
RECORD_ALGORITHM_TIMINGS = True
SHOW_PROGRESS_BARS = True

# In the baseline case, walking connectors and waiting times are not included! 
# However, in the following there are the parameters for the sensitivity analysis with walking connectors
WALK_RADIUS_M = 300.0
WALK_SPEED_KMH = 4.5
WALK_MAX_NEIGHBORS_PER_STOP = None
WALK_KEY_PREFIX = "__walk__"

OUTPUT_ROOT = Path("results")

TIMING_SUMMARY_ROWS = []
TIMING_NESTING = []
ALGORITHM_TIMING_ROWS = []

plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

# To grab information of only buses we need both these types! (Otherwise we miss some lines)
def default_bus_route_types():
    return tuple(['3'] + [str(x) for x in range(700, 717)])

def is_bus_route_type(value, bus_route_types=None):
    bus_route_types = default_bus_route_types() if bus_route_types is None else tuple((str(x) for x in bus_route_types))
    if pd.isna(value):
        return False
    return str(value) in bus_route_types

def parse_gtfs_time(tstr):
    if pd.isna(tstr):
        return None
    tstr = str(tstr)
    parts = tstr.split(':')
    if len(parts) != 3:
        return None
    try:
        h, m, s = [int(x) for x in parts]
        return h * 3600 + m * 60 + s
    except Exception:
        return None

def parse_yyyymmdd(value):
    if pd.isna(value):
        return None
    s = str(value)
    if len(s) == 8 and s.isdigit():
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return None

def load_gtfs(source, timeout=180):
    source_path = Path(source)
    if source_path.exists():
        zf = zipfile.ZipFile(source_path)
    else:
        raw = requests.get(str(source), timeout=timeout)
        raw.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(raw.content))

    def read_csv(name):
        try:
            return pd.read_csv(zf.open(name), dtype=str)
        except KeyError:
            return pd.DataFrame()
    return {'routes': read_csv('routes.txt'), 'trips': read_csv('trips.txt'), 'stop_times': read_csv('stop_times.txt'), 'stops': read_csv('stops.txt'), 'calendar': read_csv('calendar.txt'), 'calendar_dates': read_csv('calendar_dates.txt'), 'agency': read_csv('agency.txt')}

def choose_default_analysis_date(gtfs):
    cal = gtfs.get('calendar', pd.DataFrame())
    if cal is None or cal.empty:
        return None
    c = cal.copy()
    c['start_date_d'] = c['start_date'].apply(parse_yyyymmdd)
    c['end_date_d'] = c['end_date'].apply(parse_yyyymmdd)
    starts = [x for x in c['start_date_d'].tolist() if x is not None]
    ends = [x for x in c['end_date_d'].tolist() if x is not None]
    if not starts or not ends:
        return None
    start = min(starts)
    end = max(ends)
    preferred = {0, 1, 2, 3, 4} # Weekdays
    cur = start
    while cur <= end:
        if cur.weekday() in preferred:
            return cur
        cur += dt.timedelta(days=1) # If it is not a weekday, we move it by 1
    return start

def active_service_ids_for_date(gtfs, date_obj):
    cal = gtfs.get('calendar', pd.DataFrame())
    cald = gtfs.get('calendar_dates', pd.DataFrame())
    active = set()
    if cal is not None and (not cal.empty):
        c = cal.copy()
        c['start_date_d'] = c['start_date'].apply(parse_yyyymmdd)
        c['end_date_d'] = c['end_date'].apply(parse_yyyymmdd)
        dow_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        dow = dow_cols[date_obj.weekday()]
        mask = (c['start_date_d'] <= date_obj) & (c['end_date_d'] >= date_obj) & (c[dow] == '1')
        active |= set(c.loc[mask, 'service_id'].astype(str))
    if cald is not None and (not cald.empty):
        cd = cald.copy()
        cd['date_d'] = cd['date'].apply(parse_yyyymmdd)
        mask = cd['date_d'] == date_obj
        for _, row in cd.loc[mask].iterrows():
            sid = str(row['service_id'])
            typ = str(row.get('exception_type', ''))
            if typ == '1':
                active.add(sid)
            elif typ == '2':
                active.discard(sid)
    return active

def filter_trips_for_date(gtfs, date_obj):
    trips = gtfs['trips'].copy()
    if date_obj is None or 'service_id' not in trips.columns:
        return trips
    active = active_service_ids_for_date(gtfs, date_obj)
    if not active:
        message = f'No active service_ids were found for {date_obj}. Use the GTFS archive corresponding to the case-study date, or change ANALYSIS_DATE.'
        if STRICT_ANALYSIS_DATE:
            raise ValueError(message)
        warnings.warn(message + ' Falling back to all trips because STRICT_ANALYSIS_DATE=False.')
        return trips
    return trips[trips['service_id'].astype(str).isin(active)].copy()

def normalize_text_for_key(x):
    if pd.isna(x):
        return ''
    return str(x).strip()

# We need to define clearly the service using its route and direction
def make_service_key(route_id, direction_id=None, trip_headsign=None, mode='route_direction'):
    route_txt = normalize_text_for_key(route_id)
    dir_txt = normalize_text_for_key(direction_id) if direction_id is not None else 'na'
    head_txt = normalize_text_for_key(trip_headsign)
    if mode == 'route_direction_headsign':
        return f'{route_txt}|{dir_txt}|{head_txt}' # So we can have a clear identification of each line
    return f'{route_txt}|{dir_txt}'

def build_bus_multidigraph(gtfs, date_obj=None, bus_route_types=None, include_headsign=True, service_key_mode=None):
    # Get the necessary information for creating the multi(di)graph
    routes = gtfs['routes'].copy()
    trips = filter_trips_for_date(gtfs, date_obj)
    if service_key_mode is None:
        service_key_mode = 'route_direction'
    stop_times = gtfs['stop_times'].copy()
    stops = gtfs['stops'].copy()
    if 'route_type' in routes.columns:
        bus_route_types = default_bus_route_types() if bus_route_types is None else tuple((str(x) for x in bus_route_types))
        routes = routes[routes['route_type'].apply(lambda x: is_bus_route_type(x, bus_route_types))].copy()
    trips = trips[trips['route_id'].isin(routes['route_id'])].copy()
    trip_cols = ['trip_id', 'route_id']
    if 'direction_id' in trips.columns:
        trip_cols.append('direction_id')
    if 'trip_headsign' in trips.columns:
        trip_cols.append('trip_headsign')
    if 'service_id' in trips.columns:
        trip_cols.append('service_id')
    st = stop_times.merge(trips[trip_cols], on='trip_id', how='inner')
    if st.empty:
        raise RuntimeError('No GTFS stop_times/trips available after filtering.')
    st['stop_sequence_int'] = st['stop_sequence'].astype(int)
    st = st.sort_values(['trip_id', 'stop_sequence_int'])
    stop_meta = stops.set_index('stop_id').to_dict('index')
    route_meta = routes.set_index('route_id').to_dict('index')
    
    # Create the multigraph
    Gm = nx.MultiDiGraph()
    used_stops = set(st['stop_id'].unique())
    # Node-stop addition
    for sid in used_stops:
        m = stop_meta.get(sid, {})
        if not m:
            continue
        Gm.add_node(sid, name=m.get('stop_name', sid), lat=float(m.get('stop_lat', np.nan)), lon=float(m.get('stop_lon', np.nan)), parent_station=m.get('parent_station', ''), location_type=m.get('location_type', '0'))
    seen = set()
    
    # Edge-line connection addition
    for _, grp in st.groupby('trip_id', sort=False):
        route_id = grp['route_id'].iloc[0]
        direction_id = grp['direction_id'].iloc[0] if 'direction_id' in grp.columns else None
        headsign = grp['trip_headsign'].iloc[0] if 'trip_headsign' in grp.columns else None
        key = make_service_key(route_id, direction_id, headsign if include_headsign else None, mode=service_key_mode)
        seq = grp['stop_id'].tolist()
        for u, v in zip(seq, seq[1:]):
            if u not in Gm or v not in Gm:
                continue
            dedup = (u, v, key)
            if dedup in seen:
                continue
            seen.add(dedup)
            rmeta = route_meta.get(route_id, {})
            Gm.add_edge(u, v, key=key, route_id=route_id, route_short_name=rmeta.get('route_short_name', ''), route_long_name=rmeta.get('route_long_name', ''), direction_id=direction_id, trip_headsign=headsign, tau=np.nan, freq=np.nan, edge_type='transit', is_transit=True)
    return Gm

def compute_edge_times_and_freqs_from_gtfs(gtfs, Gm, window_start, window_end, date_obj=None, default_tau_min=2.0, service_key_mode='route_direction'):
    trips = filter_trips_for_date(gtfs, date_obj)
    if service_key_mode is None:
        service_key_mode = 'route_direction'
    stop_times = gtfs['stop_times'].copy()
    trip_cols = ['trip_id', 'route_id']
    if 'direction_id' in trips.columns:
        trip_cols.append('direction_id')
    if 'trip_headsign' in trips.columns:
        trip_cols.append('trip_headsign')
    st = stop_times.merge(trips[trip_cols], on='trip_id', how='inner')
    st['stop_sequence_int'] = st['stop_sequence'].astype(int)
    st = st.sort_values(['trip_id', 'stop_sequence_int'])
    ws = parse_gtfs_time(window_start)
    we = parse_gtfs_time(window_end)
    if ws is None or we is None or we <= ws:
        raise ValueError('Bad time window.')
    seg_durations = defaultdict(list)
    seg_counts = Counter()
    uv_any_durations = defaultdict(list)
    trip_meta = st.groupby('trip_id').agg({'route_id': 'first', **({'direction_id': 'first'} if 'direction_id' in st.columns else {}), **({'trip_headsign': 'first'} if 'trip_headsign' in st.columns else {})}).to_dict('index')
    for trip_id, grp in st.groupby('trip_id', sort=False):
        stop_ids = grp['stop_id'].tolist()
        arrs = grp['arrival_time'].tolist() if 'arrival_time' in grp.columns else [None] * len(grp)
        deps = grp['departure_time'].tolist() if 'departure_time' in grp.columns else [None] * len(grp)
        times = [parse_gtfs_time(d) if pd.notna(d) else parse_gtfs_time(a) for a, d in zip(arrs, deps)]
        meta = trip_meta[trip_id]
        key = make_service_key(meta.get('route_id', ''), meta.get('direction_id', 'na'), meta.get('trip_headsign', ''), mode=service_key_mode)
        for (u, tu), (v, tv) in zip(zip(stop_ids, times), zip(stop_ids[1:], times[1:])):
            if tu is None or tv is None:
                continue
            if not ws <= tu < we:
                continue
            dur = (tv - tu) / 60.0
            if dur <= 0:
                continue
            uv_any_durations[u, v].append(dur)
            if Gm.has_edge(u, v, key=key):
                k = key
            elif Gm.has_edge(u, v):
                k = next(iter(Gm[u][v].keys()))
            else:
                continue
            seg_durations[u, v, k].append(dur)
            seg_counts[u, v, k] += 1
    window_hours = (we - ws) / 3600.0
    for u, v, k, data in list(Gm.edges(keys=True, data=True)):
        durs = seg_durations.get((u, v, k), [])
        if durs:
            tau = float(np.median(durs)) # We take the median to avoid skewed values! It can be also considered the average otherwise
        else:
            uvd = uv_any_durations.get((u, v), [])
            tau = float(np.median(uvd)) if uvd else float(default_tau_min)
        cnt = seg_counts.get((u, v, k), 0)
        freq = float(cnt) / window_hours if window_hours > 0 else 0.0
        Gm[u][v][k]['tau'] = tau
        Gm[u][v][k]['freq'] = freq
    return Gm

def build_service_metadata(Gm):
    meta = {}
    for _, _, k, data in Gm.edges(keys=True, data=True):
        if k not in meta:
            meta[k] = {'service_key': k, 'route_id': data.get('route_id', ''), 'route_short_name': data.get('route_short_name', ''), 'route_long_name': data.get('route_long_name', ''), 'direction_id': data.get('direction_id', ''), 'trip_headsign': data.get('trip_headsign', '')}
    return meta

def line_id_from_service_meta(m):
    short = (m.get('route_short_name') or '').strip()
    dirid = m.get('direction_id', 'na')
    head = (m.get('trip_headsign') or '').strip()
    if short and head:
        return f'{short} | dir {dirid} | {head}'
    if short:
        return f'{short} | dir {dirid}'
    return f"route_id={m.get('route_id', 'na')} | dir {dirid}"

def expected_wait_minutes(freq_total, beta=30.0):
    if freq_total is None:
        return float('inf')
    f = float(freq_total)
    if not np.isfinite(f) or f <= 0:
        return float('inf')
    return float(beta) / f

def build_parallel_groups(Gm):
    groups = defaultdict(list)
    for u, v, k, data in Gm.edges(keys=True, data=True):
        tau = float(data.get('tau', np.nan))
        freq = float(data.get('freq', np.nan))
        if not np.isfinite(tau):
            tau = 2.0
        if not np.isfinite(freq):
            freq = 0.0
        groups[u, v].append({'key': k, 'tau': tau, 'freq': freq, 'data': data})
    return groups


# We consider two kind of graphs: a digraph (the aggregated version) and a multigraph (the service-level graph)
# Here we also include the beta for the waiting times, but we do not use it anymore in the use case (beta=0.0)
def build_aggregated_digraph_with_wait(Gm, beta=30.0):
    groups = build_parallel_groups(Gm)
    G = nx.DiGraph()
    for n, d in Gm.nodes(data=True):
        G.add_node(n, **d)
    for (u, v), services in groups.items():
        active_services = [s for s in services if float(s.get('freq', 0.0)) > 0]
        f_total = sum((max(0.0, s['freq']) for s in active_services))
        if f_total <= 0:
            continue
        tau_inveh = sum((max(0.0, s['freq']) * s['tau'] for s in active_services)) / f_total
        tau_gen = tau_inveh + expected_wait_minutes(f_total, beta=beta)
        G.add_edge(u, v, tau=tau_gen, tau_inveh=tau_inveh, freq=f_total, n_parallel_services=len(active_services), service_keys=[s['key'] for s in active_services], edge_type='transit', is_transit=True)
    return (G, groups)

def build_service_routing_multidigraph(Gm, beta=30.0):
    G = nx.MultiDiGraph()
    for n, d in Gm.nodes(data=True):
        G.add_node(n, **d)
    for u, v, k, data in Gm.edges(keys=True, data=True):
        tau = float(data.get('tau', np.nan))
        freq = float(data.get('freq', np.nan))
        if not np.isfinite(tau):
            tau = 2.0
        if not np.isfinite(freq) or freq <= 0:
            continue
        tau_gen = tau + expected_wait_minutes(freq, beta=beta)
        if not np.isfinite(tau_gen):
            continue
        G.add_edge(u, v, key=k, tau=tau_gen, tau_inveh=tau, freq=freq, route_id=data.get('route_id', ''), route_short_name=data.get('route_short_name', ''), route_long_name=data.get('route_long_name', ''), direction_id=data.get('direction_id', ''), trip_headsign=data.get('trip_headsign', ''), edge_type='transit', is_transit=True)
    return G

# We consider the haversine distance to reflect better the positions and the relative walking distances 
def _haversine_distance_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(a)))

def build_walking_connector_pairs(G, radius_m=300.0, max_neighbors=6):
    # We first consider a spatial grid to restrict the candidate stop pairs, and then we evaluate the haversine distance
    radius_m = float(radius_m)
    max_neighbors = None if max_neighbors is None else int(max_neighbors)
    if radius_m <= 0 or (max_neighbors is not None and max_neighbors <= 0):
        return []
    coords = {}
    lat_values = []
    for node, data in G.nodes(data=True):
        lat = float(data.get('lat', np.nan))
        lon = float(data.get('lon', np.nan))
        if np.isfinite(lat) and np.isfinite(lon):
            coords[node] = (lat, lon)
            lat_values.append(lat)
    if len(coords) < 2:
        return []
    earth_radius_m = 6371000.0
    ref_lat = math.radians(float(np.mean(lat_values)))
    projected = {node: (earth_radius_m * math.cos(ref_lat) * math.radians(lon), earth_radius_m * math.radians(lat)) for node, (lat, lon) in coords.items()}
    grid = defaultdict(list)
    for node, (x, y) in projected.items():
        grid[math.floor(x / radius_m), math.floor(y / radius_m)].append(node)
    selected_pairs = set()
    for node, (x, y) in projected.items():
        cell_x = math.floor(x / radius_m)
        cell_y = math.floor(y / radius_m)
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((cell_x + dx, cell_y + dy), []):
                    if other == node:
                        continue
                    lat1, lon1 = coords[node]
                    lat2, lon2 = coords[other]
                    distance_m = _haversine_distance_m(lat1, lon1, lat2, lon2)
                    if distance_m <= radius_m:
                        candidates.append((distance_m, str(other), other))
        candidates.sort(key=lambda row: (row[0], row[1]))
        for distance_m, _, other in candidates[:max_neighbors]:
            pair = tuple(sorted((node, other), key=str))
            selected_pairs.add((pair[0], pair[1], float(distance_m)))
    unique_pairs = {}
    for u, v, _ in selected_pairs:
        pair = tuple(sorted((u, v), key=str))
        lat1, lon1 = coords[pair[0]]
        lat2, lon2 = coords[pair[1]]
        unique_pairs[pair] = _haversine_distance_m(lat1, lon1, lat2, lon2)
    return [(u, v, distance_m) for (u, v), distance_m in sorted(unique_pairs.items(), key=lambda x: (str(x[0][0]), str(x[0][1])))]

def add_walking_connectors(Gagg, Gservice, radius_m=300.0, speed_kmh=4.5, max_neighbors=10):
    speed_kmh = float(speed_kmh)
    if speed_kmh <= 0:
        raise ValueError('WALK_SPEED_KMH must be positive.')
    rows = []
    pairs = build_walking_connector_pairs(Gagg, radius_m=radius_m, max_neighbors=max_neighbors)
    for u, v, distance_m in pairs:
        walk_minutes = max(float(distance_m) / 1000.0 / speed_kmh * 60.0, 1e-06)
        for a, b in ((u, v), (v, u)):
            if Gagg.has_edge(a, b):
                continue
            edge_data = {'tau': walk_minutes, 'tau_inveh': 0.0, 'freq': 0.0, 'edge_type': 'walking', 'is_transit': False, 'distance_m': float(distance_m), 'walking_speed_kmh': speed_kmh, 'service_keys': [], 'n_parallel_services': 0}
            Gagg.add_edge(a, b, **edge_data)
            walk_key = f'{WALK_KEY_PREFIX}|{a}|{b}'
            Gservice.add_edge(a, b, key=walk_key, **edge_data)
            rows.append({'u': a, 'v': b, 'u_name': Gagg.nodes[a].get('name', a), 'v_name': Gagg.nodes[b].get('name', b), 'distance_m': float(distance_m), 'walk_time_min': walk_minutes, 'walking_speed_kmh': speed_kmh})
    return pd.DataFrame(rows)

def _updated_bottleneck_frequency(current_bottleneck, edge_data):
    if not bool(edge_data.get('is_transit', True)):
        return current_bottleneck
    edge_frequency = float(edge_data.get('freq', 0.0))
    if not np.isfinite(current_bottleneck):
        return edge_frequency
    return min(current_bottleneck, edge_frequency)

def _finalize_bottleneck_frequencies(dist, bott, source):
    for node, distance in dist.items():
        if node != source and np.isfinite(distance) and (not np.isfinite(bott[node])):
            bott[node] = 0.0

def lexicographic_shortest_paths_digraph(G, source):
    dist = {n: math.inf for n in G.nodes}
    bott = {n: 0.0 for n in G.nodes}
    pred = {n: None for n in G.nodes}
    dist[source] = 0.0
    bott[source] = float('inf')
    pq = [(0.0, -bott[source], source)]
    while pq:
        dcur, neg_bcur, u = heapq.heappop(pq)
        bcur = -neg_bcur
        if dcur > dist[u] + 1e-09:
            continue
        if abs(dcur - dist[u]) <= 1e-09 and bcur < bott[u] - 1e-09:
            continue
        for v, data in G[u].items():
            tau = float(data.get('tau', 1000000000.0))
            nd = dcur + tau
            nb = _updated_bottleneck_frequency(bcur, data)
            better = False
            if nd < dist[v] - 1e-09:
                better = True
            elif abs(nd - dist[v]) <= 1e-09 and nb > bott[v] + 1e-09:
                better = True
            if better:
                dist[v] = nd
                bott[v] = nb
                pred[v] = u
                heapq.heappush(pq, (nd, -nb, v))
    _finalize_bottleneck_frequencies(dist, bott, source)
    return (dist, bott, pred)

def compute_all_pairs_canonical_paths_digraph(G):
    dst, wst, preds = ({}, {}, {})
    nodes = list(G.nodes)
    log(f'[digraph] canonical shortest paths for {len(nodes)} sources')
    for s in progress_iter(nodes, desc='Digraph shortest paths'):
        d, w, pred = lexicographic_shortest_paths_digraph(G, s)
        dst[s] = d
        wst[s] = w
        preds[s] = pred
    return (dst, wst, preds)

def second_best_from_source_digraph(G, source):
    INF = float('inf')
    best = {n: INF for n in G.nodes}
    second = {n: INF for n in G.nodes}
    best[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        dcur, u = heapq.heappop(pq)
        if dcur > second[u] + 1e-09:
            continue
        for v, data in G[u].items():
            w = float(data.get('tau', 1000000000.0))
            nd = dcur + w
            if nd + 1e-09 < best[v]:
                second[v] = best[v]
                best[v] = nd
                heapq.heappush(pq, (nd, v))
            elif best[v] + 1e-09 < nd < second[v] - 1e-09:
                second[v] = nd
                heapq.heappush(pq, (nd, v))
    return (best, second)

def compute_second_best_allpairs_digraph(G):
    d2 = {}
    nodes = list(G.nodes)
    log(f'[digraph] second-best paths for {len(nodes)} sources')
    for s in progress_iter(nodes, desc='Digraph backup paths'):
        _, second = second_best_from_source_digraph(G, s)
        d2[s] = second
    return d2

# Now we repeat similar functions but for the multigraph case (so we have to iterate over k lines)
def lexicographic_shortest_paths_multigraph(G, source):
    dist = {n: math.inf for n in G.nodes}
    bott = {n: 0.0 for n in G.nodes}
    pred = {n: None for n in G.nodes}
    dist[source] = 0.0
    bott[source] = float('inf')
    pq = [(0.0, -bott[source], source)]
    while pq:
        dcur, neg_bcur, u = heapq.heappop(pq)
        bcur = -neg_bcur
        if dcur > dist[u] + 1e-09:
            continue
        if abs(dcur - dist[u]) <= 1e-09 and bcur < bott[u] - 1e-09:
            continue
        for v, keydict in G[u].items():
            for k, data in keydict.items(): # This is the main difference with the digraph only
                tau = float(data.get('tau', 1000000000.0))
                nd = dcur + tau
                nb = _updated_bottleneck_frequency(bcur, data)
                better = False
                if nd < dist[v] - 1e-09:
                    better = True
                elif abs(nd - dist[v]) <= 1e-09 and nb > bott[v] + 1e-09:
                    better = True
                if better:
                    dist[v] = nd
                    bott[v] = nb
                    pred[v] = (u, k)
                    heapq.heappush(pq, (nd, -nb, v))
    _finalize_bottleneck_frequencies(dist, bott, source)
    return (dist, bott, pred)

def compute_all_pairs_canonical_paths_multigraph(G):
    dst, wst, preds = ({}, {}, {})
    nodes = list(G.nodes)
    log(f'[multigraph] canonical shortest paths for {len(nodes)} sources')
    for s in progress_iter(nodes, desc='Service shortest paths'):
        d, w, pred = lexicographic_shortest_paths_multigraph(G, s)
        dst[s] = d
        wst[s] = w
        preds[s] = pred
    return (dst, wst, preds)

def second_best_from_source_multigraph(G, source):
    INF = float('inf')
    best = {n: INF for n in G.nodes}
    second = {n: INF for n in G.nodes}
    best[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        dcur, u = heapq.heappop(pq)
        if dcur > second[u] + 1e-09:
            continue
        for v, keydict in G[u].items():
            for _, data in keydict.items():
                w = float(data.get('tau', 1000000000.0))
                nd = dcur + w
                if nd + 1e-09 < best[v]:
                    second[v] = best[v]
                    best[v] = nd
                    heapq.heappush(pq, (nd, v))
                elif best[v] + 1e-09 < nd < second[v] - 1e-09:
                    second[v] = nd
                    heapq.heappush(pq, (nd, v))
    return (best, second)

def compute_second_best_allpairs_multigraph(G):
    d2 = {}
    nodes = list(G.nodes)
    log(f'[multigraph] second-best paths for {len(nodes)} sources')
    for s in progress_iter(nodes, desc='Service backup paths'):
        _, second = second_best_from_source_multigraph(G, s)
        d2[s] = second
    return d2

def _service_route_token(service_key, service_meta):
    meta = (service_meta or {}).get(service_key, {})
    route_id = meta.get('route_id', '')
    if route_id is not None and str(route_id).strip():
        return str(route_id).strip()
    route_short_name = meta.get('route_short_name', '')
    if route_short_name is not None and str(route_short_name).strip():
        return str(route_short_name).strip()
    return str(service_key).split('|')[0]

def _histogram_quantile(counter, q):
    total = int(sum(counter.values()))
    if total <= 0:
        return np.nan
    rank = max(1, int(math.ceil(float(q) * total)))
    cumulative = 0
    for value in sorted(counter):
        cumulative += int(counter[value])
        if cumulative >= rank:
            return int(value)
    return int(max(counter))

def _new_path_stat_accumulator():
    return {'count': 0, 'sum_transfers': 0.0, 'max_transfers': 0, 'transfer_hist': Counter(), 'sum_segments': 0.0, 'max_segments': 0, 'segment_hist': Counter(), 'sum_walking_connectors': 0.0, 'max_walking_connectors': 0, 'walking_connector_hist': Counter(), 'paths_using_walking': 0, 'all_walking_paths': 0, 'sum_total_arcs': 0.0, 'max_total_arcs': 0, 'sum_distance': 0.0, 'max_distance': 0.0, 'extreme_examples': []}

def _update_path_stat_accumulator(acc, transfers, segments, walking_connectors, total_arcs, distance, source, target, G, max_examples=10):
    transfers = int(transfers)
    segments = int(segments)
    walking_connectors = int(walking_connectors)
    total_arcs = int(total_arcs)
    distance = float(distance)
    acc['count'] += 1
    acc['sum_transfers'] += transfers
    acc['max_transfers'] = max(acc['max_transfers'], transfers)
    acc['transfer_hist'][transfers] += 1
    acc['sum_segments'] += segments
    acc['max_segments'] = max(acc['max_segments'], segments)
    acc['segment_hist'][segments] += 1
    acc['sum_walking_connectors'] += walking_connectors
    acc['max_walking_connectors'] = max(acc['max_walking_connectors'], walking_connectors)
    acc['walking_connector_hist'][walking_connectors] += 1
    acc['paths_using_walking'] += int(walking_connectors > 0)
    acc['all_walking_paths'] += int(walking_connectors > 0 and segments == 0)
    acc['sum_total_arcs'] += total_arcs
    acc['max_total_arcs'] = max(acc['max_total_arcs'], total_arcs)
    acc['sum_distance'] += distance
    acc['max_distance'] = max(acc['max_distance'], distance)
    example = {'source': source, 'source_name': G.nodes[source].get('name', source), 'target': target, 'target_name': G.nodes[target].get('name', target), 'transfers': transfers, 'transit_segments': segments, 'walking_connectors': walking_connectors, 'total_arcs': total_arcs, 'distance_min': distance}
    current_max = max((row['transfers'] for row in acc['extreme_examples']), default=-1)
    if transfers > current_max:
        acc['extreme_examples'] = [example]
    elif transfers == current_max and len(acc['extreme_examples']) < max_examples:
        acc['extreme_examples'].append(example)

def _finalize_path_statistics(model_name, definition, interpretation, acc):
    count = int(acc['count'])
    transfer_hist = acc['transfer_hist']
    summary = {'model': model_name, 'path_type': 'canonical_shortest_path', 'transfer_definition': definition, 'interpretation': interpretation, 'reachable_ordered_od_pairs': count, 'average_transfers': acc['sum_transfers'] / count if count else np.nan, 'median_transfers': _histogram_quantile(transfer_hist, 0.5), 'p95_transfers': _histogram_quantile(transfer_hist, 0.95), 'maximum_transfers': acc['max_transfers'] if count else np.nan, 'share_zero_transfers': transfer_hist.get(0, 0) / count if count else np.nan, 'share_at_most_one_transfer': (transfer_hist.get(0, 0) + transfer_hist.get(1, 0)) / count if count else np.nan, 'segment_definition': 'number of transit stop-to-stop arcs', 'average_segments': acc['sum_segments'] / count if count else np.nan, 'maximum_segments': acc['max_segments'] if count else np.nan, 'average_walking_connectors': acc['sum_walking_connectors'] / count if count else np.nan, 'maximum_walking_connectors': acc['max_walking_connectors'] if count else np.nan, 'share_paths_using_walking': acc['paths_using_walking'] / count if count else np.nan, 'share_all_walking_paths': acc['all_walking_paths'] / count if count else np.nan, 'average_total_arcs': acc['sum_total_arcs'] / count if count else np.nan, 'maximum_total_arcs': acc['max_total_arcs'] if count else np.nan, 'average_path_cost_min': acc['sum_distance'] / count if count else np.nan, 'maximum_path_cost_min': acc['max_distance'] if count else np.nan}
    transfer_distribution = [{'model': model_name, 'path_type': 'canonical_shortest_path', 'transfer_definition': definition, 'transfers': int(value), 'ordered_od_pairs': int(n_pairs), 'share': float(n_pairs) / count if count else np.nan} for value, n_pairs in sorted(transfer_hist.items())]
    segment_distribution = [{'model': model_name, 'path_type': 'canonical_shortest_path', 'segments': int(value), 'ordered_od_pairs': int(n_pairs), 'share': float(n_pairs) / count if count else np.nan} for value, n_pairs in sorted(acc['segment_hist'].items())]
    walking_distribution = [{'model': model_name, 'path_type': 'canonical_shortest_path', 'walking_connectors': int(value), 'ordered_od_pairs': int(n_pairs), 'share': float(n_pairs) / count if count else np.nan} for value, n_pairs in sorted(acc['walking_connector_hist'].items())]
    extreme_examples = []
    for row in acc['extreme_examples']:
        extreme_examples.append({'model': model_name, 'path_type': 'canonical_shortest_path', 'transfer_definition': definition, **row})
    return (summary, transfer_distribution, segment_distribution, walking_distribution, extreme_examples)

def compute_canonical_path_statistics(G, dst, preds, model_name, service_meta=None):
    is_multi = isinstance(G, (nx.MultiDiGraph, nx.MultiGraph))
    definitions = {'service_pattern_change': _new_path_stat_accumulator(), 'route_change': _new_path_stat_accumulator()}
    for s in progress_iter(list(G.nodes), desc=f'{model_name} path statistics'):
        pred = preds[s]
        reachable = [t for t, distance in dst[s].items() if t != s and np.isfinite(distance) and (distance > 0)]
        reachable.sort(key=lambda t: float(dst[s][t]))
        transit_segments = {s: 0}
        walking_connectors = {s: 0}
        total_arcs = {s: 0}
        if is_multi:
            service_transfers = {s: 0}
            route_transfers = {s: 0}
            last_service = {s: None}
            last_route = {s: None}
            used_transit = {s: False}
        else:
            service_dp = {s: {(None, False): 0}}
            route_dp = {s: {(None, False): 0}}
        for t in reachable:
            if is_multi:
                step = pred.get(t)
                if step is None:
                    continue
                parent, service_key = step
                if parent not in transit_segments:
                    continue
                edge_data = G[parent][t][service_key]
                is_transit_edge = bool(edge_data.get('is_transit', True))
                total_arcs[t] = total_arcs[parent] + 1
                if not is_transit_edge:
                    transit_segments[t] = transit_segments[parent]
                    walking_connectors[t] = walking_connectors[parent] + 1
                    service_transfers[t] = service_transfers[parent]
                    route_transfers[t] = route_transfers[parent]
                    last_service[t] = None
                    last_route[t] = None
                    used_transit[t] = used_transit[parent]
                else:
                    route_token = _service_route_token(service_key, service_meta)
                    transit_segments[t] = transit_segments[parent] + 1
                    walking_connectors[t] = walking_connectors[parent]
                    if not used_transit[parent]:
                        service_increment = 0
                        route_increment = 0
                    else:
                        service_increment = int(last_service[parent] is None or last_service[parent] != service_key)
                        route_increment = int(last_route[parent] is None or last_route[parent] != route_token)
                    service_transfers[t] = service_transfers[parent] + service_increment
                    route_transfers[t] = route_transfers[parent] + route_increment
                    last_service[t] = service_key
                    last_route[t] = route_token
                    used_transit[t] = True
                transfer_values = {'service_pattern_change': service_transfers[t], 'route_change': route_transfers[t]}
            else:
                parent = pred.get(t)
                if parent is None or parent not in transit_segments:
                    continue
                edge_data = G[parent][t]
                is_transit_edge = bool(edge_data.get('is_transit', True))
                total_arcs[t] = total_arcs[parent] + 1
                previous_service_dp = service_dp[parent]
                previous_route_dp = route_dp[parent]
                if not is_transit_edge:
                    transit_segments[t] = transit_segments[parent]
                    walking_connectors[t] = walking_connectors[parent] + 1
                    service_dp[t] = {}
                    route_dp[t] = {}
                    for (_, has_used), value in previous_service_dp.items():
                        state = (None, has_used)
                        service_dp[t][state] = min(service_dp[t].get(state, math.inf), value)
                    for (_, has_used), value in previous_route_dp.items():
                        state = (None, has_used)
                        route_dp[t][state] = min(route_dp[t].get(state, math.inf), value)
                else:
                    available_services = list(dict.fromkeys(edge_data.get('service_keys', [])))
                    if not available_services:
                        available_services = [f'{parent}->{t}']
                    available_routes = list(dict.fromkeys((_service_route_token(key, service_meta) for key in available_services)))
                    transit_segments[t] = transit_segments[parent] + 1
                    walking_connectors[t] = walking_connectors[parent]
                    service_dp[t] = {}
                    route_dp[t] = {}
                    for key in available_services:
                        best = min((value + int(has_used and (previous_key is None or previous_key != key)) for (previous_key, has_used), value in previous_service_dp.items()))
                        service_dp[t][key, True] = best
                    for key in available_routes:
                        best = min((value + int(has_used and (previous_key is None or previous_key != key)) for (previous_key, has_used), value in previous_route_dp.items()))
                        route_dp[t][key, True] = best
                transfer_values = {'service_pattern_change': min(service_dp[t].values()), 'route_change': min(route_dp[t].values())}
            for definition, transfers in transfer_values.items():
                _update_path_stat_accumulator(definitions[definition], transfers, transit_segments[t], walking_connectors[t], total_arcs[t], dst[s][t], s, t, G)
    if is_multi:
        interpretations = {'service_pattern_change': 'Exact changes of route-direction service key along the selected service-level path; boarding transit after a walking connector counts as a change if transit was used previously.', 'route_change': 'Exact changes of route_id along the selected service-level path; boarding transit after a walking connector counts as a change if transit was used previously.'}
    else:
        interpretations = {'service_pattern_change': 'Minimum route-direction service-key changes compatible with the aggregated stop sequence; reboarding after walking counts as a change if transit was used previously.', 'route_change': 'Minimum route_id changes compatible with the aggregated stop sequence; reboarding after walking counts as a change if transit was used previously.'}
    summary_rows = []
    transfer_rows = []
    segment_rows = []
    walking_rows = []
    extreme_rows = []
    for definition, acc in definitions.items():
        summary, transfers, segment_distribution, walking_distribution, extremes = _finalize_path_statistics(model_name, definition, interpretations[definition], acc)
        summary_rows.append(summary)
        transfer_rows.extend(transfers)
        if definition == 'route_change':
            segment_rows.extend(segment_distribution)
            walking_rows.extend(walking_distribution)
        extreme_rows.extend(extremes)
    return {'summary': pd.DataFrame(summary_rows), 'transfer_distribution': pd.DataFrame(transfer_rows), 'segment_distribution': pd.DataFrame(segment_rows), 'walking_distribution': pd.DataFrame(walking_rows), 'extreme_examples': pd.DataFrame(extreme_rows)}

# In the following we compute the node fragility measures
def compute_BFrag_bundle(dst, wst, dbackup, demand_matrix=None):
    Bst_map = {s: {} for s in dst}
    source_sum = {s: 0.0 for s in dst}
    efw_num_source = {s: 0.0 for s in dst}
    Z_source = {s: 0.0 for s in dst}
    Z = 0.0
    total_B = 0.0
    for s in dst:
        for t, d in dst[s].items():
            if s == t:
                continue
            Dst = 1.0
            if demand_matrix is not None:
                Dst = float(demand_matrix.get(s, {}).get(t, 1.0))
            Z += Dst
            Z_source[s] += Dst
            if np.isfinite(d) and d > 0:
                w = float(wst[s].get(t, 0.0))
                efw_num_source[s] += Dst * (w / d)
                db = dbackup.get(s, {}).get(t, np.inf)
                inv_d = 1.0 / d
                inv_db = 0.0 if not np.isfinite(db) or db <= 0 else 1.0 / db
                B = Dst * w * (inv_d - inv_db)
            else:
                B = 0.0
            Bst_map[s][t] = B
            source_sum[s] += B
            total_B += B
    EFW_V = {s: efw_num_source[s] / Z_source[s] if Z_source[s] > 0 else np.nan for s in dst}
    EFW_G = sum(efw_num_source.values()) / Z if Z > 0 else np.nan
    BFrag_G = total_B / Z if Z > 0 else np.nan
    BFrag_V = {s: source_sum[s] / Z_source[s] if Z_source[s] > 0 else np.nan for s in dst}
    NBFrag_G = BFrag_G / EFW_G if np.isfinite(BFrag_G) and np.isfinite(EFW_G) and (EFW_G > 0) else np.nan
    NBFrag_V = {s: BFrag_V[s] / EFW_V[s] if np.isfinite(BFrag_V[s]) and np.isfinite(EFW_V[s]) and (EFW_V[s] > 0) else np.nan for s in dst}
    return (Bst_map, BFrag_G, BFrag_V, EFW_G, EFW_V, Z, NBFrag_G, NBFrag_V)

# We compute now the fragility edge metrics
def compute_edge_metrics_fast_digraph(G, dst, wst, preds, Bst_map):
    DElas = defaultdict(float)
    SAC = defaultdict(float)
    nodes = list(G.nodes)
    log(f'[edge metrics] computing digraph DElas/SAC on {len(nodes)} sources')
    for s in progress_iter(nodes, desc='Digraph edge metrics'):
        pred = preds[s]
        children = defaultdict(list)
        for v, p in pred.items():
            if p is not None:
                children[p].append(v)
        v_delas = {}
        v_sac = {}
        for t in nodes:
            if t == s:
                continue
            d = dst[s].get(t, np.inf)
            if not np.isfinite(d) or d <= 0:
                continue
            w = float(wst[s].get(t, 0.0))
            v_delas[t] = w / d ** 2
            v_sac[t] = float(Bst_map[s].get(t, 0.0))
        roots = [v for v, p in pred.items() if p is None]
        agg_d = defaultdict(float)
        agg_s = defaultdict(float)
        for r in roots:
            stack = [(r, 0)]
            post = []
            while stack:
                node, state = stack.pop()
                if state == 0:
                    stack.append((node, 1))
                    for c in children.get(node, []):
                        stack.append((c, 0))
                else:
                    post.append(node)
            for v in post:
                subtotal_d = float(v_delas.get(v, 0.0))
                subtotal_s = float(v_sac.get(v, 0.0))
                for c in children.get(v, []):
                    subtotal_d += agg_d[c]
                    subtotal_s += agg_s[c]
                agg_d[v] = subtotal_d
                agg_s[v] = subtotal_s
                p = pred.get(v)
                if p is not None and bool(G[p][v].get('is_transit', True)):
                    DElas[p, v] += agg_d[v]
                    SAC[p, v] += agg_s[v]
    return (dict(DElas), dict(SAC))

# We create the edge to pairs so that we can use them directly for the F_gap
def build_edge_to_pairs_digraph(G, preds):
    edge_to_pairs = defaultdict(list)
    log(f'[edge coverage] building digraph edge-to-pairs map for {len(preds)} sources')
    for s in progress_iter(preds, desc='Digraph edge coverage', total=len(preds)):
        pred = preds[s]
        for t in pred:
            if s == t:
                continue
            path = []
            cur = t
            while cur is not None and cur != s:
                p = pred.get(cur)
                if p is None:
                    path = []
                    break
                if bool(G[p][cur].get('is_transit', True)):
                    path.append((p, cur))
                cur = p
            if cur != s or not path:
                continue
            for e in path:
                edge_to_pairs[e].append((s, t))
    return edge_to_pairs

def compute_edge_metrics_fast_multigraph(G, dst, wst, preds, Bst_map):
    DElas = defaultdict(float)
    SAC = defaultdict(float)
    nodes = list(G.nodes)
    log(f'[edge metrics] computing multigraph DElas/SAC on {len(nodes)} sources')
    for s in progress_iter(nodes, desc='Service edge metrics'):
        pred = preds[s]
        children = defaultdict(list)
        pred_edge = {}
        for v, step in pred.items():
            if step is None:
                continue
            p, k = step
            children[p].append(v)
            if bool(G[p][v][k].get('is_transit', True)):
                pred_edge[v] = (p, v, k)
        v_delas = {}
        v_sac = {}
        for t in nodes:
            if t == s:
                continue
            d = dst[s].get(t, np.inf)
            if not np.isfinite(d) or d <= 0:
                continue
            w = float(wst[s].get(t, 0.0))
            v_delas[t] = w / d ** 2
            v_sac[t] = float(Bst_map[s].get(t, 0.0))
        roots = [v for v in nodes if pred.get(v) is None]
        agg_d = defaultdict(float)
        agg_s = defaultdict(float)
        for r in roots:
            stack = [(r, 0)]
            post = []
            while stack:
                node, state = stack.pop()
                if state == 0:
                    stack.append((node, 1))
                    for c in children.get(node, []):
                        stack.append((c, 0))
                else:
                    post.append(node)
            for v in post:
                subtotal_d = float(v_delas.get(v, 0.0))
                subtotal_s = float(v_sac.get(v, 0.0))
                for c in children.get(v, []):
                    subtotal_d += agg_d[c]
                    subtotal_s += agg_s[c]
                agg_d[v] = subtotal_d
                agg_s[v] = subtotal_s
                if v in pred_edge:
                    e = pred_edge[v]
                    DElas[e] += agg_d[v]
                    SAC[e] += agg_s[v]
    return (dict(DElas), dict(SAC))

def build_edge_to_pairs_multigraph(G, preds):
    edge_to_pairs = defaultdict(list)
    log(f'[edge coverage] building service edge-to-pairs map for {len(preds)} sources')
    for s in progress_iter(preds, desc='Service edge coverage', total=len(preds)):
        pred = preds[s]
        for t in pred:
            if s == t:
                continue
            path = []
            cur = t
            while cur is not None and cur != s:
                step = pred.get(cur)
                if step is None:
                    path = []
                    break
                p, k = step
                if bool(G[p][cur][k].get('is_transit', True)):
                    path.append((p, cur, k))
                cur = p
            if cur != s or not path:
                continue
            for e in path:
                edge_to_pairs[e].append((s, t))
    return edge_to_pairs

# We use a lazy-greedy implementation where the marginal gains are recomputed only when an edge reaches the top of the priority queue
def greedy_topk_from_edge_to_pairs(edge_to_pairs, Bst_map, k):
    pair_weight = {(s, t): float(w) for s, row in Bst_map.items() for t, w in row.items() if float(w) > 0}
    edge_pairs = {e: set(pairs) for e, pairs in edge_to_pairs.items()}
    heap = []
    for e, pairs in edge_pairs.items():
        gain = sum((pair_weight.get(pair, 0.0) for pair in pairs))
        if gain > 0:
            heapq.heappush(heap, (-gain, e))
    selected = []
    selected_set = set()
    covered = set()
    current = 0.0
    log(f'[top-k] evaluating exact greedy selection over {len(edge_pairs)} candidate edges (k={k})')
    for step_idx in progress_iter(range(1, k + 1), desc='Exact greedy top-k', total=k):
        best = None
        while heap:
            neg_gain, e = heapq.heappop(heap)
            if e in selected_set:
                continue
            true_gain = 0.0
            for pair in edge_pairs[e]:
                if pair not in covered:
                    true_gain += pair_weight.get(pair, 0.0)
            if not heap or true_gain >= -heap[0][0] - 1e-12:
                best = (e, true_gain)
                break
            heapq.heappush(heap, (-true_gain, e))
        if best is None or best[1] <= 0:
            log(f'[top-k] stopped early at step {step_idx}: no positive marginal gain left')
            break
        best_e, best_gain = best
        selected.append((best_e, best_gain))
        selected_set.add(best_e)
        covered.update(edge_pairs[best_e])
        current += best_gain
        log(f'[top-k] step {len(selected)}/{k}: selected {best_e} | marginal={best_gain:.6g} | cumulative={current:.6g} | covered_pairs={len(covered)}')
    return (selected, current)

def Fgap_for_edge_set(edge_to_pairs, Bst_map, edge_set):
    covered = set()
    total = 0.0
    for e in edge_set:
        for s, t in edge_to_pairs.get(e, []):
            if (s, t) in covered:
                continue
            covered.add((s, t))
            total += float(Bst_map[s].get(t, 0.0))
    return total

def build_line_to_service_edges(serviceG, service_meta):
    out = defaultdict(set)
    for u, v, k, data in serviceG.edges(keys=True, data=True):
        if not bool(data.get('is_transit', True)):
            continue
        meta = service_meta.get(k, {'service_key': k, 'route_short_name': data.get('route_short_name', ''), 'direction_id': data.get('direction_id', ''), 'trip_headsign': data.get('trip_headsign', '')})
        lid = line_id_from_service_meta(meta)
        out[lid].add((u, v, k))
    return out

# For the line criticality we just have to iterate for each line the SAC computation
def compute_line_criticality_service(serviceG, service_meta, edge_to_pairs, Bst_map, SAC, DElas):
    line_to_edges = build_line_to_service_edges(serviceG, service_meta)
    rows = []
    log(f'[lines] service line criticality for {len(line_to_edges)} lines')
    for lid, edgeset in progress_iter(line_to_edges.items(), desc='Service line criticality', total=len(line_to_edges)):
        fgap = Fgap_for_edge_set(edge_to_pairs, Bst_map, edgeset)
        sac_sum = sum((float(SAC.get(e, 0.0)) for e in edgeset))
        delas_sum = sum((float(DElas.get(e, 0.0)) for e in edgeset))
        freq_sum = 0.0
        unique_stops = set()
        for u, v, k in edgeset:
            freq_sum += float(serviceG[u][v][k]['freq'])
            unique_stops.update([u, v])
        seg_count = len(edgeset)
        rows.append({'line_id': lid, 'n_service_edges': len(edgeset), 'Fgap_line_exact': fgap, 'SAC_sum': sac_sum, 'DElas_sum': delas_sum, 'freq_sum_over_segments': freq_sum, 'freq_mean_over_segments': freq_sum / seg_count if seg_count > 0 else np.nan, 'segments_count': seg_count, 'unique_stops_count': len(unique_stops)})
    return pd.DataFrame(rows).sort_values('Fgap_line_exact', ascending=False)

# We need this since the betweenness centrality has to be computed in undirected graphs!
def as_undirected_topology(G):
    if isinstance(G, nx.MultiDiGraph) or isinstance(G, nx.MultiGraph):
        H = nx.MultiGraph()
    else:
        H = nx.Graph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, k, data in G.edges(keys=True, data=True):
            if H.has_edge(u, v, key=k):
                existing = H[u][v][k]
                if bool(existing.get('is_transit', True)) and (not bool(data.get('is_transit', True))):
                    continue
            H.add_edge(u, v, key=k, **data)
    else:
        for u, v, data in G.edges(data=True):
            if H.has_edge(u, v):
                existing = H[u][v]
                if bool(existing.get('is_transit', True)) and (not bool(data.get('is_transit', True))):
                    continue
            H.add_edge(u, v, **data)
    return H

def _edge_data_is_transit(data):
    return bool((data or {}).get('is_transit', True))

def _transit_edge_candidates(G, candidate_edges=None):
    is_multi = isinstance(G, (nx.MultiDiGraph, nx.MultiGraph))
    if candidate_edges is None:
        if is_multi:
            return [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if _edge_data_is_transit(data)]
        return [(u, v) for u, v, data in G.edges(data=True) if _edge_data_is_transit(data)]
    filtered = []
    for edge in candidate_edges:
        if is_multi:
            if len(edge) != 3:
                continue
            u, v, k = edge
            data = G.get_edge_data(u, v, key=k)
            if data is None:
                data = G.get_edge_data(v, u, key=k)
        else:
            if len(edge) != 2:
                continue
            u, v = edge
            data = G.get_edge_data(u, v)
            if data is None:
                data = G.get_edge_data(v, u)
        if data is not None and _edge_data_is_transit(data):
            filtered.append(edge)
    return filtered

def global_efficiency_unweighted_graph(G):
    nodes = list(G.nodes)
    N = len(nodes)
    if N <= 1:
        return 0.0
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    total = 0.0
    for i in nodes:
        li = lengths.get(i, {})
        for j in nodes:
            if i == j:
                continue
            d = li.get(j)
            if d is None or d <= 0:
                continue
            total += 1.0 / d
    return total / (N * (N - 1))

def node_contraction_graph(G, v):
    H = nx.Graph()
    UG = G if isinstance(G, nx.Graph) else nx.Graph(G)
    cluster = set([v]) | set(UG.neighbors(v))
    supernode = f'contract_{v}'
    for n, d in UG.nodes(data=True):
        if n in cluster:
            continue
        H.add_node(n, **d)
    lats = [UG.nodes[x].get('lat', np.nan) for x in cluster]
    lons = [UG.nodes[x].get('lon', np.nan) for x in cluster]
    H.add_node(supernode, lat=float(np.nanmean(lats)), lon=float(np.nanmean(lons)), name=f'contract({v})')
    for a, b in UG.edges():
        if a in cluster or b in cluster:
            continue
        H.add_edge(a, b)
    boundary = set()
    for x in cluster:
        for y in UG.neighbors(x):
            if y not in cluster:
                boundary.add(y)
    for y in boundary:
        H.add_edge(supernode, y)
    return H

def network_agglomeration_phi(G):
    nodes = list(G.nodes)
    N = len(nodes)
    if N <= 1:
        return 0.0
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    s = 0.0
    cnt = 0
    for i in nodes:
        li = lengths.get(i, {})
        for j in nodes:
            if i == j:
                continue
            d = li.get(j)
            if d is None:
                continue
            s += d
            cnt += 1
    if cnt == 0:
        return 0.0
    L = s / cnt
    return 1.0 / (N * L) if L > 0 else 0.0

# In here we approximate the flow by considering the times and frequency (since only these info are available)
def total_flow_proxy(G):
    tot = 0.0
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for _, _, _, data in G.edges(keys=True, data=True):
            f = float(data.get('freq', 0.0))
            tau = float(data.get('tau_inveh', data.get('tau', 0.0)))
            if np.isfinite(f) and np.isfinite(tau):
                tot += f * tau
    else:
        for _, _, data in G.edges(data=True):
            f = float(data.get('freq', 0.0))
            tau = float(data.get('tau_inveh', data.get('tau', 0.0)))
            if np.isfinite(f) and np.isfinite(tau):
                tot += f * tau
    return tot

def compute_literature_node_metrics(G):
    UG = as_undirected_topology(G)
    E0 = global_efficiency_unweighted_graph(UG)
    phi0 = network_agglomeration_phi(nx.Graph(UG))
    F0 = total_flow_proxy(G)
    node_bet = nx.betweenness_centrality(UG)
    rows = []
    for n in progress_iter(UG.nodes, desc="Literature node measures", total=UG.number_of_nodes()):
        H = UG.copy()
        if H.has_node(n):
            H.remove_node(n)
        E1 = global_efficiency_unweighted_graph(H) if H.number_of_nodes() > 1 else 0.0
        HC = node_contraction_graph(nx.Graph(UG), n)
        phi1 = network_agglomeration_phi(HC) if HC.number_of_nodes() > 1 else 0.0
        Gcopy = G.copy()
        if Gcopy.has_node(n):
            Gcopy.remove_node(n)
        F1 = total_flow_proxy(Gcopy)
        rows.append({
            'node': n,
            'BetV': float(node_bet.get(n, np.nan)),
            'CV': (E0 - E1) / E0 if E0 > 0 else np.nan,
            'CF': (F0 - F1) / F0 if F0 > 0 else np.nan,
            'IMV': (phi1 - phi0) / phi0 if phi0 > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def compute_literature_edge_metrics(G):
    UG = as_undirected_topology(G)
    E0 = global_efficiency_unweighted_graph(UG)
    edge_bet = nx.edge_betweenness_centrality(UG)
    rows = []
    iterator = _transit_edge_candidates(UG)
    for edge in progress_iter(iterator, desc="Literature edge measures", total=len(iterator)):
        H = UG.copy()
        if isinstance(UG, nx.MultiGraph):
            u, v, k = edge
            if H.has_edge(u, v, key=k):
                H.remove_edge(u, v, key=k)
            edge_id = (u, v, k)
            bet_val = edge_bet.get((u, v, k), np.nan)
        else:
            u, v = edge
            if H.has_edge(u, v):
                H.remove_edge(u, v)
            edge_id = (u, v)
            bet_val = edge_bet.get((u, v), edge_bet.get((v, u), np.nan))
        E1 = global_efficiency_unweighted_graph(H) if H.number_of_nodes() > 1 else 0.0
        rows.append({
            'edge': edge_id,
            'BetE': float(bet_val),
            'CE': (E0 - E1) / E0 if E0 > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def compute_model_result(
    graph,
    model_name,
    topk,
    service_meta=None,
    demand_matrix=None,
    record_algorithm_timings=False,
    collect_path_statistics=False,
):
    is_multi = isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph))
    weight_mode = 'freq'
    case_label = model_name
    log(f'[model] starting {case_label} | nodes={graph.number_of_nodes()} | edges={graph.number_of_edges()}')
    with pipeline_section(f'{case_label} - shortest paths'):
        with algorithm_timer(graph, model_name, weight_mode, 1, 'Canonical all-pairs shortest paths', 'O(N M log N)', enabled=record_algorithm_timings):
            if is_multi:
                dst, wst_base, preds = compute_all_pairs_canonical_paths_multigraph(graph)
            else:
                dst, wst_base, preds = compute_all_pairs_canonical_paths_digraph(graph)
    path_statistics = None
    if collect_path_statistics:
        with pipeline_section(f'{case_label} - canonical path statistics'):
            path_statistics = compute_canonical_path_statistics(graph, dst, preds, model_name, service_meta=service_meta)
    with pipeline_section(f'{case_label} - backup paths'):
        with algorithm_timer(graph, model_name, weight_mode, 2, 'All-pairs second-best distances', 'O(N M log N)', enabled=record_algorithm_timings):
            if is_multi:
                d2 = compute_second_best_allpairs_multigraph(graph)
            else:
                d2 = compute_second_best_allpairs_digraph(graph)
    with pipeline_section(f'{case_label} - fragility bundle'):
        with algorithm_timer(graph, model_name, weight_mode, 3, 'BFrag and NBFrag aggregation', 'O(N^2)', enabled=record_algorithm_timings):
            wst = wst_base
            Bst_map, BFrag_G, BFrag_V, EFW_G, EFW_V, Z, NBFrag_G, NBFrag_V = compute_BFrag_bundle(dst, wst, d2, demand_matrix=demand_matrix)
    with pipeline_section(f'{case_label} - edge metrics'):
        with algorithm_timer(graph, model_name, weight_mode, 4, 'SAC and DElas aggregation', 'O(N M)', enabled=record_algorithm_timings):
            if is_multi:
                DElas, SAC = compute_edge_metrics_fast_multigraph(graph, dst, wst, preds, Bst_map)
            else:
                DElas, SAC = compute_edge_metrics_fast_digraph(graph, dst, wst, preds, Bst_map)
    with pipeline_section(f'{case_label} - greedy top-{topk} Fgap'):
        with algorithm_timer(graph, model_name, weight_mode, 5, f'Greedy top-{topk} Fgap', 'O(k N M)', enabled=record_algorithm_timings):
            if is_multi:
                edge_to_pairs = build_edge_to_pairs_multigraph(graph, preds)
            else:
                edge_to_pairs = build_edge_to_pairs_digraph(graph, preds)
            topk_edges, Fgap_topk = greedy_topk_from_edge_to_pairs(edge_to_pairs, Bst_map, topk)
    line_df = None
    if service_meta is not None:
        with pipeline_section(f'{case_label} - line criticality'):
            line_df = compute_line_criticality_service(graph, service_meta, edge_to_pairs, Bst_map, SAC, DElas)
    result = {'model_name': model_name, 'weight_mode': weight_mode, 'graph': graph, 'dst': dst, 'wst': wst, 'preds': preds, 'd2': d2, 'Bst_map': Bst_map, 'EFW_G': EFW_G, 'EFW_V': EFW_V, 'BFrag_G': BFrag_G, 'BFrag_V': BFrag_V, 'NBFrag_G': NBFrag_G, 'NBFrag_V': NBFrag_V, 'Z': Z, 'DElas': DElas, 'SAC': SAC, 'edge_to_pairs': edge_to_pairs, 'topk_edges': topk_edges, 'Fgap_topk': Fgap_topk, 'line_df': line_df, 'path_statistics': path_statistics}
    log(f'[model] completed {case_label}')
    return result

# Below we report the functions for saving tables and figures for the paper 

def metric_summary_table(result_dict):
    rows = []
    for key, res in result_dict.items():
        rows.append({'case': key, 'model': res['model_name'], 'weight_mode': res['weight_mode'], 'EFW_G': res['EFW_G'], 'BFrag_G': res['BFrag_G'], 'NBFrag_G': res['NBFrag_G'], 'Fgap_topk': res['Fgap_topk'], 'n_nodes': res['graph'].number_of_nodes(), 'n_edges': res['graph'].number_of_edges()})
    return pd.DataFrame(rows)

def node_metric_dataframe(result, literature_node_df=None):
    G = result['graph']
    df = pd.DataFrame({'node': list(G.nodes), 'name': [G.nodes[n].get('name', '') for n in G.nodes], 'lat': [G.nodes[n].get('lat', np.nan) for n in G.nodes], 'lon': [G.nodes[n].get('lon', np.nan) for n in G.nodes], 'EFW_V': [float(result['EFW_V'].get(n, np.nan)) for n in G.nodes], 'BFrag_V': [float(result['BFrag_V'].get(n, np.nan)) for n in G.nodes], 'NBFrag_V': [float(result['NBFrag_V'].get(n, np.nan)) for n in G.nodes]})
    if literature_node_df is not None and (not literature_node_df.empty):
        df = df.merge(literature_node_df, on='node', how='left')
    return df

def edge_metric_dataframe(result, literature_edge_df=None):
    G = result['graph']
    rows = []
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, k, data in G.edges(keys=True, data=True):
            rows.append({
                'edge': (u, v, k),
                'u': u,
                'v': v,
                'key': k,
                'line': data.get('route_short_name', ''),
                'direction': data.get('trip_headsign', ''),
                'u_name': G.nodes[u].get('name', ''),
                'v_name': G.nodes[v].get('name', ''),
                'freq': float(data.get('freq', np.nan)),
                'tau': float(data.get('tau', np.nan)),
                'SAC': float(result['SAC'].get((u, v, k), 0.0)),
                'DElas': float(result['DElas'].get((u, v, k), 0.0)),
            })
    else:
        for u, v, data in G.edges(data=True):
            rows.append({'edge': (u, v), 'u': u, 'v': v, 'u_name': G.nodes[u].get('name', ''), 'v_name': G.nodes[v].get('name', ''), 'freq': float(data.get('freq', np.nan)), 'tau': float(data.get('tau', np.nan)), 'SAC': float(result['SAC'].get((u, v), 0.0)), 'DElas': float(result['DElas'].get((u, v), 0.0))})
    df = pd.DataFrame(rows)
    if literature_edge_df is not None and (not literature_edge_df.empty):
        df = df.merge(literature_edge_df, on='edge', how='left')
    return df

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def graph_center(G):
    lats = [G.nodes[n].get('lat', np.nan) for n in G.nodes]
    lons = [G.nodes[n].get('lon', np.nan) for n in G.nodes]
    return (float(np.nanmedian(lats)), float(np.nanmedian(lons)))

def plot_osm_base(G, radius_m=9000):
    if not OSMNX_AVAILABLE:
        raise RuntimeError('OSMnx not available.')
    center = graph_center(G)
    G_osm = ox.graph_from_point(center, dist=radius_m, network_type='drive', simplify=True)
    fig, ax = ox.plot_graph(G_osm, node_size=0, edge_linewidth=0.5, show=False, close=False)
    return (fig, ax)

def make_parallel_offset_segments(G, edge_values, offset_scale=8e-05):
    groups = defaultdict(list)
    for (u, v, k), val in edge_values.items():
        groups[u, v].append((k, val))
    segs = []
    vals = []
    for (u, v), items in groups.items():
        x1, y1 = (G.nodes[u]['lon'], G.nodes[u]['lat'])
        x2, y2 = (G.nodes[v]['lon'], G.nodes[v]['lat'])
        dx = x2 - x1
        dy = y2 - y1
        norm = math.hypot(dx, dy)
        if norm == 0:
            nxp, nyp = (0.0, 0.0)
        else:
            nxp, nyp = (-dy / norm, dx / norm)
        items = sorted(items, key=lambda x: str(x[0]))
        for idx, (k, val) in enumerate(items):
            shift = (idx - (len(items) - 1) / 2.0) * offset_scale
            sx1, sy1 = (x1 + shift * nxp, y1 + shift * nyp)
            sx2, sy2 = (x2 + shift * nxp, y2 + shift * nyp)
            segs.append([(sx1, sy1), (sx2, sy2)])
            vals.append(float(val))
    return (segs, vals)

def plot_network_overview(G, out_path, title, multigraph=False, radius_m=9000, edge_color='#1f77b4'):
    if not OSMNX_AVAILABLE: #Without OSMNX you cannot visualize the fancy maps!
        return None
    fig, ax = plot_osm_base(G, radius_m=radius_m)
    if multigraph and isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        edge_values = {(u, v, k): 1.0 for u, v, k in G.edges(keys=True)}
        segs, _ = make_parallel_offset_segments(G, edge_values, offset_scale=5e-05)
        lc = LineCollection(segs, colors=edge_color, linewidths=0.8, alpha=0.35)
        ax.add_collection(lc)
    else:
        segs = []
        for u, v in G.edges():
            segs.append([(G.nodes[u]['lon'], G.nodes[u]['lat']), (G.nodes[v]['lon'], G.nodes[v]['lat'])])
        lc = LineCollection(segs, colors=edge_color, linewidths=0.9, alpha=0.45)
        ax.add_collection(lc)
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def _prepare_values(values, log_scale=False):
    vals = np.array([float(v) for v in values], dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    if log_scale:
        vals = np.log10(np.clip(vals, 1e-12, None))
    return vals

def plot_node_metric_map(G, node_metric, out_path, title, radius_m=9000, log_scale=False, size=18):
    if not OSMNX_AVAILABLE:  #Without OSMNX you cannot visualize the fancy maps!
        return None
    fig, ax = plot_osm_base(G, radius_m=radius_m)
    xs, ys, vals = ([], [], [])
    for n in G.nodes:
        lon = G.nodes[n].get('lon', np.nan)
        lat = G.nodes[n].get('lat', np.nan)
        if not np.isfinite(lon) or not np.isfinite(lat):
            continue
        xs.append(lon)
        ys.append(lat)
        vals.append(float(node_metric.get(n, 0.0)))
    vals = _prepare_values(vals, log_scale=log_scale)
    finite = np.isfinite(vals)
    if not finite.any():
        return None
    vmin = np.nanpercentile(vals[finite], 5) if finite.sum() > 10 else np.nanmin(vals[finite])
    vmax = np.nanpercentile(vals[finite], 95) if finite.sum() > 10 else np.nanmax(vals[finite])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(xs, ys, c=vals, s=size, cmap='viridis', norm=norm, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('log10(metric)' if log_scale else 'metric')
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_edge_metric_map(G, edge_metric, out_path, title, radius_m=9000, log_scale=False, linewidth=2.0):
    if not OSMNX_AVAILABLE:  #Without OSMNX you cannot visualize the fancy maps!
        return None
    fig, ax = plot_osm_base(G, radius_m=radius_m)
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        segs, vals = make_parallel_offset_segments(G, edge_metric)
    else:
        segs = []
        vals = []
        for (u, v), val in edge_metric.items():
            if u not in G.nodes or v not in G.nodes:
                continue
            segs.append([(G.nodes[u]['lon'], G.nodes[u]['lat']), (G.nodes[v]['lon'], G.nodes[v]['lat'])])
            vals.append(float(val))
    vals = _prepare_values(vals, log_scale=log_scale)
    finite = np.isfinite(vals)
    if not finite.any() or len(segs) == 0:
        return None
    vmin = np.nanpercentile(vals[finite], 5) if finite.sum() > 10 else np.nanmin(vals[finite])
    vmax = np.nanpercentile(vals[finite], 95) if finite.sum() > 10 else np.nanmax(vals[finite])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs, array=vals, cmap='viridis', norm=norm, linewidths=linewidth, alpha=0.92)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('log10(metric)' if log_scale else 'metric')
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_topk_overlay(G, topk_edges, out_path, title, radius_m=9000, multigraph=False):
    if not OSMNX_AVAILABLE:  #Without OSMNX you cannot visualize the fancy maps!
        return None
    fig, ax = plot_osm_base(G, radius_m=radius_m)
    if multigraph:
        edge_values = {}
        for rank, (edge, gain) in enumerate(topk_edges, start=1):
            edge_values[edge] = rank
        segs, vals = make_parallel_offset_segments(G, edge_values)
        lc = LineCollection(segs, colors='#d62728', linewidths=2.3, alpha=0.95)
        ax.add_collection(lc)
    else:
        segs = []
        for edge, gain in topk_edges:
            if len(edge) == 2:
                u, v = edge
            else:
                u, v = edge[:2]
            segs.append([(G.nodes[u]['lon'], G.nodes[u]['lat']), (G.nodes[v]['lon'], G.nodes[v]['lat'])])
        lc = LineCollection(segs, colors='#d62728', linewidths=2.3, alpha=0.95)
        ax.add_collection(lc)
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_top_lines(G, line_df, out_path, title, radius_m=9000):
    if not OSMNX_AVAILABLE or line_df.empty:  #Without OSMNX you cannot visualize the fancy maps!
        return None
    fig, ax = plot_osm_base(G, radius_m=radius_m)
    colours = plt.get_cmap('tab10')
    handles = []
    for rank, line_id in enumerate(line_df.head(10)['line_id']):
        segments = []
        for u, v, _, data in G.edges(keys=True, data=True):
            if line_id_from_service_meta(data) != line_id:
                continue
            segments.append([
                (G.nodes[u]['lon'], G.nodes[u]['lat']),
                (G.nodes[v]['lon'], G.nodes[v]['lat']),
            ])
        colour = colours(rank)
        if segments:
            ax.add_collection(LineCollection(segments, colors=[colour], linewidths=2.0, alpha=0.9))
            handles.append(Line2D([0], [0], color=colour, linewidth=2.0, label=str(line_id)))
    ax.set_title(title)
    if handles:
        ax.legend(handles=handles, loc='upper left', fontsize=7, frameon=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def save_excel_workbook(workbook_path, sheets):
    workbook_path = Path(workbook_path)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            name = str(sheet_name)[:31]
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=name, index=False)
            else:
                pd.DataFrame(df).to_excel(writer, sheet_name=name, index=False)
    return workbook_path

def save_correlation_heatmap(df_corr, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(df_corr.values, cmap='viridis')
    ax.set_xticks(range(len(df_corr.columns)))
    ax.set_xticklabels(df_corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(df_corr.index)))
    ax.set_yticklabels(df_corr.index)
    for i in range(df_corr.shape[0]):
        for j in range(df_corr.shape[1]):
            ax.text(j, i, f'{df_corr.iloc[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
    ax.set_title(title)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def save_scatter(df, x, y, out_path, title, annotate_top=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    valid = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(valid[x], valid[y], alpha=0.7)
    if len(valid) >= 2:
        slope, intercept = np.polyfit(valid[x], valid[y], 1)
        x_values = np.linspace(valid[x].min(), valid[x].max(), 100)
        ax.plot(x_values, slope * x_values + intercept, color='#d62728', linewidth=1.2)
        r_squared = valid[x].corr(valid[y]) ** 2
        ax.text(0.04, 0.94, f'$R^2={r_squared:.4f}$', transform=ax.transAxes, va='top')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    if annotate_top and 'line_id' in df.columns:
        top = df.nlargest(min(10, len(df)), y)
        for _, row in top.iterrows():
            ax.annotate(str(row['line_id']), (row[x], row[y]), fontsize=7)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def build_case_study_graphs(
    gtfs,
    date_obj,
    beta=0.0,
    service_key_mode='route_direction',
    add_walking=False,
    walk_radius_m=WALK_RADIUS_M,
    walk_speed_kmh=WALK_SPEED_KMH,
    walk_max_neighbors=WALK_MAX_NEIGHBORS_PER_STOP,
):
    Gm = build_bus_multidigraph(gtfs, date_obj=date_obj, service_key_mode=service_key_mode)
    Gm = compute_edge_times_and_freqs_from_gtfs(
        gtfs,
        Gm,
        TIME_WINDOW[0],
        TIME_WINDOW[1],
        date_obj=date_obj,
        service_key_mode=service_key_mode,
    )
    Gagg, groups = build_aggregated_digraph_with_wait(Gm, beta=beta)
    Gservice = build_service_routing_multidigraph(Gm, beta=beta)
    service_meta = build_service_metadata(Gm)
    walking_connectors = pd.DataFrame()
    if add_walking:
        walking_connectors = add_walking_connectors(Gagg, Gservice, radius_m=walk_radius_m, speed_kmh=walk_speed_kmh, max_neighbors=walk_max_neighbors)
    return {'Gm': Gm, 'Gagg': Gagg, 'Gservice': Gservice, 'groups': groups, 'service_meta': service_meta, 'walking_connectors': walking_connectors}

def log(message):
    timestamp = time.strftime('%H:%M:%S')
    print(f'[{timestamp}] {message}', flush=True)

def progress_iter(iterable, desc='Progress', total=None, disable=None, **kwargs):
    if disable is None:
        disable = not SHOW_PROGRESS_BARS
    if tqdm is None or disable:
        return iterable
    return tqdm(iterable, desc=desc, total=total, mininterval=0.5, dynamic_ncols=True, leave=False, **kwargs)

@contextmanager
def pipeline_section(name):
    indent = '  ' * len(TIMING_NESTING)
    log(f'{indent}START | {name}')
    t0 = time.perf_counter()
    TIMING_NESTING.append(name)
    try:
        yield
    finally:
        TIMING_NESTING.pop()
        dt_sec = time.perf_counter() - t0
        TIMING_SUMMARY_ROWS.append({'step': name, 'seconds': dt_sec})
        indent = '  ' * len(TIMING_NESTING)
        log(f'{indent}DONE  | {name} | {dt_sec:,.2f} s')

@contextmanager
def algorithm_timer(graph, model_name, weight_mode, phase_order, algorithm, complexity, enabled=True):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - t0
        if enabled:
            ALGORITHM_TIMING_ROWS.append({'model': model_name, 'weight_mode': weight_mode, 'n_nodes': int(graph.number_of_nodes()), 'n_edges': int(graph.number_of_edges()), 'phase_order': int(phase_order), 'algorithm': algorithm, 'theoretical_complexity': complexity, 'seconds': float(seconds)})

def save_algorithm_timing_report(output_root, weight_mode='freq'):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    columns = ['model', 'weight_mode', 'n_nodes', 'n_edges', 'phase_order', 'algorithm', 'theoretical_complexity', 'seconds']
    raw = pd.DataFrame(ALGORITHM_TIMING_ROWS, columns=columns)
    raw_path = output_root / 'algorithm_timing_all_runs.csv'
    raw.to_csv(raw_path, index=False)
    paper = raw[raw['weight_mode'] == weight_mode].copy()
    if not paper.empty:
        paper = paper.sort_values(['model', 'phase_order', 'algorithm']).reset_index(drop=True)
        totals = paper.groupby('model')['seconds'].transform('sum')
        paper['share_of_model_timed_total_pct'] = np.where(totals > 0, 100.0 * paper['seconds'] / totals, np.nan)
    paper_path = output_root / 'algorithm_timing_paper.csv'
    paper.to_csv(paper_path, index=False)
    if paper.empty:
        comparison = pd.DataFrame(columns=['phase_order', 'algorithm', 'theoretical_complexity', 'aggregated_seconds', 'service_seconds', 'service_over_aggregated'])
    else:
        id_cols = ['phase_order', 'algorithm', 'theoretical_complexity']
        comparison = paper.pivot_table(index=id_cols, columns='model', values='seconds', aggfunc='first').reset_index()
        comparison.columns.name = None
        comparison = comparison.rename(columns={'aggregated': 'aggregated_seconds', 'service': 'service_seconds'})
        for col in ['aggregated_seconds', 'service_seconds']:
            if col not in comparison:
                comparison[col] = np.nan
        comparison['service_over_aggregated'] = np.where(comparison['aggregated_seconds'] > 0, comparison['service_seconds'] / comparison['aggregated_seconds'], np.nan)
        comparison = comparison.sort_values(['phase_order', 'algorithm']).reset_index(drop=True)
    comparison_path = output_root / 'algorithm_timing_service_vs_aggregated.csv'
    comparison.to_csv(comparison_path, index=False)
    return (raw, paper, comparison)

def save_runtime_environment_report(output_root):
    import platform
    import sys
    row = {'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'python_version': sys.version.split()[0], 'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(), 'networkx_version': nx.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__}
    try:
        # Just to see also the memory, but not sure if it will be needed (better safe than sorry atm)
        import psutil
        row['physical_cpu_count'] = psutil.cpu_count(logical=False)
        row['total_memory_gb'] = psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        row['physical_cpu_count'] = np.nan
        row['total_memory_gb'] = np.nan
    df = pd.DataFrame([row])
    path = Path(output_root) / 'runtime_environment.csv'
    df.to_csv(path, index=False)
    return df


# %% Paper outputs
def run_proposed_measures(graph_bundle, collect_path_statistics=True, record_timings=True):
    results = {}
    for model_name, graph in (
        ('service', graph_bundle['Gservice']),
        ('aggregated', graph_bundle['Gagg']),
    ):
        service_meta = graph_bundle['service_meta'] if model_name == 'service' else None
        results[model_name] = compute_model_result(
            graph,
            model_name,
            TOP_K,
            service_meta=service_meta,
            record_algorithm_timings=record_timings,
            collect_path_statistics=collect_path_statistics,
        )
    return results


def save_path_statistics(results, output_dir):
    output_dir = ensure_dir(output_dir)
    table_names = (
        'summary',
        'transfer_distribution',
        'segment_distribution',
        'walking_distribution',
        'extreme_examples',
    )
    saved = {}
    for table_name in table_names:
        frames = [
            result['path_statistics'][table_name]
            for result in results.values()
            if result.get('path_statistics') is not None
            and not result['path_statistics'][table_name].empty
        ]
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        table.to_csv(output_dir / f'canonical_paths_{table_name}.csv', index=False)
        saved[table_name] = table
    return saved


def top_node_rankings(node_df, metrics, top_n=10):
    rows = []
    for metric in metrics:
        if metric not in node_df.columns:
            continue
        ranked = node_df.dropna(subset=[metric]).nlargest(top_n, metric)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            rows.append({
                'metric': metric,
                'rank': rank,
                'node': row.node,
                'name': row.name,
                'value': getattr(row, metric),
            })
    return pd.DataFrame(rows)


def top_edge_rankings(edge_df, metrics, top_n=10):
    rows = []
    for metric in metrics:
        if metric not in edge_df.columns:
            continue
        ranked = edge_df.dropna(subset=[metric]).nlargest(top_n, metric)
        for rank, row in enumerate(ranked.to_dict('records'), start=1):
            rows.append({
                'metric': metric,
                'rank': rank,
                'edge': row['edge'],
                'line': row.get('line', ''),
                'direction': row.get('direction', ''),
                'u_name': row['u_name'],
                'v_name': row['v_name'],
                'value': row[metric],
            })
    return pd.DataFrame(rows)


def greedy_ranking(result):
    G = result['graph']
    rows = []
    for rank, (edge, gain) in enumerate(result['topk_edges'], start=1):
        u, v = edge[:2]
        rows.append({
            'rank': rank,
            'edge': edge,
            'u_name': G.nodes[u].get('name', u),
            'v_name': G.nodes[v].get('name', v),
            'marginal_gain': gain,
            'SAC': result['SAC'].get(edge, np.nan),
            'DElas': result['DElas'].get(edge, np.nan),
        })
    return pd.DataFrame(rows)


def metric_map_correlation(first, second):
    common = sorted(set(first) & set(second), key=str)
    if len(common) < 2:
        return np.nan
    frame = pd.DataFrame({
        'first': [first[key] for key in common],
        'second': [second[key] for key in common],
    })
    return frame.corr(method='spearman').iloc[0, 1]


def save_paper_figures(results, tables, output_dir):
    output_dir = ensure_dir(output_dir)
    if not OSMNX_AVAILABLE:
        log('OSMnx is not available; map generation was skipped.')
        return

    service = results['service']
    service_nodes = tables['service']['nodes']
    service_edges = tables['service']['edges']
    G = service['graph']

    plot_network_overview(G, output_dir / 'network_overview.png', 'Munich bus network', multigraph=True, edge_color='#d62728')
    plot_node_metric_map(G, service['BFrag_V'], output_dir / 'service_BFragV.png', 'Service-level BFragV')
    plot_node_metric_map(G, service['NBFrag_V'], output_dir / 'service_NBFragV.png', 'Service-level NBFragV')
    plot_edge_metric_map(G, service['DElas'], output_dir / 'service_DElas.png', 'Service-level DElas')
    plot_edge_metric_map(G, service['SAC'], output_dir / 'service_SAC.png', 'Service-level SAC')

    for metric in ('BetV', 'CV', 'CF', 'IMV'):
        if metric in service_nodes:
            values = dict(zip(service_nodes['node'], service_nodes[metric]))
            plot_node_metric_map(G, values, output_dir / f'service_{metric}.png', f'Service-level {metric}')
    for metric in ('BetE', 'CE'):
        if metric in service_edges:
            values = dict(zip(service_edges['edge'], service_edges[metric]))
            plot_edge_metric_map(G, values, output_dir / f'service_{metric}.png', f'Service-level {metric}')

    plot_topk_overlay(
        service['graph'],
        service['topk_edges'],
        output_dir / 'service_top10_Fgap.png',
        'Service-level top-10 Fgap edges',
        multigraph=True,
    )
    plot_topk_overlay(
        results['aggregated']['graph'],
        results['aggregated']['topk_edges'],
        output_dir / 'aggregated_top10_Fgap.png',
        'Aggregated top-10 Fgap edges',
        multigraph=False,
    )
    plot_top_lines(
        service['graph'],
        tables['service']['lines'],
        output_dir / 'service_top10_lines.png',
        'Top-10 lines by Fgap',
    )


def export_baseline_results(results, graph_bundle, output_dir):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / 'tables')
    figures_dir = ensure_dir(output_dir / 'figures')

    if RUN_LITERATURE_MEASURES:
        with pipeline_section('literature node measures'):
            shared_node_literature = compute_literature_node_metrics(graph_bundle['Gservice'])
        literature = {}
        for model_name, graph in (
            ('service', graph_bundle['Gservice']),
            ('aggregated', graph_bundle['Gagg']),
        ):
            with pipeline_section(f'{model_name} literature edge measures'):
                edge_literature = compute_literature_edge_metrics(graph)
            literature[model_name] = {
                'nodes': shared_node_literature,
                'edges': edge_literature,
            }
    else:
        literature = {
            model_name: {
                'nodes': pd.DataFrame(columns=['node']),
                'edges': pd.DataFrame(columns=['edge']),
            }
            for model_name in results
        }

    tables = {}
    workbook_sheets = {}
    node_metrics = ['BFrag_V', 'NBFrag_V', 'BetV', 'CV', 'CF', 'IMV']
    edge_metrics = ['DElas', 'SAC', 'BetE', 'CE']

    for model_name, result in results.items():
        node_df = node_metric_dataframe(result, literature[model_name]['nodes'])
        edge_df = edge_metric_dataframe(result, literature[model_name]['edges'])
        line_df = result['line_df'].copy() if result['line_df'] is not None else pd.DataFrame()
        tables[model_name] = {'nodes': node_df, 'edges': edge_df, 'lines': line_df}

        node_df.to_csv(tables_dir / f'{model_name}_node_metrics.csv', index=False)
        edge_df.to_csv(tables_dir / f'{model_name}_edge_metrics.csv', index=False)
        if not line_df.empty:
            line_df.to_csv(tables_dir / f'{model_name}_line_metrics.csv', index=False)

        node_top10 = top_node_rankings(node_df, node_metrics)
        edge_top10 = top_edge_rankings(edge_df, edge_metrics)
        fgap_top10 = greedy_ranking(result)
        node_top10.to_csv(tables_dir / f'{model_name}_top10_nodes.csv', index=False)
        edge_top10.to_csv(tables_dir / f'{model_name}_top10_edges.csv', index=False)
        fgap_top10.to_csv(tables_dir / f'{model_name}_top10_Fgap.csv', index=False)

        node_columns = [column for column in node_metrics if column in node_df]
        edge_columns = [column for column in edge_metrics if column in edge_df]
        node_correlation = node_df[node_columns].corr(method='spearman')
        edge_correlation = edge_df[edge_columns].corr(method='spearman')
        node_correlation.to_csv(tables_dir / f'{model_name}_node_spearman.csv')
        edge_correlation.to_csv(tables_dir / f'{model_name}_edge_spearman.csv')
        save_correlation_heatmap(node_correlation, figures_dir / f'{model_name}_node_spearman.png', f'{model_name.capitalize()} node measures')
        save_correlation_heatmap(edge_correlation, figures_dir / f'{model_name}_edge_spearman.png', f'{model_name.capitalize()} edge measures')

        workbook_sheets[f'{model_name}_nodes'] = node_df
        workbook_sheets[f'{model_name}_edges'] = edge_df
        if not line_df.empty:
            workbook_sheets[f'{model_name}_lines'] = line_df
        workbook_sheets[f'{model_name}_node_top10'] = node_top10
        workbook_sheets[f'{model_name}_edge_top10'] = edge_top10
        workbook_sheets[f'{model_name}_Fgap_top10'] = fgap_top10
        workbook_sheets[f'{model_name}_node_corr'] = node_correlation.reset_index()
        workbook_sheets[f'{model_name}_edge_corr'] = edge_correlation.reset_index()

    line_df = tables['service']['lines']
    line_df.head(10).to_csv(tables_dir / 'service_top10_lines.csv', index=False)
    line_relationships = pd.DataFrame([
        {
            'comparison': 'Fgap versus sum of SAC',
            'pearson': line_df['Fgap_line_exact'].corr(line_df['SAC_sum']),
            'spearman': line_df['Fgap_line_exact'].corr(line_df['SAC_sum'], method='spearman'),
        },
        {
            'comparison': 'Fgap versus sum of DElas',
            'pearson': line_df['Fgap_line_exact'].corr(line_df['DElas_sum']),
            'spearman': line_df['Fgap_line_exact'].corr(line_df['DElas_sum'], method='spearman'),
        },
    ])
    line_relationships['r_squared'] = line_relationships['pearson'] ** 2
    line_relationships.to_csv(tables_dir / 'line_relationships.csv', index=False)
    save_scatter(
        line_df,
        'SAC_sum',
        'Fgap_line_exact',
        figures_dir / 'line_Fgap_vs_SAC.png',
        'Line Fgap versus sum of SAC',
        annotate_top=True,
    )
    save_scatter(
        line_df,
        'DElas_sum',
        'Fgap_line_exact',
        figures_dir / 'line_Fgap_vs_DElas.png',
        'Line Fgap versus sum of DElas',
        annotate_top=True,
    )

    path_tables = save_path_statistics(results, tables_dir)
    for name, table in path_tables.items():
        workbook_sheets[f'paths_{name}'[:31]] = table

    summary = metric_summary_table(results)
    summary.to_csv(tables_dir / 'model_summary.csv', index=False)
    workbook_sheets['model_summary'] = summary
    workbook_sheets['line_relationships'] = line_relationships
    save_excel_workbook(output_dir / 'mvg_fragility_results.xlsx', workbook_sheets)

    if RUN_MAPS:
        save_paper_figures(results, tables, figures_dir)
    return tables


def run_walking_sensitivity(gtfs, analysis_date, baseline_results, output_dir):
    output_dir = ensure_dir(output_dir)
    bundle = build_case_study_graphs(
        gtfs,
        date_obj=analysis_date,
        beta=WAIT_BETA_MIN,
        service_key_mode=SERVICE_KEY_MODE,
        add_walking=True,
        walk_radius_m=WALK_RADIUS_M,
        walk_speed_kmh=WALK_SPEED_KMH,
        walk_max_neighbors=WALK_MAX_NEIGHBORS_PER_STOP,
    )
    bundle['walking_connectors'].to_csv(output_dir / 'walking_connectors.csv', index=False)
    results = run_proposed_measures(bundle, collect_path_statistics=True, record_timings=False)
    path_tables = save_path_statistics(results, output_dir)

    rows = []
    for model_name in ('service', 'aggregated'):
        baseline = baseline_results[model_name]
        walking = results[model_name]
        row = {
            'model': model_name,
            'BFrag_G_baseline': baseline['BFrag_G'],
            'BFrag_G_walking': walking['BFrag_G'],
            'NBFrag_G_baseline': baseline['NBFrag_G'],
            'NBFrag_G_walking': walking['NBFrag_G'],
            'BFrag_V_spearman': metric_map_correlation(baseline['BFrag_V'], walking['BFrag_V']),
            'NBFrag_V_spearman': metric_map_correlation(baseline['NBFrag_V'], walking['NBFrag_V']),
            'SAC_spearman': metric_map_correlation(baseline['SAC'], walking['SAC']),
            'DElas_spearman': metric_map_correlation(baseline['DElas'], walking['DElas']),
        }
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / 'walking_sensitivity_comparison.csv', index=False)
    return results, comparison, path_tables


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    gtfs = load_gtfs(GTFS_SOURCE)
    analysis_date = (
        dt.date.fromisoformat(ANALYSIS_DATE)
        if ANALYSIS_DATE
        else choose_default_analysis_date(gtfs)
    )
    if analysis_date is None:
        raise ValueError('A weekday analysis date could not be determined from the GTFS archive.')
    log(f'Weekday analysis date: {analysis_date.isoformat()}')

    baseline_bundle = build_case_study_graphs(
        gtfs,
        date_obj=analysis_date,
        beta=WAIT_BETA_MIN,
        service_key_mode=SERVICE_KEY_MODE,
        add_walking=False,
    )
    baseline_results = run_proposed_measures(
        baseline_bundle,
        collect_path_statistics=True,
        record_timings=RECORD_ALGORITHM_TIMINGS,
    )
    export_baseline_results(baseline_results, baseline_bundle, OUTPUT_ROOT)

    configuration = pd.DataFrame([{
        'analysis_date': analysis_date.isoformat(),
        'time_window_start': TIME_WINDOW[0],
        'time_window_end': TIME_WINDOW[1],
        'transfer_penalty_minutes': WAIT_BETA_MIN,
        'walking_connectors_in_baseline': False,
        'walking_radius_m': WALK_RADIUS_M,
        'walking_speed_kmh': WALK_SPEED_KMH,
        'service_nodes': baseline_bundle['Gservice'].number_of_nodes(),
        'service_edges': baseline_bundle['Gservice'].number_of_edges(),
        'aggregated_nodes': baseline_bundle['Gagg'].number_of_nodes(),
        'aggregated_edges': baseline_bundle['Gagg'].number_of_edges(),
    }])
    configuration.to_csv(OUTPUT_ROOT / 'run_configuration.csv', index=False)

    if RECORD_ALGORITHM_TIMINGS:
        save_algorithm_timing_report(OUTPUT_ROOT)
        save_runtime_environment_report(OUTPUT_ROOT)

    if RUN_WALKING_SENSITIVITY:
        run_walking_sensitivity(
            gtfs,
            analysis_date,
            baseline_results,
            OUTPUT_ROOT / 'walking_sensitivity',
        )

    log(f'Results saved under: {OUTPUT_ROOT.resolve()}')


if __name__ == '__main__':
    main()
