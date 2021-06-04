import time
import pandas as pd
import copy
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
import geopandas as gpd

def cal_arc(p1, p2, degree=False):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    arc = np.pi - np.arctan2(dy, dx)
    return arc / np.pi * 180 if degree else arc



def cal_euclidean(p1, p2):
    return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])

def get_shape_mbr(df_shape):
    oid = 'OID' if 'FID' in df_shape.columns else 'OBJECTID'
    df_mbr = copy.deepcopy(df_shape[[oid, 'geometry']])
    df_mbr.reset_index(drop=True, inplace=True)
    df_mbr['geometry'] = pd.Series([geo.minimum_rotated_rectangle for geo in df_mbr['geometry']])
    df_mbr['xy'] = pd.Series([list(geo.exterior.coords) for geo in df_mbr['geometry']])
    #
    df_mbr['x0'] = pd.Series([xy[0][0] for xy in df_mbr['xy']])
    df_mbr['x1'] = pd.Series([xy[1][0] for xy in df_mbr['xy']])
    df_mbr['x2'] = pd.Series([xy[2][0] for xy in df_mbr['xy']])

    df_mbr['y0'] = pd.Series([xy[0][1] for xy in df_mbr['xy']])
    df_mbr['y1'] = pd.Series([xy[1][1] for xy in df_mbr['xy']])
    df_mbr['y2'] = pd.Series([xy[2][1] for xy in df_mbr['xy']])
    #
    df_mbr['l1'] = pd.Series(
        [cal_euclidean([x0, y0], [x1, y1]) for x0, y0, x1, y1 in df_mbr[['x0', 'y0', 'x1', 'y1']].values])
    df_mbr['l2'] = pd.Series(
        [cal_euclidean([x0, y0], [x1, y1]) for x0, y0, x1, y1 in df_mbr[['x1', 'y1', 'x2', 'y2']].values])

    df_mbr['a1'] = pd.Series(
        [cal_arc([x0, y0], [x1, y1], True) for x0, y0, x1, y1 in df_mbr[['x0', 'y0', 'x1', 'y1']].values])
    df_mbr['a2'] = pd.Series(
        [cal_arc([x0, y0], [x1, y1], True) for x0, y0, x1, y1 in df_mbr[['x1', 'y1', 'x2', 'y2']].values])
    #
    df_mbr['longer'] = df_mbr['l1'] >= df_mbr['l2']
    #
    df_mbr['lon_len'] = pd.Series([l1 if longer else l2 for l1, l2, longer in df_mbr[['l1', 'l2', 'longer']].values])
    df_mbr['short_len'] = pd.Series([l2 if longer else l1 for l1, l2, longer in df_mbr[['l1', 'l2', 'longer']].values])
    df_mbr['lon_arc'] = pd.Series([a1 if longer else a2 for a1, a2, longer in df_mbr[['a1', 'a2', 'longer']].values])
    df_mbr['short_arc'] = pd.Series([a2 if longer else a1 for a1, a2, longer in df_mbr[['a1', 'a2', 'longer']].values])

    df_mbr.drop(['x0', 'x1', 'x2', 'y0', 'y1', 'y2', 'l1', 'l2', 'a1', 'a2'], axis=1, inplace=True)
    #
    df_shape = pd.merge(df_shape, df_mbr[[oid, 'lon_len', 'short_len', 'lon_arc', 'short_arc']], how='left', on=oid)
    return df_mbr, df_shape

def get_shape_normalize_final(df_use, if_scale_y):
    df_use = copy.deepcopy(df_use)
    #
    df_use['mu_x'] = pd.Series([geo.centroid.x for geo in df_use['geometry']])
    df_use['mu_y'] = pd.Series([geo.centroid.y for geo in df_use['geometry']])
    df_use['geometry'] = pd.Series(
        [affinity.translate(geo, -mx, -my) for mx, my, geo in df_use[['mu_x', 'mu_y', 'geometry']].values])
    df_use['x_max'] = pd.Series([max(geo.exterior.xy[0]) for geo in df_use['geometry']])
    df_use['x_min'] = pd.Series([min(geo.exterior.xy[0]) for geo in df_use['geometry']])
    df_use['scale_x'] = (df_use['x_max'] - df_use['x_min'])

    df_use['y_max'] = pd.Series([max(geo.exterior.xy[1]) for geo in df_use['geometry']])
    df_use['y_min'] = pd.Series([min(geo.exterior.xy[1]) for geo in df_use['geometry']])
    df_use['scale_y'] = (df_use['y_max'] - df_use['y_min'])

    # -1~1
    if if_scale_y:
        df_use['geometry'] = pd.Series(
            [affinity.scale(geo, 1 / del_x, 1 / del_y, origin='centroid') for del_x, del_y, geo in
             df_use[['scale_x', 'scale_y', 'geometry']].values])
    else:
        df_use['geometry'] = pd.Series([affinity.scale(geo, 1 / del_x, 1 / del_x, origin='centroid') for del_x, geo in
                                        df_use[['scale_x', 'geometry']].values])
    df_use.drop(['mu_x', 'mu_y', 'scale_x', 'scale_y', 'x_max', 'x_min', 'y_max', 'y_min'], axis=1, inplace=True)
    #helper_print_with_time('get_shape_normalize_2:', ','.join(df_use.columns), sep='')
    return df_use

def simplify_cos_on_node(df_node, tor_cos):
    oid = 'OBJECTID'
    df_line = copy.deepcopy(df_node)
    #
    df_line = df_line[df_line['PID'] != 0].reset_index(drop=True)
    df_line['PID'] = df_line['PID'] - 1
    #
    coor_dic = {(int(oid), int(pid)): [x, y] for oid, pid, x, y in df_line[['OBJECTID', 'PID', 'x', 'y']].values}
    df_line['x_l'] = pd.Series([coor_dic[(oid, (pid - 1 if pid >= 1 else pnum - 2))][0] for oid, pid, pnum in
                                df_line[['OBJECTID', 'PID', 'p_num']].values])
    df_line['y_l'] = pd.Series([coor_dic[(oid, (pid - 1 if pid >= 1 else pnum - 2))][1] for oid, pid, pnum in
                                df_line[['OBJECTID', 'PID', 'p_num']].values])
    df_line['x_r'] = pd.Series([coor_dic[(oid, (pid + 1 if pid < (pnum - 2) else 0))][0] for oid, pid, pnum in
                                df_line[['OBJECTID', 'PID', 'p_num']].values])
    df_line['y_r'] = pd.Series([coor_dic[(oid, (pid + 1 if pid < (pnum - 2) else 0))][1] for oid, pid, pnum in
                                df_line[['OBJECTID', 'PID', 'p_num']].values])
    #
    df_line['dx_l'] = pd.Series([x - xl for x, xl in df_line[['x', 'x_l']].values])
    df_line['dy_l'] = pd.Series([y - yl for y, yl in df_line[['y', 'y_l']].values])
    df_line['dx_r'] = pd.Series([xr - x for x, xr in df_line[['x', 'x_r']].values])
    df_line['dy_r'] = pd.Series([yr - y for y, yr in df_line[['y', 'y_r']].values])
    df_line['cos'] = pd.Series(
        [(dxl * dxr + dyl * dyr) / (np.sqrt(dxl * dxl + dyl * dyl) * np.sqrt(dxr * dxr + dyr * dyr)) for
         dxl, dyl, dxr, dyr in df_line[['dx_l', 'dy_l', 'dx_r', 'dy_r']].values])
    #
    df_line = df_line[df_line['cos'] <= tor_cos].reset_index(drop=True)
    #
    df_line = reset_node_PID(df_line)
    return df_line

def reset_node_PID(df_node):
    oid = 'OBJECTID'
    df_node.reset_index(inplace=True, drop=False)
    dft = df_node.groupby([oid], as_index=False)['index'].agg({'id_min': 'min'})
    df_node = pd.merge(df_node, dft, how='left', on=oid)
    df_node['PID'] = df_node['index'] - df_node['id_min']
    df_node.drop(['index', 'id_min'], axis=1, inplace=True)
    return df_node
def node_to_polygon(df_node):
    df_node['xy'] = pd.Series([(x, y) for x, y in df_node[['x', 'y']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=True)['xy'].apply(list)
    dft = dft.reset_index(drop=False)
    dft['geometry'] = pd.Series([Polygon(xy) for xy in dft['xy']])
    dft = gpd.GeoDataFrame(dft)
    #helper_print_with_time('node_to_polygon')
    return dft

def get_shape_simplify(df_rotate, tor_dist, tor_cos, simplify_type=1):
    df_roate = copy.deepcopy(df_rotate)
    # 1.
    # df_rotate=get_shape_features(df_rotate)
    # 2.
    df_node = get_node_features(df_rotate)
    # 3.
    df_node = simplify_cos_on_node(df_node, tor_cos)
    # 4.
    df_poly = node_to_polygon(df_node)
    # 5.
    df_poly = pd.merge(df_poly, df_rotate[['OBJECTID', 'lon_len', 'short_len']], how='left', on='OBJECTID')
    df_poly['diag'] = pd.Series([w * h / np.sqrt(w * w + h * h) for w, h in df_poly[['lon_len', 'short_len']].values])
    # 6.dp
    df_poly = simplify_dp_on_shape(df_poly, tor_dist, type=simplify_type)
    df_poly.drop(['xy', 'diag'], axis=1, inplace=True)
    #helper_print_with_time('simplify_shape:', ','.join(df_poly.columns), sep='')
    return df_poly

def simplify_dp_on_shape(df_shape, tor=0.000001, type=1):
    # type=1,relative;type=2,absolute
    if type == 1:
        df_shape['geometry'] = pd.Series(
            [geo.simplify(tor * diag) for geo, diag in df_shape[['geometry', 'diag']].values])
    else:
        df_shape['geometry'] = df_shape['geometry'].simplify(tor)
    #helper_print_with_time('simplify_dp:', ','.join(df_shape.columns), sep='')
    return df_shape

def reset_start_point(df_poly):
    dfq = copy.deepcopy(df_poly)
    dfn = get_node_features(dfq).query('PID!=0')
    dfn['s'] = dfn['x'] + dfn['y']
    dft = dfn.groupby(['OBJECTID'], as_index=False)['s'].agg({'s_min': 'min'})
    dfn = pd.merge(dfn, dft, how='left', on='OBJECTID')
    dfn['ds'] = abs(dfn['s'] - dfn['s_min'])
    dfn['flag'] = dfn['ds'] < 10e-10
    dft = dfn.sort_values(['OBJECTID', 'flag'], ascending=False).groupby(['OBJECTID']).head(1)
    dic_temp = {row['OBJECTID']: row['PID'] for index, row in dft.iterrows()}
    dfn = dfn.set_index('PID')
    dft = [pd.concat([group.loc[dic_temp[index]:], group.loc[:dic_temp[index]]], axis=0)
           for index, group in dfn.groupby('OBJECTID')
           ]
    dfn = pd.concat(dft, axis=0).reset_index(drop=False)
    dfn = node_to_polygon(dfn)
    cols = [x for x in dfq.columns if 'geometry' not in x]
    dfn = pd.merge(dfq[cols], dfn[['OBJECTID', 'geometry']], how='left', on='OBJECTID')
    dfn = gpd.GeoDataFrame(dfn)
    return dfn

def get_node_features(df_shape):
    oid = 'OBJECTID'
    df_shape['xy'] = pd.Series(['|'.join(map(str, geo.exterior.coords)) for geo in df_shape['geometry']])
    df_shape['p_num'] = pd.Series([geo.exterior.coords.__len__() for geo in df_shape['geometry']])
    df_shape['length'] = df_shape['geometry'].exterior.length
    df_node = (df_shape.set_index([oid, 'p_num', 'length'])['xy']
               .str.split('|', expand=True)
               .stack()
               .reset_index(level=3, drop=True)
               .reset_index(name='xy'))

    df_node['x'] = pd.Series([float(xy.split(',')[0][1:]) for xy in df_node['xy'].values])
    df_node['y'] = pd.Series([float(xy.split(',')[1][:-1]) for xy in df_node['xy'].values])
    #
    df_node.reset_index(inplace=True, drop=False)
    dft = df_node.groupby([oid], as_index=False)['index'].agg({'id_min': 'min'})
    df_node = pd.merge(df_node, dft, how='left', on=oid)
    df_node['PID'] = df_node['index'] - df_node['id_min']
    df_node.drop(['xy', 'id_min', 'index'], axis=1, inplace=True)
    return df_node

def get_line_features_final(df_node, POINTS_SHAPE=20):
    #
    df_line = copy.deepcopy(df_node)
    #
    df_line['next_x'] = df_line.groupby(['OBJECTID'], as_index=False)['x'].shift(-1)
    df_line['next_y'] = df_line.groupby(['OBJECTID'], as_index=False)['y'].shift(-1)
    df_line['dx'] = df_line['next_x'] - df_line['x']
    df_line['dy'] = df_line['next_y'] - df_line['y']
    df_line['dl'] = pd.Series([(dx ** 2 + dy ** 2) ** 0.5 for dx, dy in df_line[['dx', 'dy']].values])
    df_line['dr'] = df_line['dl'] / df_line['length']
    #
    df_line = df_line.dropna().reset_index(drop=True)
    df_line['cum_dr'] = df_line.groupby(['OBJECTID'], as_index=False)['dr'].cumsum()
    df_line['last_cum_dr'] = df_line.groupby(['OBJECTID'], as_index=False)['cum_dr'].shift(1).fillna(0)
    df_line['cum_count'] = ((df_line['cum_dr'] - 0.00000000000001) // (1 / POINTS_SHAPE))
    df_line['last_cum'] = df_line.groupby(['OBJECTID'], as_index=False)['cum_count'].shift(1)
    df_line['dc'] = (df_line['cum_count'] - df_line['last_cum']).fillna(df_line['cum_count']).astype('int')
    #
    df_line['_dr'] = df_line['dr'] * POINTS_SHAPE
    df_line['_dx'] = df_line['dx'] / df_line['_dr']
    df_line['_dy'] = df_line['dy'] / df_line['_dr']
    #
    meta_r = 1 / POINTS_SHAPE
    df_line['last_cum'] = df_line['last_cum'].fillna(0)
    df_line['more'] = df_line['last_cum_dr'] - df_line['last_cum'] / POINTS_SHAPE
    df_line['x0'] = df_line['x'] - df_line['dx'] * df_line['more'] / df_line['dr']
    df_line['y0'] = df_line['y'] - df_line['dy'] * df_line['more'] / df_line['dr']
    #
    df_line['xs'] = pd.Series(
        [[x + (i + 1) * _dx for i in range(int(dc))] for x, _dx, dc in df_line[['x0', '_dx', 'dc']].values])
    df_line['ys'] = pd.Series(
        [[y + (i + 1) * _dy for i in range(int(dc))] for y, _dy, dc in df_line[['y0', '_dy', 'dc']].values])
    #
    df_line['xs'] = pd.Series([[x] + xs if pid == 0 else xs for xs, x, pid in df_line[['xs', 'x', 'PID']].values])
    df_line['ys'] = pd.Series([[y] + ys if pid == 0 else ys for ys, y, pid in df_line[['ys', 'y', 'PID']].values])
    df_line['len'] = pd.Series([len(x) for x in df_line['xs']])
    #
    df_line['xs'] = pd.Series(['|'.join(map(str, xs)) for xs in df_line['xs']])
    df_line['ys'] = pd.Series(['|'.join(map(str, ys)) for ys in df_line['ys']])

    df_line.drop(
        ['length', 'p_num', 'next_x', 'next_y', 'dl', 'last_cum_dr', 'cum_count', 'last_cum', '_dr', '_dx', '_dy', 'x0',
         'y0'], axis=1, inplace=True)

    return df_line

def get_inter_features(df_line):
    #
    cols = ['OBJECTID', 'PID', 'xs', 'ys']
    df_detail = df_line[df_line['len'] > 0][cols + ['len']]
    _id = df_detail.loc[:, ['OBJECTID', 'PID']].values.repeat(df_detail['len'], 0)
    _xs = df_detail['xs'].str.split('|', expand=True).stack().values.reshape(-1, 1).astype(float)
    _ys = df_detail['ys'].str.split('|', expand=True).stack().values.reshape(-1, 1).astype(float)
    df_detail = pd.DataFrame(np.hstack((_id, _xs, _ys)), columns=cols)
    df_detail['OBJECTID'] = df_detail['OBJECTID'].astype(int)
    df_detail['PID'] = df_detail['PID'].astype(int)
    del _id, _xs, _ys
    #
    df_detail = df_detail.reset_index(drop=False)
    dft = df_detail.groupby(['OBJECTID', 'PID'], as_index=False)['index'].agg({'mini': 'min'})
    df_detail = pd.merge(df_detail, dft, how='left', on=['OBJECTID', 'PID'])
    df_detail['isBegin'] = (df_detail['index'] == df_detail['mini']) * 1
    #
    dft = df_detail.groupby(['OBJECTID'], as_index=False)['index'].agg({'minimini': 'min'})
    df_detail = pd.merge(df_detail, dft, how='left', on=['OBJECTID'])
    df_detail['UID'] = df_detail['index'] - df_detail['minimini']

    df_detail['OID_UID'] = df_detail['OBJECTID'] * 1000 + df_detail['UID']
    df_detail.drop(['index', 'mini', 'minimini'], axis=1, inplace=True)

    #helper_print_with_time('interpolate_features:', ','.join(df_detail.columns), sep='')
    return df_detail
#
def get_neat_features(df_detail, seq_length, rotate_length):
    #
    df_detail['xs'] = pd.Series([round(x, 4) for x in df_detail['xs']])
    df_detail['ys'] = pd.Series([round(x, 4) for x in df_detail['ys']])
    #
    gap = seq_length // rotate_length
    df_detail['isStart'] = pd.Series([(x % gap == 0) * 1 for x in df_detail['UID']])
    #('neat_features:', ','.join(df_detail.columns), sep='')
    return df_detail
