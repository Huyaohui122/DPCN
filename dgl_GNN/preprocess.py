import time
import pandas as pd
import matplotlib.pyplot as plt
import copy
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
import geopandas as gpd
from keras import models
from keras import layers
import keras
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
import keras.backend  as K


############helper########################
# 打印时间
def helper_print_with_time(*arg, sep=','):
    print(time.strftime("%H:%M:%S", time.localtime()), sep.join(map(str, arg)))


#
def draw_shape_and_node(df_shape, oid, isLegend=True, idName='OBJECTID'):
    # True OID，False OBJECTID
    col = idName
    geos = df_shape[df_shape[col] == oid]['geometry']
    geos.plot()
    lst = list(geos.iloc[0].exterior.coords)
    for i, xy in enumerate(lst):
        if i == len(lst) - 1:
            continue
        x, y = xy
        plt.scatter(x, y, label=i)
    if isLegend:
        plt.legend()
    plt.title(col + ':' + str(oid))
    return plt.show()


#
def draw_shape_by_detail(dfg, df_detail, oid, isLegend=False, cols=['xs', 'ys']):
    dft = df_detail[df_detail['OBJECTID'] == oid]
    points = int(dft['PID'].max()) + 1
    dfg[dfg['OBJECTID'] == oid].plot()
    for i in range(points):
        if dft[dft['PID'] == i].__len__() == 0:
            continue
        plt.scatter(dft[dft['PID'] == i][cols[0]].values, dft[dft['PID'] == i][cols[1]].values,
                    label=dft[dft['PID'] == i]['PID'].values[0])
    plt.title('OBJECTID={0}'.format(oid))
    if isLegend:
        plt.legend()
    return plt.show()


#
def cal_euclidean(p1, p2):
    return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])


#
def cal_arc(p1, p2, degree=False):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    arc = np.pi - np.arctan2(dy, dx)
    return arc / np.pi * 180 if degree else arc


#
def cal_area(l1, l2, l3):
    p = (l1 + l2 + l3) / 2
    area = p * (p - l1) * (p - l2) * (p - l3)
    area = 0 if area <= 0 else np.sqrt(area)
    # area=np.sqrt(p*(p-l1)*(p-l2)*(p-l3))
    return area


############helper########################
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
    helper_print_with_time('MBR:', ','.join(df_mbr.columns), sep='')
    return df_mbr, df_shape


def get_shape_rotate(df_shape):
    oids = [x for x in ['OID', 'OBJECTID'] if x in df_shape.columns]
    df_aff = copy.deepcopy(df_shape[oids + ['geometry', 'lon_arc', 'lon_len', 'short_len', 'Name']])
    df_aff['geometry'] = pd.Series(
        [affinity.rotate(geo, lon_arc) for geo, lon_arc in df_aff[['geometry', 'lon_arc']].values])
    helper_print_with_time('Rotate:', ','.join(df_aff.columns), sep='')
    return df_aff[oids + ['geometry', 'lon_len', 'short_len', 'Name']]


def reset_node_PID(df_node):
    oid = 'OBJECTID'
    df_node.reset_index(inplace=True, drop=False)
    dft = df_node.groupby([oid], as_index=False)['index'].agg({'id_min': 'min'})
    df_node = pd.merge(df_node, dft, how='left', on=oid)
    df_node['PID'] = df_node['index'] - df_node['id_min']
    df_node.drop(['index', 'id_min'], axis=1, inplace=True)
    return df_node


def reset_node_Pnum(df_node):
    oid = 'OBJECTID'
    df_node.drop(['p_num'], axis=1, inplace=True)
    dft = df_node.groupby([oid], as_index=False)['index'].agg({'p_num': 'count'})
    df_node = pd.merge(df_node, dft, how='left', on=oid)
    return df_node


def get_shape_Pnum(df_shape):
    df_shape['p_num'] = pd.Series([geo.exterior.coords.__len__() for geo in df_shape['geometry']])
    helper_print_with_time('get_shape_Pnum:', ','.join(df_shape.columns), sep='')
    return df_shape


def get_shape_features(gdf):
    # gdf['length']=gdf['geometry'].exterior.length
    # gdf['area']=gdf['geometry'].exterior.area
    # gdf['o_x']=gdf['geometry'].boundary.centroid.x
    # gdf['o_y']=gdf['geometry'].boundary.centroid.y
    gdf['xy'] = pd.Series(['|'.join(map(str, geo.exterior.coords)) for geo in gdf['geometry']])
    gdf['p_num'] = pd.Series([geo.exterior.coords.__len__() for geo in gdf['geometry']])
    outlist = [x for x in
               ['OID', 'OBJECTID', 'Floor', 'geometry', 'length', 'area', 'xy', 'p_num', 'o_x', 'o_y', 'lon_len',
                'short_len'] if x in gdf.columns]
    gdf = gdf[outlist]
    helper_print_with_time('shape_features:', ','.join(gdf.columns), sep='')
    return gdf


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
    helper_print_with_time('node_features:', ','.join(df_node.columns), sep='')
    return df_node


#
def node_to_polygon(df_node):
    df_node['xy'] = pd.Series([(x, y) for x, y in df_node[['x', 'y']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=True)['xy'].apply(list)
    dft = dft.reset_index(drop=False)
    dft['geometry'] = pd.Series([Polygon(xy) for xy in dft['xy']])
    dft = gpd.GeoDataFrame(dft)
    helper_print_with_time('node_to_polygon')
    return dft


#
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
    helper_print_with_time('simplify_cos:', ','.join(df_line.columns), sep='')
    return df_line


#
def simplify_dp_on_shape(df_shape, tor=0.000001, type=1):
    # type=1,relative;type=2,absolute
    if type == 1:
        df_shape['geometry'] = pd.Series(
            [geo.simplify(tor * diag) for geo, diag in df_shape[['geometry', 'diag']].values])
    else:
        df_shape['geometry'] = df_shape['geometry'].simplify(tor)
    helper_print_with_time('simplify_dp:', ','.join(df_shape.columns), sep='')
    return df_shape


#
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
    helper_print_with_time('simplify_shape:', ','.join(df_poly.columns), sep='')
    return df_poly


#
def get_shape_normalize(df_use, if_scale_y):
    df_use = copy.deepcopy(df_use)
    #
    df_node = get_node_features(df_use)
    df_node['xr'] = df_node.groupby(['OBJECTID'], as_index=False)['x'].shift(-1)
    df_node['yr'] = df_node.groupby(['OBJECTID'], as_index=False)['y'].shift(-1)
    df_node['dx'] = df_node['xr'] - df_node['x']
    df_node['dy'] = df_node['yr'] - df_node['y']
    df_node['dl'] = pd.Series([(dx ** 2 + dy ** 2) ** 0.5 for dx, dy in df_node[['dx', 'dy']].values])

    #
    df_node['px'] = pd.Series([0.5 * dl * (x + xr) for x, xr, dl in df_node[['x', 'xr', 'dl']].values])
    df_node['py'] = pd.Series([0.5 * dl * (y + yr) for y, yr, dl in df_node[['y', 'yr', 'dl']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=False)['px', 'py', 'dl'].sum()
    dft['mu_x'] = dft['px'] / dft['dl']
    dft['mu_y'] = dft['py'] / dft['dl']
    df_node = pd.merge(df_node, dft[['OBJECTID', 'mu_x', 'mu_y']], how='left', on='OBJECTID')

    #
    df_node['ddx'] = pd.Series(
        [dl * ((xr - mu_x) ** 2 + (x - mu_x) ** 2 + (x - mu_x) * (xr - mu_x)) / 3 for dl, x, xr, mu_x in
         df_node[['dl', 'x', 'xr', 'mu_x']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=False)['ddx', 'dl'].sum()
    dft['del_x'] = (dft['ddx'] / dft['dl']) ** 0.5
    df_node = pd.merge(df_node, dft[['OBJECTID', 'del_x']], how='left', on='OBJECTID')
    #
    df_node['ddy'] = pd.Series(
        [dl * ((xr - mu_x) ** 2 + (x - mu_x) ** 2 + (x - mu_x) * (xr - mu_x)) / 3 for dl, x, xr, mu_x in
         df_node[['dl', 'y', 'yr', 'mu_y']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=False)['ddy', 'dl'].sum()
    dft['del_y'] = (dft['ddy'] / dft['dl']) ** 0.5
    df_node = pd.merge(df_node, dft[['OBJECTID', 'del_y']], how='left', on='OBJECTID')

    #
    dft = df_node[['OBJECTID', 'mu_x', 'mu_y', 'del_x', 'del_y']].groupby(['OBJECTID'], as_index=False).head(1)
    df_use = pd.merge(df_use, dft, how='left', on='OBJECTID')
    df_use['geometry'] = pd.Series(
        [affinity.translate(geo, -mx, -my) for mx, my, geo in df_use[['mu_x', 'mu_y', 'geometry']].values])
    #
    if if_scale_y:
        df_use['geometry'] = pd.Series([affinity.scale(geo, 1 / del_x, 1 / del_y) for del_x, del_y, geo in
                                        df_use[['del_x', 'del_y', 'geometry']].values])
    else:
        df_use['geometry'] = pd.Series(
            [affinity.scale(geo, 1 / del_x, 1 / del_x) for del_x, geo in df_use[['del_x', 'geometry']].values])
    df_use.drop(['mu_x', 'mu_y', 'del_x', 'del_y'], axis=1, inplace=True)
    helper_print_with_time('get_shape_normalize:', ','.join(df_use.columns), sep='')
    return df_use


#
def get_shape_maxmin(df_use, if_scale_y):
    df_use = copy.deepcopy(df_use)
    # XY
    df_use['x_max'] = pd.Series([max(geo.exterior.xy[0]) for geo in df_use['geometry']])
    df_use['x_min'] = pd.Series([min(geo.exterior.xy[0]) for geo in df_use['geometry']])
    df_use['mu_x'] = (df_use['x_max'] + df_use['x_min']) / 2
    df_use['scale_x'] = (df_use['x_max'] - df_use['x_min']) / 2

    df_use['y_max'] = pd.Series([max(geo.exterior.xy[1]) for geo in df_use['geometry']])
    df_use['y_min'] = pd.Series([min(geo.exterior.xy[1]) for geo in df_use['geometry']])
    df_use['mu_y'] = (df_use['y_max'] + df_use['y_min']) / 2
    df_use['scale_y'] = (df_use['y_max'] - df_use['y_min']) / 2
    #
    df_use['geometry'] = pd.Series(
        [affinity.translate(geo, -mx, -my) for mx, my, geo in df_use[['mu_x', 'mu_y', 'geometry']].values])
    # -1~1
    if if_scale_y:
        df_use['geometry'] = pd.Series([affinity.scale(geo, 1 / del_x, 1 / del_y) for del_x, del_y, geo in
                                        df_use[['scale_x', 'scale_y', 'geometry']].values])
    else:
        df_use['geometry'] = pd.Series(
            [affinity.scale(geo, 1 / del_x, 1 / del_x) for del_x, geo in df_use[['scale_x', 'geometry']].values])
    #
    df_use.drop(['mu_x', 'mu_y', 'scale_x', 'scale_y', 'x_max', 'x_min', 'y_max', 'y_min'], axis=1, inplace=True)
    helper_print_with_time('get_shape_maxmin:', ','.join(df_use.columns), sep='')
    return df_use


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


#
def get_line_features(df_node, POINTS_SHAPE=20):
    # POINTS_SHAPE
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
    df_line['more'] = df_line['last_cum_dr'] % meta_r
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

    helper_print_with_time('line_features:', ','.join(df_line.columns), sep='')
    return df_line


#
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

    helper_print_with_time('interpolate_features:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_neat_features(df_detail, seq_length, rotate_length):
    #
    df_detail['xs'] = pd.Series([round(x, 4) for x in df_detail['xs']])
    df_detail['ys'] = pd.Series([round(x, 4) for x in df_detail['ys']])
    #
    gap = seq_length // rotate_length
    df_detail['isStart'] = pd.Series([(x % gap == 0) * 1 for x in df_detail['UID']])
    helper_print_with_time('neat_features:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_single_features(df_detail, k):
    df_detail = copy.deepcopy(df_detail)
    #
    dft = df_detail.groupby(['OBJECTID'], as_index=False)['UID'].agg({'uid_max': 'max'})
    df_detail = pd.merge(df_detail, dft, how='left', on='OBJECTID')
    #
    coor_dic = {int(_id): [a_x, a_y] for _id, a_x, a_y in df_detail[['OID_UID', 'xs', 'ys']].values}
    # O
    df_detail['o_x'] = 0
    df_detail['o_y'] = 0
    # A
    df_detail.rename(inplace=True, columns={'xs': 'a_x', 'ys': 'a_y'})
    # B
    df_detail['temp'] = df_detail['UID'] - k
    df_detail['temp'] = pd.Series(
        [oid * 1000 + temp if temp >= 0 else oid * 1000 + uid_max + 1 + temp for uid_max, temp, oid in
         df_detail[['uid_max', 'temp', 'OBJECTID']].values])
    df_detail['b_x'] = pd.Series([coor_dic[key][0] for key in df_detail['temp']])
    df_detail['b_y'] = pd.Series([coor_dic[key][1] for key in df_detail['temp']])
    # C
    df_detail['temp'] = df_detail['UID'] + k
    df_detail['temp'] = pd.Series(
        [oid * 1000 + temp if temp <= uid_max else oid * 1000 + temp - uid_max - 1 for uid_max, temp, oid in
         df_detail[['uid_max', 'temp', 'OBJECTID']].values])
    df_detail['c_x'] = pd.Series([coor_dic[key][0] for key in df_detail['temp']])
    df_detail['c_y'] = pd.Series([coor_dic[key][1] for key in df_detail['temp']])

    #
    # AB,AC,BC,OA,OB,OC
    df_detail['l_ab'] = pd.Series(
        [cal_euclidean([ax, ay], [bx, by]) for ax, ay, bx, by in df_detail[['a_x', 'a_y', 'b_x', 'b_y']].values])
    df_detail['l_ac'] = pd.Series(
        [cal_euclidean([ax, ay], [cx, cy]) for ax, ay, cx, cy in df_detail[['a_x', 'a_y', 'c_x', 'c_y']].values])
    df_detail['l_bc'] = pd.Series(
        [cal_euclidean([cx, cy], [bx, by]) for cx, cy, bx, by in df_detail[['c_x', 'c_y', 'b_x', 'b_y']].values])
    df_detail['l_oa'] = pd.Series(
        [cal_euclidean([ax, ay], [ox, oy]) for ax, ay, ox, oy in df_detail[['a_x', 'a_y', 'o_x', 'o_y']].values])
    df_detail['l_ob'] = pd.Series(
        [cal_euclidean([bx, by], [ox, oy]) for bx, by, ox, oy in df_detail[['b_x', 'b_y', 'o_x', 'o_y']].values])
    df_detail['l_oc'] = pd.Series(
        [cal_euclidean([cx, cy], [ox, oy]) for cx, cy, ox, oy in df_detail[['c_x', 'c_y', 'o_x', 'o_y']].values])
    #
    #

    df_detail['arc_ba'] = round(
        1 - np.arctan2(df_detail['a_y'] - df_detail['b_y'], df_detail['a_x'] - df_detail['b_x']) / np.pi, 6)
    df_detail['arc_ac'] = round(
        1 - np.arctan2(df_detail['c_y'] - df_detail['a_y'], df_detail['c_x'] - df_detail['a_x']) / np.pi, 6)
    df_detail['arc_ob'] = round(
        1 - np.arctan2(df_detail['b_y'] - df_detail['o_y'], df_detail['b_x'] - df_detail['o_x']) / np.pi, 6)
    df_detail['arc_oc'] = round(
        1 - np.arctan2(df_detail['c_y'] - df_detail['o_y'], df_detail['c_x'] - df_detail['o_x']) / np.pi, 6)
    #
    df_detail['angle_bac'] = pd.Series([(ac - ba - 1) % 2 for ba, ac in df_detail[['arc_ba', 'arc_ac']].values])
    df_detail['angle_boc'] = pd.Series([(oc - ob) % 2 for ob, oc in df_detail[['arc_ob', 'arc_oc']].values])
    #
    df_detail['rotate_bac'] = pd.Series([angle if angle < 1 else 2 - angle for angle in df_detail['angle_bac']])
    df_detail['rotate_boc'] = pd.Series([angle if angle < 1 else 2 - angle for angle in df_detail['angle_boc']])
    #
    # Area of Tri_ABC
    df_detail['s_abc'] = pd.Series([(-1 if angle < 1 else 1) * cal_area(l1, l2, l3) for l1, l2, l3, angle in
                                    df_detail[['l_ab', 'l_bc', 'l_ac', 'angle_bac']].values])
    # Area of Tri_OBC
    df_detail['s_obc'] = pd.Series([cal_area(l1, l2, l3) for l1, l2, l3 in df_detail[['l_ob', 'l_bc', 'l_oc']].values])
    # of Tri_OBC
    df_detail['c_obc'] = (df_detail['l_ob'] + df_detail['l_oc'] + df_detail['l_bc']) / 3
    df_detail['r_obc'] = df_detail['s_obc'] / df_detail['c_obc']
    cols = ['OBJECTID', 'PID', 'UID', 'OID_UID', 'isBegin', 'isStart'
        , 'l_bc', 'l_oa'
        , 'rotate_bac', 'rotate_boc'
        , 's_abc', 's_obc', 'c_obc', 'r_obc'
            # ,'dx_bc','dy_bc','dx_ac','dy_ac'
            # ,'l_ob','l_oc'
            ]
    df_detail = df_detail[[x for x in cols if x in df_detail.columns]]
    helper_print_with_time('k=', k, '\tsingle_feature:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_multi_features(df_detail, klst):
    cols = ['l_bc', 's_abc', 's_obc', 'c_obc', 'r_obc', 'rotate_bac', 'rotate_boc'
            # ,'l_oa'
            ]
    pDic = {}
    for k in klst:
        dft = get_single_features(df_detail, k)
        dft.columns = ['k{0}_{1}'.format(k, x) if x in cols else x for x in dft.columns]
        pDic[k] = dft
    for k in klst:
        newcols = ['k{0}_{1}'.format(k, x) for x in cols if 'k{0}_{1}'.format(k, x) in pDic[k].columns]
        if k == klst[-1]:
            newcols.append('l_oa')
        df_detail = pd.merge(df_detail, pDic[k][['OID_UID'] + newcols], how='left', on='OID_UID')
    helper_print_with_time('multi_feature:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_overall_features(df_features, df_use):
    dft = copy.deepcopy(df_use)
    #
    dft['Area'] = dft['geometry'].area
    dft['Perimeter'] = dft['geometry'].exterior.length
    dft['Elongation'] = dft['lon_len'] / dft['short_len']
    dft['Circularity'] = 4 * np.pi * dft['Area'] / dft['Perimeter'] / dft['Perimeter']
    dft = dft[['Area', 'Perimeter', 'Elongation', 'Circularity', 'OBJECTID']]
    #
    dfg = df_features[['OBJECTID', 'l_oa']].groupby(['OBJECTID'], as_index=False)['l_oa'].agg({'MeanRedius': 'mean'})
    #
    df_features = pd.merge(df_features, dft, how='left', on='OBJECTID')
    df_features = pd.merge(df_features, dfg, how='left', on='OBJECTID')
    #
    helper_print_with_time('overall_feature:', ','.join(df_features.columns), sep='')
    return df_features


#
def get_normalize_features(df_features, norm_type):
    df_features = copy.deepcopy(df_features)
    cols = [x for x in df_features.columns if 'k' in x and 'rotate' not in x] + ['Elongation', 'Circularity',
                                                                                 'Rectangularity', 'Convexity', 'l_oa',
                                                                                 'MeanRedius']
    #     +['l_oa','Area','Perimeter','Elongation','Circularity','MeanRedius']
    #
    df_stat = df_features[cols].describe().transpose()
    for col in cols:
        col_min, col_max, col_std, col_mean = df_stat.loc[col][['min', 'max', 'std', 'mean']].values
        if norm_type == 'zscore':
            df_features[col] = (df_features[col] - col_mean) / col_std
        elif norm_type == 'minmax':
            df_features[col] = (df_features[col] - col_min) / (col_max - col_min)
    #
    helper_print_with_time('normalize_feature:', ','.join(df_features.columns), sep='')
    return df_features


#
def get_train_sequence(df_features, cols, rotate_type):
    #
    df_features['features'] = pd.Series([list(tple) for tple in df_features[cols].values])
    #
    if rotate_type == 'vertex':
        beginDic = {group[0]: list(group[1]['UID'].values) for group in
                    df_features[df_features['isBegin'] == 1][['OBJECTID', 'UID']].groupby('OBJECTID')}
    elif rotate_type == 'equal':
        beginDic = {group[0]: list(group[1]['UID'].values) for group in
                    df_features[df_features['isStart'] == 1][['OBJECTID', 'UID']].groupby('OBJECTID')}
    elif rotate_type == 'none':
        beginDic = {group[0]: list(group[1]['UID'].values) for group in
                    df_features[df_features['UID'] == 0][['OBJECTID', 'UID']].groupby('OBJECTID')}
    maxLen = max([len(value) for key, value in beginDic.items()])
    #
    features_list = {group[0]: list(group[1]['features'].values) for group in
                     df_features[['OBJECTID', 'features']].groupby('OBJECTID')}
    #
    features_list_list = {key: [value[begin:] + value[:begin] for begin in beginDic[key]] for key, value in
                          features_list.items()}
    features_list_list = {key: value + [np.nan] * (maxLen - len(value)) for key, value in features_list_list.items()}
    # pd.DataFrame
    dft = pd.DataFrame(features_list_list).transpose().stack().apply(pd.Series)
    dft.columns = ['f_{0}'.format(x) for x in dft.columns]
    dft.reset_index(drop=False, inplace=True)
    dft.rename(inplace=True, columns={'level_0': 'OBJECTID', 'level_1': 'LID'})
    helper_print_with_time('train_seq:', ','.join(dft.columns), sep='')
    return dft


#
def get_seq2seq_train_dataset(df_seq, circles=1):
    featureList = [x for x in df_seq.columns if 'f_' in x]
    de_input_list = en_input_list = featureList * circles
    de_target_list = de_input_list[1:] + de_input_list[:1]
    return de_input_list, en_input_list, de_target_list


############model########################
#
def AE_model(hparams, features_len):
    rnn = layers.LSTM if hparams['rnn_type'] == 'lstm' else layers.GRU
    rnn_gpu = layers.CuDNNLSTM if hparams['rnn_type'] == 'lstm' else layers.CuDNNGRU
    # GPU RNN
    if hparams['GPU']:
        encoder_lstm = rnn_gpu(units=hparams['z_size'], return_state=True, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn_gpu(units=hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    else:
        encoder_lstm = rnn(hparams['z_size'], return_state=True, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn(hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    # Encoder,init_h, init_c
    encoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='encoder_input')
    encoder_output, *encoder_states = encoder_lstm(encoder_input)

    # Decoder
    decoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='decoder_input')
    rnn_output = decoder_lstm(decoder_input, initial_state=encoder_states)

    decoder_output = layers.Dense(features_len)(rnn_output)
    # Model
    ae_model = keras.Model([encoder_input, decoder_input], [decoder_output])
    enc_model = keras.Model(encoder_input, encoder_states)
    embed_model = keras.Model(encoder_input, encoder_output)
    ae_model.compile(optimizer=hparams['optimizer'], loss='mse')
    return ae_model, enc_model, embed_model


# peeky
def peeky_AE_model(hparams, features_len):
    rnn = layers.LSTM if hparams['rnn_type'] == 'lstm' else layers.GRU
    rnn_gpu = layers.CuDNNLSTM if hparams['rnn_type'] == 'lstm' else layers.CuDNNGRU
    # GPU RNN
    if hparams['GPU']:
        encoder_lstm = rnn_gpu(units=hparams['z_size'], return_state=True, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn_gpu(units=hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    else:
        encoder_lstm = rnn(hparams['z_size'], return_state=True, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn(hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    # Encoder,init_h, init_c
    encoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='encoder_input')
    encoder_output, *encoder_states = encoder_lstm(encoder_input)

    # Repeat latent_z
    tile_z = layers.RepeatVector(hparams['seq_length'])(encoder_output)

    # Decoder
    decoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='decoder_input')
    decoder_full_input = layers.Concatenate()([decoder_input, tile_z])
    rnn_output = decoder_lstm(decoder_full_input, initial_state=encoder_states)

    decoder_output = layers.Dense(features_len)(rnn_output)
    # Model
    peeky_ae_model = keras.Model([encoder_input, decoder_input], [decoder_output])
    enc_model = keras.Model(encoder_input, encoder_states)
    embed_model = keras.Model(encoder_input, encoder_output)
    peeky_ae_model.compile(optimizer=hparams['optimizer'], loss='mse')
    return peeky_ae_model, enc_model, embed_model


#
def VAE_model(hparams, features_len):
    def sampling(args):
        z_mu, z_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mu)[0], hparams['z_size']), mean=0., stddev=1.)
        return z_mu + K.exp(0.5 * z_sigma) * epsilon

    rnn = layers.LSTM if hparams['rnn_type'] == 'lstm' else layers.GRU
    rnn_gpu = layers.CuDNNLSTM if hparams['rnn_type'] == 'lstm' else layers.CuDNNGRU
    # GPU RNN
    if hparams['GPU']:
        encoder_lstm = rnn_gpu(units=hparams['z_size'], return_state=False, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn_gpu(units=hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    else:
        encoder_lstm = rnn(hparams['z_size'], return_state=False, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn(hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')

    # Encoder
    encoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='encoder_input')
    encoder_output = encoder_lstm(encoder_input)

    z_mu = layers.Dense(hparams['z_size'], name='mu')(encoder_output)
    z_sigma = layers.Dense(hparams['z_size'], name='sigma')(encoder_output)

    z = layers.Lambda(sampling)([z_mu, z_sigma])

    # initial state
    init_h = layers.Dense(units=hparams['z_size'], activation='tanh', name='dec_initial_h')(z)
    init_c = layers.Dense(units=hparams['z_size'], activation='tanh', name='dec_initial_c')(z)

    # Decoder
    decoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='decoder_input')
    rnn_output = decoder_lstm(decoder_input, initial_state=[init_h, init_c])

    decoder_output = layers.Dense(features_len)(rnn_output)

    # Model
    vae_model = keras.Model([encoder_input, decoder_input], [decoder_output])
    enc_model = keras.Model(encoder_input, z_mu)
    embed_model = keras.Model(encoder_input, z_mu)

    # compile
    def md_loss_func(y_true, y_pred):
        md_loss = keras.metrics.mse(y_true, y_pred)
        return md_loss

    def kl_loss_func(*args, **kwargs):
        kl_cost = -0.5 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma))
        return kl_cost

    def model_loss():
        md_loss = md_loss_func
        kl_loss = kl_loss_func
        kl_weight = hparams['kl_weight']

        def vae_loss(y_true, y_pred):
            model_loss = kl_weight * kl_loss() + md_loss(y_true, y_pred)
            return model_loss

        return vae_loss

    vae_model.compile(optimizer=hparams['optimizer'], loss=model_loss(), metrics=[md_loss_func, kl_loss_func])
    return vae_model, enc_model, embed_model


############model########################


import time
import pandas as pd
import matplotlib.pyplot as plt
import copy
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
import geopandas as gpd
from keras import models
from keras import layers
import keras
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
import keras.backend  as K


############helper########################
#
def helper_print_with_time(*arg, sep=','):
    print(time.strftime("%H:%M:%S", time.localtime()), sep.join(map(str, arg)))


#
def draw_shape_and_node(df_shape, oid, isLegend=True, idName='OBJECTID'):
    #
    col = idName
    geos = df_shape[df_shape[col] == oid]['geometry']
    geos.plot()
    lst = list(geos.iloc[0].exterior.coords)
    for i, xy in enumerate(lst):
        if i == len(lst) - 1:
            continue
        x, y = xy
        plt.scatter(x, y, label=i)
    if isLegend:
        plt.legend()
    plt.title(col + ':' + str(oid))
    return plt.show()


#
def draw_shape_by_detail(dfg, df_detail, oid, isLegend=False, cols=['xs', 'ys']):
    dft = df_detail[df_detail['OBJECTID'] == oid]
    points = dft['PID'].max() + 1
    dfg[dfg['OBJECTID'] == oid].plot()
    for i in range(points):
        plt.scatter(dft[dft['PID'] == i][cols[0]].values, dft[dft['PID'] == i][cols[1]].values,
                    label=dft[dft['PID'] == i]['PID'].values[0])
    plt.title('OBJECTID={0}'.format(oid))
    plt.legend()
    return plt.show()


#
def cal_euclidean(p1, p2):
    return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])


#
def cal_arc(p1, p2, degree=False):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    arc = np.pi - np.arctan2(dy, dx)
    return arc / np.pi * 180 if degree else arc


# ()
def cal_area(l1, l2, l3):
    p = (l1 + l2 + l3) / 2
    area = p * (p - l1) * (p - l2) * (p - l3)
    area = 0 if area <= 0 else np.sqrt(area)
    # area=np.sqrt(p*(p-l1)*(p-l2)*(p-l3))
    return area


############helper########################
#
#
def get_shape_mbr(df_shape):
    oid = 'OID' if 'OID' in df_shape.columns else 'OBJECTID'
    df_mbr = copy.deepcopy(df_shape[[oid, 'geometry']])
    df_mbr.reset_index(drop=True, inplace=True)#重置索引，并删去原索引
    df_mbr['geometry'] = pd.Series([geo.minimum_rotated_rectangle for geo in df_mbr['geometry']])#计算最小外接矩形 geo.minimum_rotated_rectangle
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
    helper_print_with_time('MBR:', ','.join(df_mbr.columns), sep='')
    return df_mbr, df_shape


#
#
def get_shape_rotate(df_shape):
    oids = [x for x in ['OID', 'OBJECTID'] if x in df_shape.columns]
    df_aff = copy.deepcopy(df_shape[oids + ['geometry', 'lon_arc', 'lon_len', 'short_len', 'Name']])
    df_aff['geometry'] = pd.Series(
        [affinity.rotate(geo, lon_arc) for geo, lon_arc in df_aff[['geometry', 'lon_arc']].values])
    helper_print_with_time('Rotate:', ','.join(df_aff.columns), sep='')
    return df_aff[oids + ['geometry', 'lon_len', 'short_len', 'Name']]


#
def reset_node_PID(df_node):
    oid = 'OBJECTID'
    df_node.reset_index(inplace=True, drop=False)
    dft = df_node.groupby([oid], as_index=False)['index'].agg({'id_min': 'min'})
    df_node = pd.merge(df_node, dft, how='left', on=oid)
    df_node['PID'] = df_node['index'] - df_node['id_min']
    df_node.drop(['index', 'id_min'], axis=1, inplace=True)
    return df_node


def reset_node_Pnum(df_node):
    oid = 'OBJECTID'
    df_node.drop(['p_num'], axis=1, inplace=True)
    dft = df_node.groupby([oid], as_index=False)['index'].agg({'p_num': 'count'})
    df_node = pd.merge(df_node, dft, how='left', on=oid)
    return df_node


def get_shape_Pnum(df_shape):
    df_shape['p_num'] = pd.Series([geo.exterior.coords.__len__() for geo in df_shape['geometry']])
    helper_print_with_time('get_shape_Pnum:', ','.join(df_shape.columns), sep='')
    return df_shape


#
def get_shape_features(gdf):
    #
    #
    # gdf['length']=gdf['geometry'].exterior.length
    # gdf['area']=gdf['geometry'].exterior.area
    # gdf['o_x']=gdf['geometry'].boundary.centroid.x
    # gdf['o_y']=gdf['geometry'].boundary.centroid.y
    gdf['xy'] = pd.Series(['|'.join(map(str, geo.exterior.coords)) for geo in gdf['geometry']])
    gdf['p_num'] = pd.Series([geo.exterior.coords.__len__() for geo in gdf['geometry']])
    outlist = [x for x in
               ['OID', 'OBJECTID', 'Floor', 'geometry', 'length', 'area', 'xy', 'p_num', 'o_x', 'o_y', 'lon_len',
                'short_len'] if x in gdf.columns]
    gdf = gdf[outlist]
    helper_print_with_time('shape_features:', ','.join(gdf.columns), sep='')
    return gdf


#
def get_node_features(df_shape):
    #
    oid = 'OBJECTID'
    df_shape['xy'] = pd.Series(['|'.join(map(str, geo.exterior.coords)) for geo in df_shape['geometry']])
    df_shape['p_num'] = pd.Series([geo.exterior.coords.__len__() for geo in df_shape['geometry']])
    df_shape['length'] = df_shape['geometry'].exterior.length
    df_node = (df_shape.set_index([oid, 'p_num', 'length'])['xy']
               .str.split('|', expand=True)
               .stack()
               .reset_index(level=3, drop=True)
               .reset_index(name='xy'))
    #
    df_node['x'] = pd.Series([float(xy.split(',')[0][1:]) for xy in df_node['xy'].values])
    df_node['y'] = pd.Series([float(xy.split(',')[1][:-1]) for xy in df_node['xy'].values])
    #
    df_node.reset_index(inplace=True, drop=False)
    dft = df_node.groupby([oid], as_index=False)['index'].agg({'id_min': 'min'})
    df_node = pd.merge(df_node, dft, how='left', on=oid)
    df_node['PID'] = df_node['index'] - df_node['id_min']
    df_node.drop(['xy', 'id_min', 'index'], axis=1, inplace=True)
    helper_print_with_time('node_features:', ','.join(df_node.columns), sep='')
    return df_node


#
def node_to_polygon(df_node):
    df_node['xy'] = pd.Series([(x, y) for x, y in df_node[['x', 'y']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=True)['xy'].apply(list)
    dft = dft.reset_index(drop=False)
    dft['geometry'] = pd.Series([Polygon(xy) for xy in dft['xy']])
    dft = gpd.GeoDataFrame(dft)
    helper_print_with_time('node_to_polygon')
    return dft


#
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
    helper_print_with_time('simplify_cos:', ','.join(df_line.columns), sep='')
    return df_line


#
def simplify_dp_on_shape(df_shape, tor=0.000001, type=1):
    # type=1,relative;type=2,absolute
    if type == 1:
        df_shape['geometry'] = pd.Series(
            [geo.simplify(tor * diag) for geo, diag in df_shape[['geometry', 'diag']].values])
    else:
        df_shape['geometry'] = df_shape['geometry'].simplify(tor)
    helper_print_with_time('simplify_dp:', ','.join(df_shape.columns), sep='')
    return df_shape


#
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
    # 6.
    df_poly = simplify_dp_on_shape(df_poly, tor_dist, type=simplify_type)
    df_poly.drop(['xy', 'diag'], axis=1, inplace=True)
    helper_print_with_time('simplify_shape:', ','.join(df_poly.columns), sep='')
    return df_poly


#
def get_shape_normalize(df_use, if_scale_y):
    df_use = copy.deepcopy(df_use)
    #
    df_node = get_node_features(df_use)
    df_node['xr'] = df_node.groupby(['OBJECTID'], as_index=False)['x'].shift(-1)
    df_node['yr'] = df_node.groupby(['OBJECTID'], as_index=False)['y'].shift(-1)
    df_node['dx'] = df_node['xr'] - df_node['x']
    df_node['dy'] = df_node['yr'] - df_node['y']
    df_node['dl'] = pd.Series([(dx ** 2 + dy ** 2) ** 0.5 for dx, dy in df_node[['dx', 'dy']].values])

    #
    df_node['px'] = pd.Series([0.5 * dl * (x + xr) for x, xr, dl in df_node[['x', 'xr', 'dl']].values])
    df_node['py'] = pd.Series([0.5 * dl * (y + yr) for y, yr, dl in df_node[['y', 'yr', 'dl']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=False)['px', 'py', 'dl'].sum()
    dft['mu_x'] = dft['px'] / dft['dl']
    dft['mu_y'] = dft['py'] / dft['dl']
    df_node = pd.merge(df_node, dft[['OBJECTID', 'mu_x', 'mu_y']], how='left', on='OBJECTID')

    #
    df_node['ddx'] = pd.Series(
        [dl * ((xr - mu_x) ** 2 + (x - mu_x) ** 2 + (x - mu_x) * (xr - mu_x)) / 3 for dl, x, xr, mu_x in
         df_node[['dl', 'x', 'xr', 'mu_x']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=False)['ddx', 'dl'].sum()
    dft['del_x'] = (dft['ddx'] / dft['dl']) ** 0.5
    df_node = pd.merge(df_node, dft[['OBJECTID', 'del_x']], how='left', on='OBJECTID')
    #
    df_node['ddy'] = pd.Series(
        [dl * ((xr - mu_x) ** 2 + (x - mu_x) ** 2 + (x - mu_x) * (xr - mu_x)) / 3 for dl, x, xr, mu_x in
         df_node[['dl', 'y', 'yr', 'mu_y']].values])
    dft = df_node.groupby(['OBJECTID'], as_index=False)['ddy', 'dl'].sum()
    dft['del_y'] = (dft['ddy'] / dft['dl']) ** 0.5
    df_node = pd.merge(df_node, dft[['OBJECTID', 'del_y']], how='left', on='OBJECTID')

    #
    dft = df_node[['OBJECTID', 'mu_x', 'mu_y', 'del_x', 'del_y']].groupby(['OBJECTID'], as_index=False).head(1)
    df_use = pd.merge(df_use, dft, how='left', on='OBJECTID')
    df_use['geometry'] = pd.Series(
        [affinity.translate(geo, -mx, -my) for mx, my, geo in df_use[['mu_x', 'mu_y', 'geometry']].values])
    #
    if if_scale_y:
        df_use['geometry'] = pd.Series([affinity.scale(geo, 1 / del_x, 1 / del_y) for del_x, del_y, geo in
                                        df_use[['del_x', 'del_y', 'geometry']].values])
    else:
        df_use['geometry'] = pd.Series(
            [affinity.scale(geo, 1 / del_x, 1 / del_x) for del_x, geo in df_use[['del_x', 'geometry']].values])
    df_use.drop(['mu_x', 'mu_y', 'del_x', 'del_y'], axis=1, inplace=True)
    helper_print_with_time('get_shape_normalize:', ','.join(df_use.columns), sep='')
    return df_use


# -1~-1~1
def get_shape_maxmin(df_use, if_scale_y):
    df_use = copy.deepcopy(df_use)
    #
    df_use['x_max'] = pd.Series([max(geo.exterior.xy[0]) for geo in df_use['geometry']])
    df_use['x_min'] = pd.Series([min(geo.exterior.xy[0]) for geo in df_use['geometry']])
    df_use['mu_x'] = (df_use['x_max'] + df_use['x_min']) / 2
    df_use['scale_x'] = (df_use['x_max'] - df_use['x_min']) / 2

    df_use['y_max'] = pd.Series([max(geo.exterior.xy[1]) for geo in df_use['geometry']])
    df_use['y_min'] = pd.Series([min(geo.exterior.xy[1]) for geo in df_use['geometry']])
    df_use['mu_y'] = (df_use['y_max'] + df_use['y_min']) / 2
    df_use['scale_y'] = (df_use['y_max'] - df_use['y_min']) / 2
    # ,0
    df_use['geometry'] = pd.Series(
        [affinity.translate(geo, -mx, -my) for mx, my, geo in df_use[['mu_x', 'mu_y', 'geometry']].values])
    # -1~1
    if if_scale_y:
        df_use['geometry'] = pd.Series([affinity.scale(geo, 1 / del_x, 1 / del_y) for del_x, del_y, geo in
                                        df_use[['scale_x', 'scale_y', 'geometry']].values])
    else:
        df_use['geometry'] = pd.Series(
            [affinity.scale(geo, 1 / del_x, 1 / del_x) for del_x, geo in df_use[['scale_x', 'geometry']].values])
    #
    df_use.drop(['mu_x', 'mu_y', 'scale_x', 'scale_y', 'x_max', 'x_min', 'y_max', 'y_min'], axis=1, inplace=True)
    helper_print_with_time('get_shape_maxmin:', ','.join(df_use.columns), sep='')
    return df_use


#
def get_line_features(df_node, POINTS_SHAPE=20):
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
    df_line['more'] = df_line['last_cum_dr'] % meta_r
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

    helper_print_with_time('line_features:', ','.join(df_line.columns), sep='')
    return df_line


#
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

    helper_print_with_time('interpolate_features:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_neat_features(df_detail, seq_length, rotate_length):
    #
    df_detail['xs'] = pd.Series([round(x, 4) for x in df_detail['xs']])
    df_detail['ys'] = pd.Series([round(x, 4) for x in df_detail['ys']])
    #
    gap = seq_length // rotate_length
    df_detail['isStart'] = pd.Series([(x % gap == 0) * 1 for x in df_detail['UID']])
    helper_print_with_time('neat_features:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_single_features(df_detail, k):
    df_detail = copy.deepcopy(df_detail)
    #
    dft = df_detail.groupby(['OBJECTID'], as_index=False)['UID'].agg({'uid_max': 'max'})
    df_detail = pd.merge(df_detail, dft, how='left', on='OBJECTID')
    #
    coor_dic = {int(_id): [a_x, a_y] for _id, a_x, a_y in df_detail[['OID_UID', 'xs', 'ys']].values}
    #
    df_detail['o_x'] = 0
    df_detail['o_y'] = 0
    #
    df_detail.rename(inplace=True, columns={'xs': 'a_x', 'ys': 'a_y'})
    #
    df_detail['temp'] = df_detail['UID'] - k
    df_detail['temp'] = pd.Series(
        [oid * 1000 + temp if temp >= 0 else oid * 1000 + uid_max + 1 + temp for uid_max, temp, oid in
         df_detail[['uid_max', 'temp', 'OBJECTID']].values])
    df_detail['b_x'] = pd.Series([coor_dic[key][0] for key in df_detail['temp']])
    df_detail['b_y'] = pd.Series([coor_dic[key][1] for key in df_detail['temp']])
    #
    df_detail['temp'] = df_detail['UID'] + k
    df_detail['temp'] = pd.Series(
        [oid * 1000 + temp if temp <= uid_max else oid * 1000 + temp - uid_max - 1 for uid_max, temp, oid in
         df_detail[['uid_max', 'temp', 'OBJECTID']].values])
    df_detail['c_x'] = pd.Series([coor_dic[key][0] for key in df_detail['temp']])
    df_detail['c_y'] = pd.Series([coor_dic[key][1] for key in df_detail['temp']])

    #
    # AB,AC,BC,OA,OB,OC
    df_detail['l_ab'] = pd.Series(
        [cal_euclidean([ax, ay], [bx, by]) for ax, ay, bx, by in df_detail[['a_x', 'a_y', 'b_x', 'b_y']].values])
    df_detail['l_ac'] = pd.Series(
        [cal_euclidean([ax, ay], [cx, cy]) for ax, ay, cx, cy in df_detail[['a_x', 'a_y', 'c_x', 'c_y']].values])
    df_detail['l_bc'] = pd.Series(
        [cal_euclidean([cx, cy], [bx, by]) for cx, cy, bx, by in df_detail[['c_x', 'c_y', 'b_x', 'b_y']].values])
    df_detail['l_oa'] = pd.Series(
        [cal_euclidean([ax, ay], [ox, oy]) for ax, ay, ox, oy in df_detail[['a_x', 'a_y', 'o_x', 'o_y']].values])
    df_detail['l_ob'] = pd.Series(
        [cal_euclidean([bx, by], [ox, oy]) for bx, by, ox, oy in df_detail[['b_x', 'b_y', 'o_x', 'o_y']].values])
    df_detail['l_oc'] = pd.Series(
        [cal_euclidean([cx, cy], [ox, oy]) for cx, cy, ox, oy in df_detail[['c_x', 'c_y', 'o_x', 'o_y']].values])
    #
    # ~
    #
    df_detail['arc_ba'] = round(
        1 - np.arctan2(df_detail['a_y'] - df_detail['b_y'], df_detail['a_x'] - df_detail['b_x']) / np.pi, 6)
    df_detail['arc_ac'] = round(
        1 - np.arctan2(df_detail['c_y'] - df_detail['a_y'], df_detail['c_x'] - df_detail['a_x']) / np.pi, 6)
    df_detail['arc_ob'] = round(
        1 - np.arctan2(df_detail['b_y'] - df_detail['o_y'], df_detail['b_x'] - df_detail['o_x']) / np.pi, 6)
    df_detail['arc_oc'] = round(
        1 - np.arctan2(df_detail['c_y'] - df_detail['o_y'], df_detail['c_x'] - df_detail['o_x']) / np.pi, 6)
    #
    df_detail['angle_bac'] = pd.Series([(ac - ba - 1) % 2 for ba, ac in df_detail[['arc_ba', 'arc_ac']].values])
    df_detail['angle_boc'] = pd.Series([(oc - ob) % 2 for ob, oc in df_detail[['arc_ob', 'arc_oc']].values])
    #
    df_detail['rotate_bac'] = pd.Series([angle if angle < 1 else 2 - angle for angle in df_detail['angle_bac']])
    df_detail['rotate_boc'] = pd.Series([angle if angle < 1 else 2 - angle for angle in df_detail['angle_boc']])
    #
    # Area of Tri_ABC
    df_detail['s_abc'] = pd.Series([(-1 if angle < 1 else 1) * cal_area(l1, l2, l3) for l1, l2, l3, angle in
                                    df_detail[['l_ab', 'l_bc', 'l_ac', 'angle_bac']].values])
    # Area of Tri_OBC
    df_detail['s_obc'] = pd.Series([cal_area(l1, l2, l3) for l1, l2, l3 in df_detail[['l_ob', 'l_bc', 'l_oc']].values])
    # of Tri_OBC
    df_detail['c_obc'] = (df_detail['l_ob'] + df_detail['l_oc'] + df_detail['l_bc']) / 3
    df_detail['r_obc'] = df_detail['s_obc'] / df_detail['c_obc']
    cols = ['OBJECTID', 'PID', 'UID', 'OID_UID', 'isBegin', 'isStart'
        , 'l_bc', 'l_oa'
        , 'rotate_bac', 'rotate_boc'
        , 's_abc', 's_obc', 'c_obc', 'r_obc'
            # ,'dx_bc','dy_bc','dx_ac','dy_ac'
            # ,'l_ob','l_oc'
            ]
    df_detail = df_detail[[x for x in cols if x in df_detail.columns]]
    helper_print_with_time('k=', k, '\tsingle_feature:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_multi_features(df_detail, klst):
    cols = ['l_bc', 's_abc', 's_obc', 'c_obc', 'r_obc', 'rotate_bac', 'rotate_boc'
            # ,'l_oa'
            ]
    pDic = {}
    for k in klst:
        dft = get_single_features(df_detail, k)
        dft.columns = ['k{0}_{1}'.format(k, x) if x in cols else x for x in dft.columns]
        pDic[k] = dft
    for k in klst:
        newcols = ['k{0}_{1}'.format(k, x) for x in cols if 'k{0}_{1}'.format(k, x) in pDic[k].columns]
        if k == klst[-1]:
            newcols.append('l_oa')
        df_detail = pd.merge(df_detail, pDic[k][['OID_UID'] + newcols], how='left', on='OID_UID')
    helper_print_with_time('multi_feature:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_overall_features(df_features, df_use):
    dft = copy.deepcopy(df_use)
    #
    dft['Area'] = dft['geometry'].area
    dft['Perimeter'] = dft['geometry'].exterior.length
    dft['Elongation'] = dft['lon_len'] / dft['short_len']
    dft['Circularity'] = 4 * np.pi * dft['Area'] / dft['Perimeter'] / dft['Perimeter']
    dft = dft[['Area', 'Perimeter', 'Elongation', 'Circularity', 'OBJECTID']]
    #
    dfg = df_features[['OBJECTID', 'l_oa']].groupby(['OBJECTID'], as_index=False)['l_oa'].agg({'MeanRedius': 'mean'})
    #
    df_features = pd.merge(df_features, dft, how='left', on='OBJECTID')
    df_features = pd.merge(df_features, dfg, how='left', on='OBJECTID')
    #
    helper_print_with_time('overall_feature:', ','.join(df_features.columns), sep='')
    return df_features


#
def get_normalize_features(df_features, norm_type):
    df_features = copy.deepcopy(df_features)
    cols = [x for x in df_features.columns if 'k' in x and 'rotate' not in x] + ['l_oa', 'Area', 'Perimeter',
                                                                                 'Elongation', 'Circularity',
                                                                                 'MeanRedius']
    #
    df_stat = df_features[cols].describe().transpose()
    for col in cols:
        col_min, col_max, col_std, col_mean = df_stat.loc[col][['min', 'max', 'std', 'mean']].values
        if norm_type == 'zscore':
            df_features[col] = (df_features[col] - col_mean) / col_std
        elif norm_type == 'minmax':
            df_features[col] = (df_features[col] - col_min) / (col_max - col_min)
    #
    helper_print_with_time('normalize_feature:', ','.join(df_features.columns), sep='')
    return df_features


#
def get_train_sequence(df_features, cols, rotate_type):
    #
    df_features['features'] = pd.Series([list(tple) for tple in df_features[cols].values])
    #
    if rotate_type == 'vertex':
        beginDic = {group[0]: list(group[1]['UID'].values) for group in
                    df_features[df_features['isBegin'] == 1][['OBJECTID', 'UID']].groupby('OBJECTID')}
    elif rotate_type == 'equal':
        beginDic = {group[0]: list(group[1]['UID'].values) for group in
                    df_features[df_features['isStart'] == 1][['OBJECTID', 'UID']].groupby('OBJECTID')}
    elif rotate_type == 'none':
        beginDic = {group[0]: list(group[1]['UID'].values) for group in
                    df_features[df_features['UID'] == 0][['OBJECTID', 'UID']].groupby('OBJECTID')}
    maxLen = max([len(value) for key, value in beginDic.items()])
    #
    features_list = {group[0]: list(group[1]['features'].values) for group in
                     df_features[['OBJECTID', 'features']].groupby('OBJECTID')}
    #
    features_list_list = {key: [value[begin:] + value[:begin] for begin in beginDic[key]] for key, value in
                          features_list.items()}
    features_list_list = {key: value + [np.nan] * (maxLen - len(value)) for key, value in features_list_list.items()}
    # pd.
    dft = pd.DataFrame(features_list_list).transpose().stack().apply(pd.Series)
    dft.columns = ['f_{0}'.format(x) for x in dft.columns]
    dft.reset_index(drop=False, inplace=True)
    dft.rename(inplace=True, columns={'level_0': 'OBJECTID', 'level_1': 'LID'})
    helper_print_with_time('train_seq:', ','.join(dft.columns), sep='')
    return dft


#
def get_seq2seq_train_dataset(df_seq, circles=1):
    featureList = [x for x in df_seq.columns if 'f_' in x]
    de_input_list = en_input_list = featureList * circles
    de_target_list = de_input_list[1:] + de_input_list[:1]
    return de_input_list, en_input_list, de_target_list


############model########################
#
def AE_model(hparams, features_len):
    rnn = layers.LSTM if hparams['rnn_type'] == 'lstm' else layers.GRU
    rnn_gpu = layers.CuDNNLSTM if hparams['rnn_type'] == 'lstm' else layers.CuDNNGRU
    # GPU RNN
    if hparams['GPU']:
        encoder_lstm = rnn_gpu(units=hparams['z_size'], return_state=True, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn_gpu(units=hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    else:
        encoder_lstm = rnn(hparams['z_size'], return_state=True, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn(hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    # Encoder,init_h, init_c
    encoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='encoder_input')
    encoder_output, *encoder_states = encoder_lstm(encoder_input)

    # Decoder
    decoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='decoder_input')
    rnn_output = decoder_lstm(decoder_input, initial_state=encoder_states)

    decoder_output = layers.Dense(features_len)(rnn_output)
    # Model
    ae_model = keras.Model([encoder_input, decoder_input], [decoder_output])
    enc_model = keras.Model(encoder_input, encoder_states)
    embed_model = keras.Model(encoder_input, encoder_output)
    ae_model.compile(optimizer=hparams['optimizer'], loss='mse')
    return ae_model, enc_model, embed_model


# peeky
def peeky_AE_model(hparams, features_len):
    rnn = layers.LSTM if hparams['rnn_type'] == 'lstm' else layers.GRU
    rnn_gpu = layers.CuDNNLSTM if hparams['rnn_type'] == 'lstm' else layers.CuDNNGRU
    # GPU RNN
    if hparams['GPU']:
        encoder_lstm = rnn_gpu(units=hparams['z_size'], return_state=True, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn_gpu(units=hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    else:
        encoder_lstm = rnn(hparams['z_size'], return_state=True, return_sequences=False, name='encoder_lstm',
                           dropout=hparams['dropout'], recurrent_dropout=hparams['rnn_dropout'])
        decoder_lstm = rnn(hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm',
                           dropout=hparams['dropout'], recurrent_dropout=hparams['rnn_dropout'])
    # Encoder,init_h, init_c
    encoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='encoder_input')
    encoder_output, *encoder_states = encoder_lstm(encoder_input)

    # Repeat latent_z
    tile_z = layers.RepeatVector(hparams['seq_length'])(encoder_output)

    # Decoder
    decoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='decoder_input')
    decoder_full_input = layers.Concatenate()([decoder_input, tile_z])
    rnn_output = decoder_lstm(decoder_full_input, initial_state=encoder_states)

    decoder_output = layers.Dense(features_len)(rnn_output)
    # Model
    peeky_ae_model = keras.Model([encoder_input, decoder_input], [decoder_output])
    enc_model = keras.Model(encoder_input, encoder_states)
    embed_model = keras.Model(encoder_input, encoder_output)
    peeky_ae_model.compile(optimizer=hparams['optimizer'], loss='mse')
    return peeky_ae_model, enc_model, embed_model


#
def VAE_model(hparams, features_len):
    def sampling(args):
        z_mu, z_sigma = args
        batch = K.shape(z_mu)[0]
        dim = K.int_shape(z_mu)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mu + K.exp(0.5 * z_sigma) * epsilon

    rnn = layers.LSTM if hparams['rnn_type'] == 'lstm' else layers.GRU
    rnn_gpu = layers.CuDNNLSTM if hparams['rnn_type'] == 'lstm' else layers.CuDNNGRU
    # GPU RNN
    if hparams['GPU']:
        encoder_lstm = rnn_gpu(units=hparams['z_size'], return_state=False, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn_gpu(units=hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    else:
        encoder_lstm = rnn(hparams['z_size'], return_state=False, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn(hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')

    # Encoder
    encoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='encoder_input')
    encoder_output = encoder_lstm(encoder_input)

    z_mu = layers.Dense(hparams['z_size'], name='mu')(encoder_output)
    z_sigma = layers.Dense(hparams['z_size'], name='sigma')(encoder_output)

    z = layers.Lambda(sampling)([z_mu, z_sigma])

    # initial state
    #     init_h=layers.Dense(units=hparams['z_size'],activation='tanh',name='dec_initial_h')(z)
    #     init_c=layers.Dense(units=hparams['z_size'],activation='tanh',name='dec_initial_c')(z)
    # Repeat latent_z
    tile_z = layers.RepeatVector(hparams['seq_length'])(z)

    # Decoder
    decoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='decoder_input')
    decoder_full_input = layers.Concatenate()([decoder_input, tile_z])

    rnn_output = decoder_lstm(decoder_full_input)

    decoder_output = layers.Dense(features_len)(rnn_output)

    # Model
    vae_model = keras.Model([encoder_input, decoder_input], [decoder_output])
    enc_model = keras.Model(encoder_input, z_mu)
    embed_model = keras.Model(encoder_input, z_mu)

    # compile
    def md_loss_func(y_true, y_pred):
        md_loss = keras.losses.mse(y_true, y_pred)
        return md_loss

    def kl_loss_func(*args, **kwargs):
        kl_cost = -0.5 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma))
        return kl_cost

    def model_loss():
        md_loss = md_loss_func
        kl_loss = kl_loss_func
        kl_weight = hparams['kl_weight']

        def vae_loss(y_true, y_pred):
            model_loss = kl_weight * kl_loss() + md_loss(y_true, y_pred)
            return model_loss

        return vae_loss

    vae_model.compile(optimizer=hparams['optimizer'], loss=model_loss(), metrics=[md_loss_func, kl_loss_func])
    return vae_model, enc_model, embed_model


#
def _VAE_model(hparams, features_len):
    def sampling(args):
        z_mu, z_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mu)[0], hparams['z_size']), mean=0., stddev=1.)
        return z_mu + K.exp(0.5 * z_sigma) * epsilon

    rnn = layers.LSTM if hparams['rnn_type'] == 'lstm' else layers.GRU
    rnn_gpu = layers.CuDNNLSTM if hparams['rnn_type'] == 'lstm' else layers.CuDNNGRU
    # GPU RNN
    if hparams['GPU']:
        encoder_lstm = rnn_gpu(units=hparams['z_size'], return_state=False, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn_gpu(units=hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    else:
        encoder_lstm = rnn(hparams['z_size'], return_state=False, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn(hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')

    # Encoder
    encoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='encoder_input')
    encoder_output = encoder_lstm(encoder_input)

    z_mu = layers.Dense(hparams['z_size'], name='mu')(encoder_output)
    z_sigma = layers.Dense(hparams['z_size'], name='sigma')(encoder_output)

    z = layers.Lambda(sampling)([z_mu, z_sigma])

    # initial state
    init_h = layers.Dense(units=hparams['z_size'], activation='tanh', name='dec_initial_h')(z)
    init_c = layers.Dense(units=hparams['z_size'], activation='tanh', name='dec_initial_c')(z)

    # Decoder
    decoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='decoder_input')
    rnn_output = decoder_lstm(decoder_input, initial_state=[init_h, init_c])

    decoder_output = layers.Dense(features_len)(rnn_output)

    # Model
    vae_model = keras.Model([encoder_input, decoder_input], [decoder_output])
    enc_model = keras.Model(encoder_input, z_mu)
    embed_model = keras.Model(encoder_input, z_mu)

    # compile
    def md_loss_func(y_true, y_pred):
        md_loss = keras.metrics.mse(y_true, y_pred)
        return md_loss

    def kl_loss_func(*args, **kwargs):
        kl_cost = -0.5 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma))
        return kl_cost

    def model_loss():
        md_loss = md_loss_func
        kl_loss = kl_loss_func
        kl_weight = hparams['kl_weight']

        def vae_loss(y_true, y_pred):
            model_loss = kl_weight * kl_loss() + md_loss(y_true, y_pred)
            return model_loss

        return vae_loss

    vae_model.compile(optimizer=hparams['optimizer'], loss=model_loss(), metrics=[md_loss_func, kl_loss_func])
    return vae_model, enc_model, embed_model


#
def peeky_VAE_model(hparams, features_len):
    def sampling(args):
        z_mu, z_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mu)[0], hparams['z_size']), mean=0., stddev=1.)
        return z_mu + K.exp(0.5 * z_sigma) * epsilon

    rnn = layers.LSTM if hparams['rnn_type'] == 'lstm' else layers.GRU
    rnn_gpu = layers.CuDNNLSTM if hparams['rnn_type'] == 'lstm' else layers.CuDNNGRU
    # GPU RNN
    if hparams['GPU']:
        encoder_lstm = rnn_gpu(units=hparams['z_size'], return_state=False, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn_gpu(units=hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')
    else:
        encoder_lstm = rnn(hparams['z_size'], return_state=False, return_sequences=False, name='encoder_lstm')
        decoder_lstm = rnn(hparams['z_size'], return_sequences=True, return_state=False, name='decoder_lstm')

    # Encoder
    encoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='encoder_input')
    encoder_output = encoder_lstm(encoder_input)

    z_mu = layers.Dense(hparams['z_size'], name='mu')(encoder_output)
    z_sigma = layers.Dense(hparams['z_size'], name='sigma')(encoder_output)

    z = layers.Lambda(sampling)([z_mu, z_sigma])

    # initial state
    init_h = layers.Dense(units=hparams['z_size'], activation='tanh', name='dec_initial_h')(z)
    init_c = layers.Dense(units=hparams['z_size'], activation='tanh', name='dec_initial_c')(z)

    # Repeat latent_z
    tile_z = layers.RepeatVector(hparams['seq_length'])(z)

    # Decoder
    decoder_input = keras.Input(shape=(hparams['seq_length'], features_len), name='decoder_input')
    decoder_full_input = layers.Concatenate()([decoder_input, tile_z])

    rnn_output = decoder_lstm(decoder_full_input, initial_state=[init_h, init_c])

    decoder_output = layers.Dense(features_len)(rnn_output)

    # Model
    vae_model = keras.Model([encoder_input, decoder_input], [decoder_output])
    enc_model = keras.Model(encoder_input, z_mu)
    embed_model = keras.Model(encoder_input, z_mu)

    # compile
    def md_loss_func(y_true, y_pred):
        md_loss = keras.metrics.mse(y_true, y_pred)
        return md_loss

    def kl_loss_func(*args, **kwargs):
        kl_cost = -0.5 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma))
        return kl_cost

    def model_loss():
        md_loss = md_loss_func
        kl_loss = kl_loss_func
        kl_weight = hparams['kl_weight']

        def vae_loss(y_true, y_pred):
            model_loss = kl_weight * kl_loss() + md_loss(y_true, y_pred)
            return model_loss

        return vae_loss

    vae_model.compile(optimizer=hparams['optimizer'], loss=model_loss(), metrics=[md_loss_func, kl_loss_func])
    return vae_model, enc_model, embed_model


############model########################


#
def get_shape_normalize_final(df_use, if_scale_y):
    df_use = copy.deepcopy(df_use)
    #
    df_use['mu_x'] = pd.Series([geo.centroid.x for geo in df_use['geometry']])
    df_use['mu_y'] = pd.Series([geo.centroid.y for geo in df_use['geometry']])
    df_use['geometry'] = pd.Series(
        [affinity.translate(geo, -mx, -my) for mx, my, geo in df_use[['mu_x', 'mu_y', 'geometry']].values])
    #
    # df_use['geometry']=pd.Series([affinity.rotate(geo,lon_arc,origin='centroid') for geo,lon_arc in df_use[['geometry','lon_arc']].values])
    # df_use['right_rotate']=pd.Series([geo.minimum_rotated_rectangle.centroid.y<0 for geo in df_use['geometry']])
    # df_use['geometry']=pd.Series([affinity.rotate(geo,180,origin='centroid')  if flag else geo for geo,flag in df_use[['geometry','right_rotate']].values])

    #
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
    helper_print_with_time('get_shape_normalize_2:', ','.join(df_use.columns), sep='')
    return df_use


#
def draw_shape_by_detail_final(dfg, df_detail, oid, isline=False, ispoint=True, isLegend=False, cols=['xs', 'ys']):
    dft = df_detail[df_detail['OBJECTID'] == oid]
    points = int(dft['PID'].max()) + 1
    dfg[dfg['OBJECTID'] == oid].plot()
    if isline:
        xlst = list(dft[cols[0]].values)
        ylst = list(dft[cols[1]].values)
        plt.plot(xlst + xlst[:1], ylst + ylst[:1], color='red', linestyle='--')
    if ispoint:
        for i in range(points):
            if dft[dft['PID'] == i].__len__() == 0:
                continue
            plt.scatter(dft[dft['PID'] == i][cols[0]].values, dft[dft['PID'] == i][cols[1]].values,
                        label=dft[dft['PID'] == i]['PID'].values[0])
    plt.title('OBJECTID={0}'.format(oid))
    if isLegend:
        plt.legend()
    return plt.show()


#
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

    helper_print_with_time('line_features:', ','.join(df_line.columns), sep='')
    return df_line


#
def get_overall_features_final(df_features, df_use):
    dft = copy.deepcopy(df_use)
    #
    dft['Area'] = dft['geometry'].area
    dft['Perimeter'] = dft['geometry'].exterior.length
    dft['Area_convex'] = dft['geometry'].convex_hull.area

    dft['Elongation'] = dft['lon_len'] / dft['short_len']
    dft['Circularity'] = 4 * np.pi * dft['Area'] / dft['Perimeter'] / dft['Perimeter']
    dft['Rectangularity'] = pd.Series(
        [area / geo.minimum_rotated_rectangle.area for area, geo in dft[['Area', 'geometry']].values])
    dft['Convexity'] = dft['Area'] / dft['Area_convex']

    dft = dft[['Elongation', 'Circularity', 'Rectangularity', 'Convexity', 'OBJECTID']]
    #
    dfg = df_features[['OBJECTID', 'l_oa']].groupby(['OBJECTID'], as_index=False)['l_oa'].agg({'MeanRedius': 'mean'})
    #
    df_features = pd.merge(df_features, dft, how='left', on='OBJECTID')
    df_features = pd.merge(df_features, dfg, how='left', on='OBJECTID')
    #
    helper_print_with_time('overall_feature:', ','.join(df_features.columns), sep='')
    return df_features


#
def get_single_features_final(df_detail, k):
    df_detail = copy.deepcopy(df_detail)
    #
    dft = df_detail.groupby(['OBJECTID'], as_index=False)['UID'].agg({'uid_max': 'max'})
    df_detail = pd.merge(df_detail, dft, how='left', on='OBJECTID')
    #
    coor_dic = {int(_id): [a_x, a_y] for _id, a_x, a_y in df_detail[['OID_UID', 'xs', 'ys']].values}
    #
    df_detail['o_x'] = 0
    df_detail['o_y'] = 0
    #
    df_detail.rename(inplace=True, columns={'xs': 'a_x', 'ys': 'a_y'})
    # ()
    df_detail['temp'] = df_detail['UID'] - k
    df_detail['temp'] = pd.Series(
        [oid * 1000 + temp if temp >= 0 else oid * 1000 + uid_max + 1 + temp for uid_max, temp, oid in
         df_detail[['uid_max', 'temp', 'OBJECTID']].values])
    df_detail['b_x'] = pd.Series([coor_dic[key][0] for key in df_detail['temp']])
    df_detail['b_y'] = pd.Series([coor_dic[key][1] for key in df_detail['temp']])
    # ()
    df_detail['temp'] = df_detail['UID'] + k
    df_detail['temp'] = pd.Series(
        [oid * 1000 + temp if temp <= uid_max else oid * 1000 + temp - uid_max - 1 for uid_max, temp, oid in
         df_detail[['uid_max', 'temp', 'OBJECTID']].values])
    df_detail['c_x'] = pd.Series([coor_dic[key][0] for key in df_detail['temp']])
    df_detail['c_y'] = pd.Series([coor_dic[key][1] for key in df_detail['temp']])

    #
    # AB,AC,BC,OA,OB,OC
    df_detail['l_ab'] = pd.Series(
        [cal_euclidean([ax, ay], [bx
, by]) for ax, ay, bx, by in df_detail[['a_x', 'a_y', 'b_x', 'b_y']].values])
    df_detail['l_ac'] = pd.Series(
        [cal_euclidean([ax, ay], [cx, cy]) for ax, ay, cx, cy in df_detail[['a_x', 'a_y', 'c_x', 'c_y']].values])
    df_detail['l_bc'] = pd.Series(
        [cal_euclidean([cx, cy], [bx, by]) for cx, cy, bx, by in df_detail[['c_x', 'c_y', 'b_x', 'b_y']].values])
    df_detail['l_oa'] = pd.Series(
        [cal_euclidean([ax, ay], [ox, oy]) for ax, ay, ox, oy in df_detail[['a_x', 'a_y', 'o_x', 'o_y']].values])
    df_detail['l_ob'] = pd.Series(
        [cal_euclidean([bx, by], [ox, oy]) for bx, by, ox, oy in df_detail[['b_x', 'b_y', 'o_x', 'o_y']].values])
    df_detail['l_oc'] = pd.Series(
        [cal_euclidean([cx, cy], [ox, oy]) for cx, cy, ox, oy in df_detail[['c_x', 'c_y', 'o_x', 'o_y']].values])
    #

    df_detail['arc_ba'] = round(
        1 - np.arctan2(df_detail['a_y'] - df_detail['b_y'], df_detail['a_x'] - df_detail['b_x']) / np.pi, 6)
    df_detail['arc_ac'] = round(
        1 - np.arctan2(df_detail['c_y'] - df_detail['a_y'], df_detail['c_x'] - df_detail['a_x']) / np.pi, 6)
    df_detail['arc_ob'] = round(
        1 - np.arctan2(df_detail['b_y'] - df_detail['o_y'], df_detail['b_x'] - df_detail['o_x']) / np.pi, 6)
    df_detail['arc_oc'] = round(
        1 - np.arctan2(df_detail['c_y'] - df_detail['o_y'], df_detail['c_x'] - df_detail['o_x']) / np.pi, 6)
    # ;
    df_detail['angle_bac'] = pd.Series([(ac - ba - 1) % 2 for ba, ac in df_detail[['arc_ba', 'arc_ac']].values])
    df_detail['angle_boc'] = pd.Series([(oc - ob) % 2 for ob, oc in df_detail[['arc_ob', 'arc_oc']].values])
    #
    df_detail['angle_bac_change'] = pd.Series([(ac - ba) % 2 for ba, ac in df_detail[['arc_ba', 'arc_ac']].values])
    df_detail['angle_bac_change'] = pd.Series(
        [change if change <= 1 else change - 2 for change in df_detail['angle_bac_change']])

    #
    df_detail['rotate_bac'] = pd.Series([angle if angle < 1 else 2 - angle for angle in df_detail['angle_bac']])
    df_detail['rotate_boc'] = pd.Series([angle if angle < 1 else 2 - angle for angle in df_detail['angle_boc']])
    #
    # Area of Tri_ABC
    df_detail['s_abc'] = pd.Series([(-1 if angle < 1 else 1) * cal_area(l1, l2, l3) for l1, l2, l3, angle in
                                    df_detail[['l_ab', 'l_bc', 'l_ac', 'angle_bac']].values])
    # Area of Tri_OBC
    df_detail['s_obc'] = pd.Series([cal_area(l1, l2, l3) for l1, l2, l3 in df_detail[['l_ob', 'l_bc', 'l_oc']].values])
    # of Tri_OBC
    df_detail['c_obc'] = (df_detail['l_ob'] + df_detail['l_oc'] + df_detail['l_bc']) / 3
    df_detail['r_obc'] = df_detail['s_obc'] / df_detail['c_obc']

    #     #
    df_detail['s_abc'] = pd.Series([(-1 if angle < 1 else 1) * feat / area for feat, area, angle in
                                    df_detail[['s_abc', 'Area', 'angle_bac']].values])
    df_detail['l_bc'] = pd.Series([(-1 if angle < 1 else 1) * feat / np.sqrt(area) for feat, area, angle in
                                   df_detail[['l_bc', 'Area', 'angle_bac']].values])
    df_detail['s_obc'] = pd.Series([feat / area for feat, area in df_detail[['s_obc', 'Area']].values])
    df_detail['c_obc'] = pd.Series([feat / np.sqrt(area) for feat, area in df_detail[['c_obc', 'Area']].values])
    df_detail['r_obc'] = pd.Series([feat / np.sqrt(area) for feat, area in df_detail[['r_obc', 'Area']].values])
    df_detail['l_oa'] = pd.Series([feat / np.sqrt(area) for feat, area in df_detail[['l_oa', 'Area']].values])

    cols = ['OBJECTID', 'PID', 'UID', 'OID_UID', 'isBegin', 'isStart', 'Area'
        , 'l_bc', 'l_oa'
        , 'rotate_bac', 'rotate_boc'
        , 'angle_bac_change', 'angle_bac', 'angle_boc'
        , 's_abc', 's_obc', 'c_obc', 'r_obc'
            ]
    df_detail = df_detail[[x for x in cols if x in df_detail.columns]]
    helper_print_with_time('k=', k, '\tsingle_feature:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_multi_features_final(df_detail, df_use, klst):
    #
    df_use['Area'] = df_use['geometry'].area
    df_detail = pd.merge(df_detail, df_use[['OBJECTID', 'Area']], how='left', on='OBJECTID')
    cols = [
        'l_bc'
        #         ,'rotate_bac','rotate_boc','angle_bac'
        , 'angle_bac_change', 's_abc'
        , 's_obc', 'c_obc', 'r_obc', 'angle_boc'
    ]
    #     cols=['l_bc','s_abc','s_obc','c_obc','r_obc','rotate_bac','rotate_boc']
    pDic = {}
    for k in klst:
        dft = get_single_features_final(df_detail, k)
        dft.columns = ['k{0}_{1}'.format(k, x) if x in cols else x for x in dft.columns]
        pDic[k] = dft
    for k in klst:
        newcols = ['k{0}_{1}'.format(k, x) for x in cols if 'k{0}_{1}'.format(k, x) in pDic[k].columns]
        if k == klst[-1]:
            newcols.append('l_oa')
        df_detail = pd.merge(df_detail, pDic[k][['OID_UID'] + newcols], how='left', on='OID_UID')
    helper_print_with_time('multi_feature:', ','.join(df_detail.columns), sep='')
    return df_detail


#
def get_normalize_features_final_ex(df_features, norm_type, col_min, col_max, col_std, col_mean):
    df_features = copy.deepcopy(df_features)
    cols = [x for x in df_features.columns if 'k' in x and 'rotate' not in x] + ['Elongation', 'Circularity',
                                                                                 'Rectangularity', 'Convexity', 'l_oa',
                                                                                 'MeanRedius']
    #     +['l_oa','Area','Perimeter','Elongation','Circularity','MeanRedius']
    #
    df_stat = df_features[cols].describe().transpose()
    for col in cols:
        # col_min,col_max,col_std,col_mean=df_stat.loc[col][['min','max','std','mean']].values
        if norm_type == 'zscore':
            df_features[col] = (df_features[col] - col_mean) / col_std
        elif norm_type == 'minmax':
            df_features[col] = (df_features[col] - col_min) / (col_max - col_min)
    #
    helper_print_with_time('normalize_feature:', ','.join(df_features.columns), sep='')
    return df_features, col_min, col_max, col_std, col_mean


#
def get_normalize_features_final(df_features, norm_type):
    df_features = copy.deepcopy(df_features)
    cols = [x for x in df_features.columns if 'k' in x and 'rotate' not in x] + ['Elongation', 'Circularity',
                                                                                 'Rectangularity', 'Convexity', 'l_oa',
                                                                                 'MeanRedius']
    #     +['l_oa','Area','Perimeter','Elongation','Circularity','MeanRedius']
    #
    df_stat = df_features[cols].describe().transpose()
    for col in cols:
        col_min, col_max, col_std, col_mean = df_stat.loc[col][['min', 'max', 'std', 'mean']].values
        if norm_type == 'zscore':
            df_features[col] = (df_features[col] - col_mean) / col_std
        elif norm_type == 'minmax':
            df_features[col] = (df_features[col] - col_min) / (col_max - col_min)
    #
    helper_print_with_time('normalize_feature:', ','.join(df_features.columns), sep='')
    return df_features