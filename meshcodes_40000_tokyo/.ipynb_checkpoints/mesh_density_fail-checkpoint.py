import time
import pandas as pd
import numpy as np
import datetime as dt
import math as mt
import jismesh.utils as ju
from jpmesh import Angle, Coordinate, FirstMesh, SecondMesh, ThirdMesh, HalfMesh, QuarterMesh, OneEighthMesh, parse_mesh_code
from Mesh200x200 import Mesh

RAW_COLUMNS = ['id', 'time', 'lat', 'lon']

def select_data(input_file, output_file, day, mesh):
    next_day = (dt.datetime.strptime(day, '%Y%m%d') + dt.timedelta(days=1)).strftime('%Y%m%d')
    df = pd.read_csv(input_file, header=None)
    df.columns = RAW_COLUMNS
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time']>=day) & (df['time']<next_day)]
    df = df[(df.lon>=float(mesh.minLon))&(df.lon<=float(mesh.maxLon))&(df.lat>=float(mesh.minLat))&(df.lat<=float(mesh.maxLat))]
    df.sort_values(['id', 'time'], inplace=True) # first sort_values here.
    df.to_csv(output_file, index=False, float_format='%.6f') # with header
    print('select_data is successful', time.ctime())
    
def f(one_user):
    one_user.drop(['id'], axis=1, inplace=True)
    one_user.set_index('time', inplace=True)
    one_user.lat = one_user.lat.ffill().bfill()
    one_user.lon = one_user.lon.ffill().bfill()
    return one_user

def interpo_data(input_file, output_file, day, time_interval):
    next_day = (dt.datetime.strptime(day, '%Y%m%d') + dt.timedelta(days=1)).strftime('%Y%m%d')
    
    raw_data = pd.read_csv(input_file)
    # columns could be omitted here
    # raw_data.sort_values(['id', 'time'], inplace=True) could be omitted here.
    raw_data.set_index('id', inplace=True)
    raw_data.dropna(inplace=True)
    raw_data.drop_duplicates(inplace=True)

    raw_data['time'] = pd.to_datetime(raw_data['time'])
    time_line = pd.DataFrame({'time': pd.date_range(start=day, end=next_day, freq=time_interval)})
    raw_data['flag'] = np.nan
    time_line['lon'] = np.nan
    time_line['lat'] = np.nan
    time_line['flag'] = 1
    interpo_date = raw_data.groupby(raw_data.index).apply(lambda x: pd.concat([x, time_line]))
    interpo_date.reset_index('id', inplace=True)
    interpo_date.sort_values(by=['id', 'time'], inplace=True)
    interpo_gps = interpo_date.groupby(interpo_date.id).apply(f)
    interpo_gps.reset_index(inplace=True)
    interpo_gps.sort_values(by=['id', 'time'], inplace=True)
    interpo_gps.dropna(inplace=True)
    interpo_gps.drop(['flag'], axis=1, inplace=True)
    interpo_gps = interpo_gps.round(6)
    interpo_gps.to_csv(output_file, index=False)
    print('interpo_data is successful', time.ctime())

def grid4th(lon, lat):  # 4次メッシュ(9桁) 8分割x10分割x2分割=160分割
    return (int(mt.floor(lat*240       / 160)) * 10000000 + int(mt.floor((lon-100)*160       / 160)) * 100000 +    # 1次メッシュ
            int(mt.floor(lat*240 % 160 / 20))  * 10000    + int(mt.floor((lon-100)*160 % 160 / 20))  * 1000   +    # 2次メッシュ
            int(mt.floor(lat*240 % 20  / 2))   * 100      + int(mt.floor((lon-100)*160 % 20  / 2))   * 10     +    # 3次メッシュ
            int(mt.floor(lat*240)) % 2         * 2        + int(mt.floor((lon-100)*160)) % 2                  + 1) # 4次メッシュ

def gps2mesh(input_file, output_file):
    interpo_data = pd.read_csv(input_file) # columns = ['id', 'time', 'lat', 'lon']
    interpo_data['meshcode'] = interpo_data.apply(lambda x: int(grid4th(x.lon, x.lat)), axis=1)
    interpo_data = interpo_data[['id', 'time', 'meshcode']]
    interpo_data.sort_values(by=['id', 'time'], inplace=True)
    interpo_data.to_csv(output_file, index=False) # columns = ['id', 'time', 'meshcode']
    print('gps2mesh is successful', time.ctime())

def gps2mesh_ju(input_file, output_file):
    mesh_level = 4 # 500m mesh size
    interpo_data = pd.read_csv(input_file) # columns = ['id', 'time', 'lat', 'lon']
    interpo_data['meshcode'] = interpo_data.apply(lambda x: int(ju.to_meshcode(x.lat, x.lon, mesh_level)), axis=1)
    interpo_data = interpo_data[['id', 'time', 'meshcode']]
    interpo_data.sort_values(by=['id', 'time'], inplace=True)
    interpo_data.to_csv(output_file, index=False) # columns = ['id', 'time', 'meshcode']
    print('gps2mesh_ju is successful', time.ctime())

def gps2mesh_jpmesh(input_file, output_file):
    interpo_data = pd.read_csv(input_file) # columns = ['id', 'time', 'lat', 'lon']
    interpo_data['meshcode'] = interpo_data.apply(lambda x: int(HalfMesh.from_coordinate(Coordinate(lon=Angle.from_degree(x.lon), lat=Angle.from_degree(x.lat)))), axis=1)
    interpo_data = interpo_data[['id', 'time', 'meshcode']]
    interpo_data.sort_values(by=['id', 'time'], inplace=True)
    interpo_data.to_csv(output_file, index=False) # columns = ['id', 'time', 'meshcode']
    print('gps2mesh_jpmesh is successful', time.ctime())
    
def g(x, all_grid):
    x = x.drop(['time'], axis=1)
    x = pd.merge(x, all_grid, how='outer')
    return x
      
def mesh2video(input_file, output_file, mesh):
    all_grid = pd.DataFrame({'grid': range(mesh.lonNum * mesh.latNum)})
    gps2grid = pd.read_csv(input_file) # columns = ['id', 'time', 'meshcode']
    gps2grid['grid'] = gps2grid.apply(lambda x: mesh.meshcode2id[x.meshcode], axis=1)
    gps2grid = gps2grid[['id', 'time', 'grid']]
    
    data_grid = gps2grid.groupby(['time', 'grid']).size().reset_index()
    data_grid = data_grid.groupby('time').apply(g, all_grid=all_grid).reset_index('time')
    data_grid = data_grid.sort_values(by=['time', 'grid'])
    data_grid = data_grid.fillna(0)
    data_grid.columns = ['time', 'grid', 'num']
    grid_npy = data_grid.groupby(['time']).apply(lambda x: x.num.tolist())
    grid_npy = np.array(grid_npy.tolist())
    grid_npy = grid_npy.reshape((-1, mesh.latNum, mesh.lonNum))[:-1, :, :, np.newaxis]
    np.save(output_file, grid_npy)
    print('mesh2video is successful', time.ctime())

def gm(x, all_grid_mesh):
    x = x.drop(['time'], axis=1)
    x = pd.merge(x, all_grid_mesh, how='outer')
    return x

def mesh2density_sort(input_file, output_file, mesh):
    all_grid_mesh = pd.DataFrame({'grid': range(mesh.lonNum * mesh.latNum), 'meshcode':mesh.meshcodes})
    gps2grid = pd.read_csv(input_file) # columns = ['id', 'time', 'meshcode']
    gps2grid['grid'] = gps2grid.apply(lambda x: mesh.meshcode2id[x.meshcode], axis=1)
    data_grid = gps2grid.groupby(['time', 'grid', 'meshcode']).size().reset_index()
    data_grid = data_grid.groupby('time').apply(gm, all_grid_mesh=all_grid_mesh).reset_index('time')
    data_grid = data_grid.sort_values(by=['time', 'grid'])
    data_grid = data_grid.fillna(0)
    data_grid.columns = ['time', 'grid', 'meshcode', 'num']
    data_grid = data_grid[['time', 'meshcode', 'num']]
    data_grid.to_csv(output_file, index=False)
    print('mesh2density with sorted meshcode is successful', time.ctime())

    
if __name__ == '__main__':
    day = '20110311'
    city = 'tokyo'
    size = '500m'
    time_interval = '5min'
    mesh = Mesh('tokyo', '500m', 200, 200)
    
    data_path = '../data/'
    raw_filename = data_path + '20110311.csv' 
    # this is the raw GPS data with id, timestamp, lon, lat
    # 00723644,2011-03-10 23:49:58,35.439067,139.365653

    select_filename = data_path + f'{day}{city}.csv.gz'
    interpo_filename = data_path + f'{day}{city}_interpo{time_interval}.csv.gz'
    mesh_filename = data_path + f'{day}{city}_interpo{time_interval}_{size}_mesh.csv.gz'
    mesh_filename_ju = data_path + f'{day}{city}_interpo{time_interval}_{size}_mesh_ju.csv.gz'
    mesh_filename_jpmesh = data_path + f'{day}{city}_interpo{time_interval}_{size}_mesh_jpmesh.csv.gz'
    density_npyfilename = data_path + f'{day}{city}_interpo{time_interval}_{size}_density_mesh.npy'
    density_csvfilename_sort = data_path + f'{day}{city}_interpo{time_interval}_{size}_density_mesh_sort.csv.gz'
    
#     print('start select_data', time.ctime())
#     select_data(raw_filename, select_filename, day, mesh)
    
#     print('start interpo_data', time.ctime())
#     interpo_data(select_filename, interpo_filename, day, time_interval)
    
#     print('start gps2mesh...', time.ctime())
#     gps2mesh(interpo_filename, mesh_filename)
    
#     print('start gps2mesh_ju...', time.ctime())
#     gps2mesh_ju(interpo_filename, mesh_filename_ju)
    
#     print('start gps2mesh_jpmesh...', time.ctime())
#     gps2mesh_ju(interpo_filename, mesh_filename_jpmesh)
                                                  
    print('start mesh2video from mesh_filename...', time.ctime())
    mesh2video(mesh_filename_jpmesh, density_npyfilename, mesh)

    print('start mesh2density_sort...', time.ctime())
    mesh2density_sort(mesh_filename_ju, density_csvfilename_sort, mesh)
    
    print('all end...', time.ctime())

    # start select_data Tue Feb 22 15:42:24 2022
    # select_data is successful Tue Feb 22 15:45:04 2022

    # start interpo_data Tue Feb 22 15:45:04 2022
    # interpo_data is successful Tue Feb 22 16:23:17 2022

    # start gps2mesh... Tue Feb 22 16:23:17 2022
    # gps2mesh is successful Tue Feb 22 17:10:38 2022

    # start gps2mesh_ju... Tue Feb 22 17:10:38 2022
    # gps2mesh_ju is successful Tue Feb 22 18:20:25 2022

    # start gps2mesh_jpmesh... Tue Feb 22 18:20:25 2022
    # gps2mesh_ju is successful Tue Feb 22 19:30:14 2022

    # start mesh2video from mesh_filename... Tue Feb 22 19:30:14 2022
