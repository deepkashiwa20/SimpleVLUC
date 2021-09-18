import time
import pandas as pd
import numpy as np
import datetime as dt
from Region import Mesh, Mesh1

def select_data(input_file, output_file, day, mesh):
    next_day = (dt.datetime.strptime(day, '%Y%m%d') + dt.timedelta(days=1)).strftime('%Y%m%d')
    df = pd.read_csv(input_file)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df[(df['ts']>=day) & (df['ts']<next_day)]
    df = df[(df.lon>=float(mesh.minLon))&(df.lon<=float(mesh.maxLon))&(df.lat>=float(mesh.minLat))&(df.lat<=float(mesh.maxLat))]
    df.to_csv(output_file, header=False, index=False, float_format='%.6f')
    print('select_data is successful', time.ctime())
    
def f(one_user):
    one_user.drop(['id'], axis=1, inplace=True)
    one_user.set_index('time', inplace=True)
    one_user.lat.interpolate(method='time', inplace=True, limit_direction='both')
    one_user.lon.interpolate(method='time', inplace=True, limit_direction='both')
    return one_user

def interpo_data(input_file, output_file, day, time_interval):
    next_day = (dt.datetime.strptime(day, '%Y%m%d') + dt.timedelta(days=1)).strftime('%Y%m%d')
    
    raw_data = pd.read_csv(input_file, header=None)
    raw_data.columns = ['id', 'time', 'lon', 'lat']
    raw_data.sort_values(['id', 'time'], inplace=True)
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
    interpo_gps.to_csv(output_file, index=False, header=0)
    print('interpo_data is successful', time.ctime())

def gps2grid(input_file, output_file, mesh):
    interpo_data = pd.read_csv(input_file, header=None)
    interpo_data.columns = ['id', 'time', 'lon', 'lat']
    interpo_data = interpo_data[(interpo_data.lon>=float(mesh.minLon))&(interpo_data.lon<=float(mesh.maxLon))&(interpo_data.lat>=float(mesh.minLat))&(interpo_data.lat<=float(mesh.maxLat))]
    interpo_data['i_num'] = np.ceil((float(mesh.maxLat) - interpo_data.lat) / float(mesh.dLat)) - 1
    interpo_data['j_num'] = np.ceil((interpo_data.lon - float(mesh.minLon)) / float(mesh.dLon)) - 1
    interpo_data.loc[interpo_data.i_num == -1, 'i_num'] = 0
    interpo_data.loc[interpo_data.j_num == -1, 'j_num'] = 0
    interpo_data.loc[interpo_data.i_num > (mesh.latNum - 1), 'i_num'] = mesh.latNum - 1
    interpo_data.loc[interpo_data.j_num > (mesh.lonNum - 1), 'j_num'] = mesh.lonNum - 1
    interpo_data['grid'] = interpo_data.i_num * mesh.lonNum + interpo_data.j_num
    interpo_data.grid = interpo_data.grid.astype(int)
    interpo_data.drop(['lat', 'lon', 'i_num', 'j_num'], axis=1, inplace=True)
    interpo_data.sort_values(by=['id', 'time'], inplace=True)
    interpo_data.to_csv(output_file, index=False, header=0)
    print('gps2grid is successful', time.ctime())
    
def gps2mesh(input_file, output_file, mesh):
    interpo_data = pd.read_csv(input_file, header=None)
    interpo_data.columns = ['id', 'time', 'lon', 'lat']
    interpo_data = interpo_data[(interpo_data.lon>=float(mesh.minLon))&(interpo_data.lon<=float(mesh.maxLon))&(interpo_data.lat>=float(mesh.minLat))&(interpo_data.lat<=float(mesh.maxLat))]
    interpo_data['i_num'] = np.ceil((float(mesh.maxLat) - interpo_data.lat) / float(mesh.dLat)) - 1
    interpo_data['j_num'] = np.ceil((interpo_data.lon - float(mesh.minLon)) / float(mesh.dLon)) - 1
    interpo_data.loc[interpo_data.i_num == -1, 'i_num'] = 0
    interpo_data.loc[interpo_data.j_num == -1, 'j_num'] = 0
    interpo_data.loc[interpo_data.i_num > (mesh.latNum - 1), 'i_num'] = mesh.latNum - 1
    interpo_data.loc[interpo_data.j_num > (mesh.lonNum - 1), 'j_num'] = mesh.lonNum - 1
    interpo_data['meshcode'] = interpo_data.apply(lambda x: mesh.id2meshcode[int(x.i_num * mesh.lonNum + x.j_num)], axis=1) # Here is the only difference with gps2grid.
    interpo_data.drop(['lat', 'lon', 'i_num', 'j_num'], axis=1, inplace=True)
    interpo_data.sort_values(by=['id', 'time'], inplace=True)
    interpo_data.to_csv(output_file, index=False, header=0)
    print('gps2mesh is successful', time.ctime())

def g(x, all_grid):
    x = x.drop(['time'], axis=1)
    x = pd.merge(x, all_grid, how='outer')
    return x

def grid2video(input_file, output_file, mesh):
    all_grid = pd.DataFrame({'grid': range(mesh.lonNum * mesh.latNum)})
    gps2grid = pd.read_csv(input_file, header=None)
    gps2grid.columns = ['id', 'time', 'grid']
    
    data_grid = gps2grid.groupby(['time', 'grid']).size().reset_index()
    data_grid = data_grid.groupby('time').apply(g, all_grid=all_grid).reset_index('time')
    data_grid = data_grid.sort_values(by=['time', 'grid'])
    data_grid = data_grid.fillna(0)
    data_grid.columns = ['time', 'grid', 'num']
    grid_npy = data_grid.groupby(['time']).apply(lambda x: x.num.tolist())
    grid_npy = np.array(grid_npy.tolist())
    grid_npy = grid_npy.reshape((-1, mesh.latNum, mesh.lonNum))[:-1, :, :, np.newaxis]
    np.save(output_file, grid_npy)
    print('grid2video is successful', time.ctime())
      
def mesh2video(input_file, output_file, mesh):
    all_grid = pd.DataFrame({'grid': range(mesh.lonNum * mesh.latNum)})
    gps2grid = pd.read_csv(input_file, header=None)
    gps2grid.columns = ['id', 'time', 'meshcode'] # Here is the difference.
    gps2grid['grid'] = gps2grid.apply(lambda x: mesh.meshcode2id[x.meshcode], axis=1) # Here is the difference.
    gps2grid = gps2grid[['id', 'time', 'grid']] # Here is the difference.
    
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

def m(x, all_mesh):
    x = x.drop(['time'], axis=1)
    x = pd.merge(x, all_mesh, how='outer')
    return x

def mesh2density(input_file, output_file, mesh):
    all_mesh = pd.DataFrame({'meshcode': mesh.meshcodes})
    gps2grid = pd.read_csv(input_file, header=None)
    gps2grid.columns = ['id', 'time', 'meshcode'] # Here is the difference.
    
    data_grid = gps2grid.groupby(['time', 'meshcode']).size().reset_index()
    data_grid = data_grid.groupby('time').apply(m, all_mesh=all_mesh).reset_index('time')
    data_grid = data_grid.sort_values(by=['time', 'meshcode'])
    data_grid = data_grid.fillna(0)
    data_grid.columns = ['time', 'meshcode', 'num']
    data_grid.to_csv(output_file, index=False, header=0)
    print('mesh2density is successful', time.ctime())

def gm(x, all_grid_mesh):
    x = x.drop(['time'], axis=1)
    x = pd.merge(x, all_grid_mesh, how='outer')
    return x

def mesh2density_sort(input_file, output_file, mesh):
    all_grid_mesh = pd.DataFrame({'grid': range(mesh.lonNum * mesh.latNum), 'meshcode':mesh.meshcodes})
    gps2grid = pd.read_csv(input_file, header=None)
    gps2grid.columns = ['id', 'time', 'meshcode'] # Here is the difference.
    gps2grid['grid'] = gps2grid.apply(lambda x: mesh.meshcode2id[x.meshcode], axis=1) # Here is the difference.
    
    data_grid = gps2grid.groupby(['time', 'grid', 'meshcode']).size().reset_index()
    data_grid = data_grid.groupby('time').apply(gm, all_grid_mesh=all_grid_mesh).reset_index('time')
    data_grid = data_grid.sort_values(by=['time', 'grid'])
    data_grid = data_grid.fillna(0)
    data_grid.columns = ['time', 'grid', 'meshcode', 'num']
    data_grid = data_grid[['time', 'meshcode', 'num']]
    data_grid.to_csv(output_file, index=False, header=0)
    print('mesh2density with sorted meshcode is successful', time.ctime())
    
if __name__ == '__main__':
    day = '20210601'
    city = 'tokyo'
    size = '500m'
    time_interval = '5min'
    mesh = Mesh1(139.3375, 35.391666666666666, 140.275, 36.016666666666666, 150, 150, size)
    # tmp = Mesh(city, size)   
    # mesh = Mesh1(tmp.minLon, tmp.minLat, tmp.maxLon, tmp.maxLat, tmp.lonNum, tmp.latNum, size)
    
    data_path = '../data/'
    raw_filename = data_path + 'simulation_result_formatted_wgs84.csv.gz' # this is the raw GPS data with id, timestamp, lon, lat
    select_filename = data_path + f'{day}{city}.csv'
    interpo_filename = data_path + f'{day}{city}_interpo{time_interval}.csv'
    grid_filename = data_path + f'{day}{city}_interpo{time_interval}_{size}_grid.csv'
    mesh_filename = data_path + f'{day}{city}_interpo{time_interval}_{size}_mesh.csv'
    density_npyfilename_g = data_path + f'{day}{city}_interpo{time_interval}_{size}_density_grid.npy'
    density_npyfilename_m = data_path + f'{day}{city}_interpo{time_interval}_{size}_density_mesh.npy'
    density_csvfilename = data_path + f'{day}{city}_interpo{time_interval}_{size}_density_mesh.csv'
    density_csvfilename_sort = data_path + f'{day}{city}_interpo{time_interval}_{size}_density_mesh_sort.csv'
    
    print('start select_data', time.ctime())
    select_data(raw_filename, select_filename, day, mesh)
    print('start interpo_data', time.ctime())
    interpo_data(select_filename, interpo_filename, day, time_interval)
    print('start gps2grid...', time.ctime())
    gps2grid(interpo_filename, grid_filename, mesh)
    print('start gps2mesh...', time.ctime())
    gps2mesh(interpo_filename, mesh_filename, mesh)
    print('start grid2video...', time.ctime())
    grid2video(grid_filename, density_npyfilename_g, mesh)
    print('start mesh2video...', time.ctime())
    mesh2video(mesh_filename, density_npyfilename_m, mesh)
    print('start mesh2density...', time.ctime())
    mesh2density(mesh_filename, density_csvfilename, mesh)
    print('start mesh2density_sort...', time.ctime())
    mesh2density_sort(mesh_filename, density_csvfilename_sort, mesh)
    print('all end...', time.ctime())

    # start select_data Sat Sep 18 02:14:45 2021
    # select_data is successful Sat Sep 18 02:16:21 2021
    # start interpo_data Sat Sep 18 02:16:21 2021
    # interpo_data is successful Sat Sep 18 02:28:44 2021
    # start gps2grid... Sat Sep 18 02:28:44 2021
    # gps2grid is successful Sat Sep 18 02:30:01 2021
    # start gps2mesh... Sat Sep 18 02:30:01 2021
    # gps2mesh is successful Sat Sep 18 02:42:21 2021
    # start grid2video... Sat Sep 18 02:42:21 2021
    # grid2video is successful Sat Sep 18 02:42:44 2021
    # start mesh2video... Sat Sep 18 02:42:44 2021
    # mesh2video is successful Sat Sep 18 02:49:49 2021
    # start mesh2density... Sat Sep 18 02:49:49 2021
    # mesh2density is successful Sat Sep 18 02:50:40 2021
    # start mesh2density_sort... Sat Sep 18 02:50:40 2021
    # mesh2density with sorted meshcode is successful Sat Sep 18 02:57:53 2021
    # all end... Sat Sep 18 02:57:53 2021
