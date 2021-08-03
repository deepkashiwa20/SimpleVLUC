import time
import pandas as pd
import numpy as np
import datetime as dt
from Region import Mesh

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
    print('grid2density is successful', time.ctime())

if __name__ == '__main__':
    day = '20210601'
    city = 'tokyo'
    size = '500m'
    time_interval = '5min'
    mesh = Mesh(city, size)
    
    data_path = '../data/'
    raw_filename = data_path + 'simulation_result_formatted_wgs84.csv.gz' # this is the raw GPS data with id, timestamp, lon, lat
    select_filename = data_path + f'{day}{city}.csv'
    interpo_filename = data_path + f'{day}{city}_interpo{time_interval}.csv'
    grid_filename = data_path + f'{day}{city}_interpo{time_interval}_{size}.csv'
    density_filename = data_path + f'{day}{city}_interpo{time_interval}_{size}_density.npy'
    
    select_data(raw_filename, select_filename, day, mesh)
    interpo_data(select_filename, interpo_filename, day, time_interval)
    gps2grid(interpo_filename, grid_filename, mesh)
    grid2video(grid_filename, density_filename, mesh)
