from jpmesh import Angle, Coordinate, FirstMesh, SecondMesh, ThirdMesh, HalfMesh , QuarterMesh , OneEighthMesh , parse_mesh_code
import jpmesh

import geojson

import pandas as pd

df = pd.read_csv('./mesh_4_sauravnew_centroid.csv')
df
row_idx = 1
col_ind = 0
data = []
tmp = []
feature_collection = []

Dict = {}

idx = 199

for i, rows in df.iterrows():
    print(i, rows)
    
    coordinate = Coordinate( lon=Angle.from_degree(rows[0]), lat=Angle.from_degree(rows[1]) )
    mesh = HalfMesh.from_coordinate(coordinate)
    
    tmp.append(mesh.code)
    col_ind += 1
    
    mesh_decode = int(mesh.code)
    diff = mesh_decode - int(rows[2])
    
    print(row_idx, col_ind, rows[0], rows[1], int(rows[2]), mesh_decode, diff)
    
    if col_ind == 200:
        print(idx)
        #data.insert(int(idx),tmp)
        data.append(tmp)
        Dict[int(row_idx)] = tmp
        tmp = []
        col_ind = 0
        row_idx +=1
        idx = idx -1
    
    
    
df_mesh = pd.DataFrame(data)
df_mesh

reversed_df =df_mesh.iloc[::-1]
reversed_df

reversed_df.to_csv('./meshcode1_200x200.csv',  sep =',', header = None, index = False)