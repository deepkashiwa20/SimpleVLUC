Dependencies

tensorflow-gpu >= 1.10.0
keras >=  2.2.0
pip install jismesh


./data 
This is the data foler.

./data/20110301_20110311.npy
Our prediction model takes such kind of 4D tensor (Timestep, Height, Width, Channel) as the input and output.
This training and testing data with .npy format. 
The tensor shape is Timestep=?, Height=80, Width=80, Channel=1.
Each day (24 hours) has 288 timestamps with 5min time interval. Therefore, for 11 days, we have 3168 timesteps in total.
Total data shape: (3168, 80, 80, 1) 
First 10 days as training data: (2880, 80, 80, 1) 
The 3.11 earthquake data as testing data: (288, 80, 80, 1)

./data/simulation_result_formatted_wgs84.csv.gz
The raw gps trajectory data with 4 columns, 'id', 'timestamp', 'lon', 'lat'.
Here we use the Osaragi-sensei's simulated trajectory data as example.

./preprocess
This folder contains the preprocessing program.

./preprocess/Region.py
This is the self-defined Mesh class. 

mesh = Mesh('tokyo', '500m')
meshcode = tokyo_mesh.toJISMesh() # pip install jismesh

./preprocess/gen_density.py
This program takes the raw GPS trajectory data as input, as output (Timestep, Height, Width, Channel) as output.
It runs day by day, therefore, the multi-process can be applied to a list of days.

You can run the preprocess with the following commands (with ./data/simulation_result_formatted_wgs84.csv.gz):

cd preprocess;
python gen_density.py;

./model
This is the model folder.
I prepare two models as follows:

./model/HistoricalAverage.py
Use the eight weekdays from 20110301 to 20110310 to calculate the average value, and use the averaged value as prediction.

You can run the model with the following commands:
cd model
python HistoricalAverage.py

./model/ConvLSTMEncoderDecoder.py
Here is the deep learning model, ConvolutionalLSTM with Encoder-Decoder Architecture. 

You can run the model with the following commands:
cd model
python ConvLSTMEncoderDecoder.py

./model/Param.py
All the common parameters among different models are summarized here.
If you want to change the parameters, you should modify this file.

./model/Metrics.py
This contains four widely used metrics MSE, RMSE, MAE, MAPE.
If you want to change the metrics, you should modify this file.

./model/Region.py
This is the self-defined Mesh class, same with ./preprocess/Region.py

./save
All the prediction results will be automaticall saved here.
