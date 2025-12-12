#imports

from app.era5 import getDataERA5 
from app.aemet import getDataAemet
from datetime import datetime, timedelta
from app.DBmanager import loadIntoDB,getDataFrame
from app.DFmanager import addFeatures,normaliceData,createWindows
import torch
from transformers import AutoModel


#funciones
def setDates(fecha_ini_dt,fecha_fin_dt):
    fecha_ini_AEMET=fecha_ini_dt.strftime("%Y-%m-%d")+"T00:00:00UTC"
    fecha_fin_AEMET=fecha_fin_dt.strftime("%Y-%m-%d")+"T23:59:59UTC"
    dates={"aemet":[fecha_ini_AEMET,fecha_fin_AEMET],"era5":[fecha_ini_dt,fecha_fin_dt]}
    return dates
def loadData(start,end):
    current_start=start
    control=1
    while current_start<=end:
        #gestión del bucle
        current_end= current_start + timedelta(days=15)
        if current_end>end:
            current_end=end
        #obtención y carga de los datos AEMET
        dates= setDates(current_start,current_end)
        data=getDataERA5(dates["era5"][0],dates["era5"][1])
        loadIntoDB(data,control)
        #siguiente fecha
        current_start=current_end+timedelta(days=1)
#variables
start=datetime(2024,1,1)
end=datetime(2024,1,16)
#código
#loadData(start,end)
df=getDataFrame()
df=addFeatures(df)
#Columnas que se utilizaran como input y que deseamos como output
features = ["u10","v10","t2m","d2m","msl","sp","tcwv","cape","blh","latitude","longitude","elevacion_m","wind_speed","wind_dir"]
targets=["wind_speed","wind_dir"]
#Normalización de los datos
X_Scaled,Y_Scaled=normaliceData(df,features,targets)
#Ventanas de tiempo
X_seq,Y_seq=createWindows(X_Scaled,Y_Scaled)
#Entrenamiento
#modelo y optimizador
model=AutoModel.from_pretrained("google/timesfm-1.0-200m")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
#tensores
X_tensor=torch.tensor(X_seq,dtype=torch.floar32)
Y_tensor=torch.tensor(Y_seq,dtype=torch.floar32)
batch_size=16
for epoch in range (10):
    for i in range (0,len(X_tensor),batch_size):
        x_batch=X_tensor[i:i+batch_size]
        y_batch=Y_tensor[i:i+batch_size]

        output=model(x_batch)

        loss=((output-y_batch)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: ",epoch,"Loss: ",loss.item())