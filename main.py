from deepxrte.geometry import Rectangle
import torch 
import torch.nn as nn 
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from model import PINNs
from utils import read_csv, write_csv
from train import train
from pathlib import Path
import time 
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

time_start = time.time()

############# LES VARIABLES ################

folder_result = '2_full_variables'  # le nom du dossier de résultat


torch.manual_seed(42537)

##### Le modèle de résolution de l'équation de la chaleur
nb_itt = 5000      # le nb d'epoch
resample_rate = 100000  # le taux de resampling
display = 500       # le taux d'affichage
poids = [1, 1]   # les poids pour la loss
    
n_data = 20000         # le nb de points initiaux
n_pde = 20000          # le nb de points pour la pde

n_data_test = 5000
n_pde_test  = 5000

Re = 100

lr = 1e-4

##### Le code ###############################
###############################################

# La data
df = pd.read_csv('data.csv')


# On adimensionne la data
df_modified = df[(df['Points:0']>= -0.07) & (df['Points:1']>= -0.1) & (df['Points:1']<= 0.1)]
x, y, t = np.array(df_modified['Points:0']), np.array(df_modified['Points:1']), np.array(df_modified['Time'])
u, v, p = np.array(df_modified['Velocity:0']), np.array(df_modified['Velocity:1']), np.array(df_modified['Pressure'])
x = x-x.min()
y = y-y.min()

x_norm = (x-x.mean())/x.std()
y_norm = (y-y.mean())/y.std()
t_norm = (t-t.mean())/t.std()
p_norm = (p-p.mean())/p.std()
u_norm = (u-u.mean())/u.std()
v_norm = (v-v.mean())/v.std()



X = np.array([x_norm, y_norm, t_norm], dtype=np.float32).T
U = np.array([u_norm, v_norm, p_norm], dtype=np.float32).T

t_norm_min = t_norm.min()
t_norm_max = t_norm.max()
t_max = t.max()

x_norm_max = x_norm.max()
y_norm_max = y_norm.max()


# On regarde si le dossier existe 
dossier = Path(folder_result)
dossier.mkdir(parents=True, exist_ok=True)

rectangle = Rectangle(x_max = x_norm_max, y_max = y_norm_max,
                      t_min=t_norm_min, t_max=t_norm_max)    # le domaine de résolution


# les points initiaux du train 
# Les points de pde 

### Pour train
points_pde = rectangle.generate_random(n_pde).to(device)   # les points pour la pde
points_data_train = np.random.choice(len(X), n_data, replace=False)
inputs_train_data = torch.from_numpy(X[points_data_train]).requires_grad_().to(device)
outputs_train_data = torch.from_numpy(U[points_data_train]).requires_grad_().to(device)

### Pour test
X_test_pde = rectangle.generate_random(n_pde_test).to(device)
points_coloc_test = np.random.choice(len(X), n_data_test, replace=False)
X_test_data = torch.from_numpy(X[points_coloc_test]).requires_grad_().to(device)
U_test_data = torch.from_numpy(U[points_coloc_test]).requires_grad_().to(device)


# Initialiser le modèle
model = PINNs().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()

# On plot les print dans un fichier texte 
with open(folder_result+'/print.txt', 'a') as f:
    # On regarde si notre modèle n'existe pas déjà 
    if Path(folder_result+'/model_weights.pth').exists() :
        model.load_state_dict(torch.load(folder_result+'/model_weights.pth'))
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        train_loss = read_csv(folder_result+'/train_loss.csv')['0'].to_list()
        test_loss = read_csv(folder_result+'/test_loss.csv')['0'].to_list()
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")
        
    else : 
        print('Nouveau modèle\n', file=f)
        print('Nouveau modèle\n')
        train_loss = []
        test_loss = []


    ######## On entraine le modèle 
    ###############################################
    train(nb_itt=nb_itt, train_loss=train_loss, test_loss=test_loss, resample_rate=resample_rate,
          display=display, poids=poids, inputs_train_data=inputs_train_data,
          outputs_train_data=outputs_train_data, points_pde=points_pde,
          model=model, loss=loss, optimizer=optimizer, X=X, U=U, n_pde=n_pde, X_test_pde=X_test_pde,
          X_test_data=X_test_data, U_test_data=U_test_data, n_data=n_data, rectangle=rectangle,
          device=device, Re=Re, time_start=time_start, f=f,
          u_mean = u.mean(), v_mean = v.mean(), x_std = x.std(), y_std = y.std(),
          t_std = t.std(), u_std = u.std(), v_std = v.std(), p_std = p.std())

    ####### On save le model et les losses
    torch.save(model.state_dict(), folder_result+'/model_weights.pth')
    write_csv(train_loss, folder_result, file_name='/train_loss.csv')
    write_csv(test_loss, folder_result, file_name='/test_loss.csv')
    