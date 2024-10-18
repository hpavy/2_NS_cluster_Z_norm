import torch 
from model import pde
import numpy as np
import time


def train(nb_itt, train_loss, test_loss, resample_rate, display, poids,
          inputs_train_data, outputs_train_data, points_pde,
          model, loss, optimizer, X, U, n_pde, X_test_pde,
          X_test_data, U_test_data, n_data, rectangle, device, Re, 
          time_start, f, x_std, y_std, u_mean, v_mean, p_std, t_std, u_std, v_std): 
    nb_it_tot = nb_itt + len(train_loss)
    # Nos datas initiales
    X_train_data = inputs_train_data
    U_train_data = outputs_train_data
    X_train_pde = points_pde
    print(f'--------------------------\nStarting at epoch: {len(train_loss)}\n--------------------------')
    print(f'--------------------------\nStarting at epoch: {len(train_loss)}\n--------------------------', file=f)
    
    for epoch in range(len(train_loss), nb_it_tot) : 
        model.train() # on dit qu'on va entrainer (on a le dropout)

        ## loss du pde
        pred_pde = model(X_train_pde)
        pred_pde1, pred_pde2, pred_pde3 = pde(pred_pde, X_train_pde, Re=Re, 
                                              x_std=x_std, y_std=y_std, u_mean=u_mean, v_mean=v_mean,
                                              p_std=p_std, t_std=t_std, u_std=u_std, v_std=v_std) 
        loss_pde = torch.mean(pred_pde1**2 + pred_pde2**2 + pred_pde**2)  #(MSE) 
        

        # loss des points de data
        pred_data = model(X_train_data)
        loss_data = loss(U_train_data, pred_data)  #(MSE)

        # loss totale
        loss_totale = poids[0]*loss_data + poids[1]*loss_pde
        train_loss.append(loss_totale.item())
        
        # Backpropagation
        loss_totale.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        
        # Pour le test :
        model.eval()
        
        # loss du pde
        test_pde = model(X_test_pde)
        test_pde1, test_pde2, test_pde3 = pde(test_pde, X_test_pde, Re=Re,
                                              x_std=x_std, y_std=y_std, u_mean=u_mean, v_mean=v_mean,
                                              p_std=p_std, t_std=t_std, u_std=u_std, v_std=v_std) 
        loss_test_pde = torch.mean(test_pde1**2 + test_pde2**2 + test_pde**2)  #(MSE) 
        
        # loss de la data
        test_data = model(X_test_data)
        loss_test_data = loss(U_test_data, test_data) #(MSE)

        # loss totale
        loss_test = poids[0]*loss_test_data + poids[1]*loss_test_pde 
        test_loss.append(loss_test.item())

        if ((epoch+1) % display == 0) or (epoch+1 == nb_it_tot):
            print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
            print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :", file=f)
            print(f"Train : loss: {np.mean(train_loss[-display:]):.3e}, data: {loss_data:.3e}, pde: {loss_pde:.3e}")
            print(f"Train : loss: {np.mean(train_loss[-display:]):.3e}, data: {loss_data:.3e}, pde: {loss_pde:.3e}", file=f)
            print(f"Test  : loss: {np.mean(test_loss[-display:]):.3e}, data: {loss_test_data:.3e}, pde: {loss_test_pde:.3e}")
            print(f"Test  : loss: {np.mean(test_loss[-display:]):.3e}, data: {loss_test_data:.3e}, pde: {loss_test_pde:.3e}", file=f)
            print(f"time: {time.time()-time_start:.0f}s")
            print(f"time: {time.time()-time_start:.0f}s", file=f)
            
        if  (epoch <= 4) :
            print(f"Epoch: {epoch+1}/{nb_it_tot}, loss: {train_loss[-1]:.3e}, data: {loss_data:.3e}, pde: {loss_pde:.3e}")
            print(f"Epoch: {epoch+1}/{nb_it_tot}, loss: {train_loss[-1]:.3e}, data: {loss_data:.3e}, pde: {loss_pde:.3e}", file=f)
        
        if (epoch+1) % resample_rate == 0:
            # On resample les points
            
            # Data 
            points_data_train = np.random.choice(len(X), n_data, replace=False)
            X_train_data = torch.from_numpy(X[points_data_train]).requires_grad_().to(device)
            U_train_data = torch.from_numpy(U[points_data_train]).requires_grad_().to(device)
            
            # PDE
            X_train_pde = rectangle.generate_random(n_pde).to(device)
            

            
