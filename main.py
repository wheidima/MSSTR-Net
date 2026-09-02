import torch
from datasets.AvecDataset import Dataset
import torchvision
import os
import torch.nn as nn
from collections import OrderedDict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch.optim as optim
from model.meuModelo import myModel
import numpy as np
import random

# Configuração de seeds para reprodutibilidade
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#seed = 42
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

#Path to saved features (from ResNet-50)
testing_root = '/home/user/depressao/features/Testing/'

model = myModel()
model.to(DEVICE)

#Path to saved features (from ResNet-50)
avec = Dataset("/home/user/depressao/features/", test_mode=False)

train_loader = torch.utils.data.DataLoader(avec,
        batch_size=64, num_workers=10, 
        pin_memory=True, shuffle=True, drop_last=False)

# Loss e Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Configurações do Loop de Treinamento
num_epochs = 100

# Scheduler para decaimento da taxa de aprendizado (Cosine Annealing)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Variáveis para rastrear as melhores métricas e salvar o melhor modelo
best_mae = float('inf')
best_rmse = float('inf')
best_epoch = 0

print('Training started!')

for epoch in range(1, num_epochs + 1):
    # --- Training Stage ---
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float()
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs.squeeze(1), labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Aplica o decaimento da taxa de aprendizado ao final da época
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    epoch_loss = running_loss / len(train_loader)

    # --- Evaluation Stage ---
    model.eval()
    tasks = os.listdir(testing_root)
    predicoes = []
    label = []

    for task in tasks:
        task_path = os.path.join(testing_root, task)
        if not os.path.isdir(task_path):
            continue
            
        users = os.listdir(task_path)
        for user in users:
            user_path = os.path.join(task_path, user)
            if not os.path.isdir(user_path):
                continue
                
            pred_user = []
            label_user = []
            avec_val = Dataset(user_path, test_mode=True)
            val_loader = torch.utils.data.DataLoader(avec_val,
                batch_size=64, num_workers=10, 
                pin_memory=True, shuffle=False, drop_last=False)
            
            with torch.no_grad():
                for imgs, target in val_loader:
                    imgs = imgs.to(DEVICE)
                    output = model(imgs)
                    resultado = output.detach().cpu().numpy().squeeze()
                    
                    # Garante que 'resultado' seja 1D mesmo se o batch/squeeze retornar escalar
                    resultado = np.atleast_1d(resultado)
                    
                    for valorEst in resultado.flatten():
                        pred_user.append(valorEst)
                    for valorReal in target:
                        val = valorReal.item() if torch.is_tensor(valorReal) else valorReal
                        label_user.append(val)
            
            if len(pred_user) > 0:
                predicoes.append(np.median(pred_user))
                label.append(np.mean(label_user))
            
    mae_value = mean_absolute_error(label, predicoes)
    rmse_value = root_mean_squared_error(label, predicoes)
    
    print(f"Epoch [{epoch}/{num_epochs}] | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} | MAE: {mae_value:.4f} | RMSE: {rmse_value:.4f}")

    # Salva os pesos se o MAE desta época for o melhor já obtido
    if mae_value < best_mae:
        best_mae = mae_value
        best_rmse = rmse_value
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")
        print(f"   --> Melhor modelo salvo na época {epoch}! (MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f})")

print("\nTreinamento finalizado!")
print(f"O melhor modelo foi salvo na época {best_epoch} com MAE = {best_mae:.4f} e RMSE = {best_rmse:.4f}")
