import torch

checkpoint = torch.load('C:\Proyectos/assetto_corsa_gym\outputs\hsac_1st_try\model/best_lap_time/policy_net.pth')
for key, value in checkpoint.items():
    print(f"{key}: {value.shape}")
print('\n\n')

checkpoint = torch.load('C:\Proyectos/assetto_corsa_gym\outputs\hsac_1st_try\model/best_lap_time/online_q_net.pth')
for key, value in checkpoint.items():
    print(f"{key}: {value.shape}")