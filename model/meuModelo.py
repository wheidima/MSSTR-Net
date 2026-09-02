import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class myModel(nn.Module):
    def __init__(self, feature_dim=512,temp=8):
        super(myModel, self).__init__()
        self.feature_dim = feature_dim
        self.temp = temp
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=feature_dim, kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3,padding=1)
        self.conv4 = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3,padding=1)
        #---------------------------
        self.bn_conv1 = nn.BatchNorm1d(feature_dim)
        self.relu_conv1 = nn.GELU()
        self.bn_conv2 = nn.BatchNorm1d(feature_dim)
        self.relu_conv2 = nn.GELU()
        self.bn_conv3 = nn.BatchNorm1d(feature_dim)
        self.relu_conv3 = nn.GELU()
        self.bn_conv4 = nn.BatchNorm1d(feature_dim)
        self.relu_conv4 = nn.GELU()
        #-------------------------------------
        self.c_proj = nn.Linear(feature_dim, feature_dim)
        self.m_proj = nn.Linear(feature_dim, feature_dim)
        self.fusion_proj1 = nn.Linear(feature_dim,temp)
        self.fusion_proj2 = nn.Linear(feature_dim,temp)
        self.t_proj = nn.Linear(feature_dim, feature_dim)
        self.gamma1 = nn.Parameter(torch.zeros(1)) # Escala aprendível para o resíduo
        self.gamma2 = nn.Parameter(torch.zeros(1)) # Escala aprendível para o resíduo
        self.gamma3 = nn.Parameter(torch.zeros(1)) # Escala aprendível para o resíduo
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # Estagio de geracao de multi-scale features
        x = x.permute(0,2,1) #B,C,T
        x = self.conv1(x) #saida 1
        x = self.bn_conv1(x)
        x = self.relu_conv1(x) #B,C,T
        
        x2 = self.conv2(x) #saida 2
        x2 = self.bn_conv2(x2)
        x2 = self.relu_conv2(x2) #B,C,T

        x3 = self.conv3(x)
        x3 = self.bn_conv3(x3)
        x3 = self.relu_conv3(x3)
        x3 = self.conv4(x3) #saida 3
        x3 = self.bn_conv4(x3)
        x3 = self.relu_conv4(x3) #B,C,T
        #--------------------------------------
        #Estagio de multi-scale attention de features e fusion
        x_short_proj = self.c_proj(x.transpose(1,2)) #B,T,C
        x_f_short_proj = torch.bmm(x_short_proj.transpose(1,2),x3.transpose(1,2))#B,C,C
        scaling = math.sqrt(self.feature_dim)
        attention1 = torch.tanh(x_f_short_proj/scaling) #B,C,C
        out1 = torch.bmm(attention1,x3)
        out1 = x3 + self.gamma1*out1 

        x_med_proj = self.c_proj(x2.transpose(1,2)) #B,T,C
        x_f_med_proj = torch.bmm(x_med_proj.transpose(1,2),x3.transpose(1,2))#B,C,C
        attention2 = torch.tanh(x_f_med_proj/scaling) #B,C,C
        out2 = torch.bmm(attention2,x3)
        out2 = x3 + self.gamma2*out2 

        buf1 = self.fusion_proj1(attention1) #B,C,T
        out12= out1 * buf1

        buf2 = self.fusion_proj2(attention2) #B,C,T
        out21 = out2*buf2

        out = out12 + out21 #B,C,T
        #Estagio de multi-scale attention temporal
        out_temp_proj = self.t_proj(out.transpose(1,2)) #B,T,C
        out_f_temp_proj = torch.bmm(out_temp_proj,out) #B,T,T
        scaling_t = math.sqrt(self.temp)
        attention_t = torch.tanh(out_f_temp_proj/scaling_t) #B,T,T
        out_final = torch.bmm(out,attention_t)
        out_final = out + self.gamma3*out_final

        saida_fea = out_final.detach().cpu().numpy().copy()
        #Regression
        saida = torch.mean(out_final,dim=2) #orig
        saida = self.fc(saida) #orig
        return saida
