import torch
import os
import numpy as np
import random
from PIL import Image
import pandas as pd

#Edit this path accordingly 
label_training='../cropDataset/AVEC2014_DepressionLabels/AVEC2014_DepressionLabels/AVEC2014_DepressionLabels/Training_DepressionLabels/'

#Edit this path accordingly
label_development='../AVEC2014_DepressionLabels/AVEC2014_DepressionLabels/AVEC2014_DepressionLabels/Development_DepressionLabels/'

#Edit this path accordingly
label_testing='../AVEC2014_Labels_Testset/AVEC2014_Labels_Testset/Testing/DepressionLabels/'

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root_path, len_segment = 8, test_mode=False):
        'Initialization'
        self.root_path = root_path
        self.len_segment = len_segment
        self.test_mode = test_mode

        self.image_list = []
        if self.test_mode:
            self._pega_elementos_teste()    
        else:
            self._pega_elementos(self.root_path +'Training')
            self._pega_elementos(self.root_path +'Development')
            
    def _pega_elementos_teste(self):
        imagens = sorted(os.listdir(self.root_path))
        for imagem in imagens:
            self.image_list.append(self.root_path+'/'+imagem)
        self.num_segments = len(self.image_list)//self.len_segment
        self.image_list = self.image_list[0:self.num_segments*self.len_segment]
        self.indice_list = np.arange(0,self.num_segments*self.len_segment,self.len_segment)

    def _pega_elementos(self,caminho):
        atividades = os.listdir(caminho)
        for ativi in atividades:
            users = os.listdir(caminho+'/'+ativi)
            for user in users:
                imagens = sorted(os.listdir(caminho+'/'+ativi+'/'+user))
                for img in imagens:
                    self.image_list.append(caminho+'/'+ativi+'/'+user+'/'+img)
                
    def _get_test_indices(self,index):
        offsets = [int(x+index) for x in range(self.len_segment)]
        return np.array(offsets)
    
    def _get_indices(self,caminho):
        offsets = [int(x+caminho) for x in range(self.len_segment)]
        return np.array(offsets)

    def _getimgs(self,lista_de_imagens):
        images = list()
        for i in range(len(lista_de_imagens)):
            img_seg = Image.open(lista_de_imagens[i])
            images.append(img_seg.resize((224,224),Image.Resampling.BILINEAR))
        return images

    def __getitem__(self, index):
        'Generates one sample of data'

        if self.test_mode:
            posicoes_string = self.root_path.split('/')
            Y = pd.read_csv(label_testing+posicoes_string[-1]+'_Depression.csv',header=None).iloc[0,0]
            indices = self._get_test_indices(self.indice_list[index])
            self.image_list = np.array(self.image_list)
            lista_tensores = []
            for elementos in self.image_list[indices]:
                feature_np = np.load(elementos)
                feature_tensor = torch.from_numpy(feature_np).float()
                lista_tensores.append(feature_tensor)

            return torch.stack(lista_tensores),float(Y)
        else:
            record = self.image_list[index]
            posicoes_string = record.split('/')
            posicoes_string = posicoes_string[:-1]
            if posicoes_string[-3] == 'Training':
                Y=pd.read_csv(label_training+posicoes_string[-1]+'_Depression.csv',header=None).iloc[0,0]
            else:
                Y=pd.read_csv(label_development+posicoes_string[-1]+'_Depression.csv',header=None).iloc[0,0]

            lista_de_indices = self._get_indices(index)
            
            lista_tensores = []
            
            for elementos in lista_de_indices:
                feature_np = np.load(self.image_list[elementos])
                feature_tensor = torch.from_numpy(feature_np).float()
                lista_tensores.append(feature_tensor)
            return torch.stack(lista_tensores),float(Y)
        
    def __len__(self):
        'Denotes the total number of samples'
        if self.test_mode:
            return len(self.indice_list)
        else:
            return len(self.image_list)-(self.len_segment)
