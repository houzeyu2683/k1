
from turtle import forward
import torch
import torchvision
from torch import nn

class constant:

    article = {'size':105542+1, 'dimension':300}
    postal = {"size":352899, 'dimension':300}
    pass

class variable(nn.Module):

    def __init__(self):

        super(variable, self).__init__()
        layer = dict()
        layer['default']     = nn.Sequential(nn.Linear(5, 300), nn.Tanh())
        layer['postal'] = nn.Embedding(constant.postal['size'], constant.postal['dimension'])
        layer['connection']  = nn.Sequential(nn.Linear(300+300, 512), nn.ReLU())
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='(batch size, feature tensor)'):

        v = [self.layer['default'](x[:,:-1]), self.layer['postal'](torch.unsqueeze(x[:, -1].type(torch.LongTensor), 0)).squeeze()]
        v = torch.cat(v, dim=1)
        y = self.layer['connection'](v)
        return(y)

    pass

class sequence(nn.Module):

    def __init__(self):

        super(sequence, self).__init__()
        layer = dict()
        layer['default'] = nn.Sequential(nn.Linear(512, constant.article['size']), nn.Softmax(dim=1))        
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='(batch size, feature tensor)'):

        y = self.layer['default'](x)
        return(y)

class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()
        layer = dict()
        layer['variable'] = variable()
        layer['sequence'] = sequence()
        self.layer = nn.ModuleDict(layer)
        pass
    
    def forward(self, x="(batch size, feature tensor)", device='cpu'):

        self.layer = self.layer.to(device)
        v = self.layer['variable'](x)
        y = self.layer['sequence'](v)
        return(y)

# class sequence(nn.Module):

#     def __init__(self):

#         super(sequence, self).__init__()
#         layer = {}
#         layer['embedding'] = nn.Embedding(constant.article['size'], constant.article['dimension'])
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x="(length, batch)"):

#         y = self.layer['embeddin'](x)
#         return(y)

#     pass

# class prediction(nn.Module):

#     def __init__(self):

#         super(prediction, self).__init__()
#         nn.Sequential(
#             nn.Linear(constant.article['dimension'], constant.article['size']),
#             nn.Softmax(dim=1)
#         )

#         return



# '''
# 影像輸入神經網路模組。
# '''
# class image(nn.Module):

#     def __init__(self):

#         super(image, self).__init__()
#         layer = dict()
#         backbone = torchvision.models.resnet152(True)
#         layer['1'] = nn.Sequential(
#             *[i for i in backbone.children()][:-4], 
#             nn.Flatten(2)
#         )
#         layer['2'] = nn.Sequential(
#             nn.Linear(784, 1024),
#             nn.Sigmoid()
#         )
#         layer['3'] = nn.Sequential(
#             nn.Linear(784, 1024),
#             nn.Sigmoid()
#         )
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='(2:batch,3:channel,224:height,224:width)'):

#         v = dict()
#         v['1'] = self.layer['1'](x).mean(1)
#         v['2'] = self.layer['2'](v['1']).unsqueeze(0)
#         v['3'] = self.layer['3'](v['1']).unsqueeze(0)
#         y = v['2'], v['3']
#         return(y)

#     pass


# '''
# 文本輸入神經網路模組。
# '''
# class mask:

#     def padding(x="(17:length,7:token)", value="padding token value"):

#         y = (x==value).transpose(0,1)
#         y = y.cuda() if(x.is_cuda) else y.cpu()
#         return(y)

#     def sequence(x="(17:length,7:token)", recourse=False):

#         if(not recourse):

#             length = len(x)
#             y = torch.full((length,length), bool(False))
#             y = y.cuda() if(x.is_cuda) else y.cpu()
#             pass

#         else:

#             length = len(x)
#             y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)
#             y = y.cuda() if(x.is_cuda) else y.cpu()
#             pass

#         return(y)

#     pass


# class text(nn.Module):

#     def __init__(self, vocabulary=None):
    
#         super(text, self).__init__()
#         self.vocabulary = vocabulary
#         size = self.vocabulary.size
#         pass

#         layer = dict()
#         layer['1'] = nn.Embedding(size, 512)
#         layer['2'] = nn.Embedding(size, 512)
#         self.layer = nn.ModuleDict(layer)
#         return

#     def position(self, x="(17:length,7:token)"):

#         empty = torch.zeros(x.shape).type(torch.LongTensor)
#         for row in range(len(empty)): empty[row,:]=row
#         y = empty.cuda() if(x.is_cuda) else empty.cpu()
#         return(y)

#     def forward(self, x="(17:length,7:token)"):

#         v = dict()
#         v['1'] = self.layer['1'](x)
#         v['2'] = self.layer['2'](x)
#         y = v['1'] + v['2']
#         return(y)


# '''
# 影像與文本輸入注意力機制模組，
# 影像編碼當作記憶力輸入編碼器。
# '''
# class attention(nn.Module):

#     def __init__(self, vocabulary=None):

#         super(attention, self).__init__()
#         self.vocabulary = vocabulary
#         size = self.vocabulary.size
#         pass

#         layer = dict()
#         layer['1'] = image()
#         layer['2'] = text(vocabulary=self.vocabulary)
#         layer['3'] = nn.LSTM(
#             input_size=512,
#             hidden_size=1024,
#             num_layers=1
#         )
#         layer['4'] = nn.Linear(1024, size)
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x="((2,3,224,224):image, (17,7):text)"):

#         image, text = x
#         pass

#         v = dict()
#         v['1'] = self.layer['1'](image)
#         v['2'] = self.layer['2'](text)
#         v['3'], _ = self.layer['3'](
#             v['2'],
#             v['1']
#         )
#         v['4'] = self.layer['4'](v['3'])
#         y = v['4']
#         return(y)

#     pass


# '''
# 模型封包。
# '''
# class model(nn.Module):

#     def __init__(self, vocabulary=None):

#         super(model, self).__init__()
#         self.vocabulary = vocabulary
#         layer = attention(vocabulary=self.vocabulary)
#         self.layer = layer
#         pass
    
#     def forward(self, x="((2,3,224,224):image, (17,7):text)", device='cpu'):

#         image, text = x
#         pass

#         self.layer = self.layer.to(device)
#         image = image.to(device)
#         text = text.to(device)
#         pass
        
#         y = self.layer(x=(image, text))
#         return(y)

#     def predict(self, x='image', device='cpu', limit=20):

#         if(len(x)!=1): return("the image dimension need (1, channel, heigh, width) shape")
#         self.layer = self.layer.to(device)
#         x = x.to(device)
#         pass

#         generation = torch.full((1, 1), self.vocabulary.index['<start>'], dtype=torch.long).to(device)
#         v = dict()
#         for _ in range(limit-1):
            
#             with torch.no_grad():

#                 v['next probability'] = self.layer(x=(x, generation))[-1,:,:]
#                 v['next prediction']  = v['next probability'].argmax(axis=1).view((1,1))
#                 generation = torch.cat([generation, v['next prediction']], dim=0)
#                 if(v['next prediction'] == self.vocabulary.index['<end>']): break
#                 pass

#             pass

#         y = list(generation.view(-1).cpu().numpy())
#         return(y)
    
#     pass


# def cost(skip=-100):

#     function = torch.nn.CrossEntropyLoss(ignore_index=skip)
#     return(function)

# def optimizer(model=None):
    
#     if(model):
        
#         function = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
#         return(function)

#     return