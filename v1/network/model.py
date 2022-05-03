
import torch
from torch import nn
from torch.nn import functional

class argument:
    
    personality = {
        'p1':[],
        'p2':[],
        'p3':[4, 8],
        'p4':[4, 8],
        'p5':[],
        'p6':[352899, 16]
    }
    # behavior = {
    #     'b1':[47224+2, 100, 1], 
    #     'b2':[45875+2, 100, 1], 
    #     'b3':[132+2, 10, 1],
    #     'b4':[131+2, 10, 1], 
    #     'b5':[19+2, 1, 1], 
    #     'b6':[30+2, 3, 1],
    #     'b7':[30+2, 3, 1], 
    #     'b8':[50+2, 5, 1],
    #     'b9':[50+2, 5, 1],
    #     'b10':[8+2, 1, 1],
    #     'b11':[8+2, 1, 1],
    #     'b12':[20+2, 2, 1], 
    #     'b13':[20+2, 2, 1],
    #     'b14':[299+2, 27, 1], 
    #     'b15':[250+2, 22, 1], 
    #     'b16':[10+2, 1, 1], 
    #     'b17':[10+2, 1, 1],
    #     'b18':[5+2, 1, 1], 
    #     'b19':[5+2, 1, 1], 
    #     'b20':[57+2, 5, 1], 
    #     'b21':[56+2, 5, 1],
    #     'b22':[21+2, 2, 1], 
    #     'b23':[21+2, 2, 1], 
    #     'b24':[43405+2, 100, 1],
    #     'b25':[105542+2, 100, 1],
    #     'r1':[1, 1, 1],
    #     'r2':[2+2, 1, 1], 
    # }
    behavior = {
        'b1':[47224+2, 16, 1], 
        'b2':[45875+2, 16, 1], 
        'b3':[132+2, 8, 1],
        'b4':[131+2, 8, 1], 
        'b5':[19+2, 4, 1], 
        'b6':[30+2, 4, 1],
        'b7':[30+2, 4, 1], 
        'b8':[50+2, 4, 1],
        'b9':[50+2, 4, 1],
        'b10':[8+2, 4, 1],
        'b11':[8+2, 4, 1],
        'b12':[20+2, 4, 1], 
        'b13':[20+2, 4, 1],
        'b14':[299+2, 8, 1], 
        'b15':[250+2, 8, 1], 
        'b16':[10+2, 4, 1], 
        'b17':[10+2, 4, 1],
        'b18':[5+2, 4, 1], 
        'b19':[5+2, 4, 1], 
        'b20':[57+2, 4, 1], 
        'b21':[56+2, 4, 1],
        'b22':[21+2, 4, 1], 
        'b23':[21+2, 4, 1], 
        'b24':[43405+2, 16, 1],
        'b25':[105542+2, 64, 1],
        'r1':[1, 1, 1],
        'r2':[2+2, 2, 1], 
    }    
    pass

# class code:

#     def __init__(self, method='hot', device='cpu'):

#         self.method = method
#         self.device = device
#         return

#     def convert(self, tensor, level):

#         if(self.method=='hot'):

#             tensor = functional.one_hot(tensor, level)
#             tensor = tensor.type(torch.FloatTensor).to(self.device)
#             pass

#         return(tensor)

#     pass

# class memory:

#     def __init__(self, shape, device='cpu'):
    
#         self.shape = shape
#         self.device = device
#         return
    
#     def reset(self):

#         tensor = torch.zeros(self.shape)
#         tensor = tensor.type(torch.FloatTensor).to(self.device)
#         return(tensor)

#     pass

class personality(nn.Module):
    
    def __init__(self, device='cpu'):

        super(personality, self).__init__()
        self.device = device
        pass

        layer = dict()
        guide = {
            '1':"p1,p2,p5 => v[1]",
            '2':"p3 => v[2]",
            '3':'p4 => v[3]',
            '4':'p6 => v[4]',
            '5':'v[1],v[2],v[3] => v[5]',
            '6':'v[4],v[5] => [v6]'
        }
        pass

        layer['1'] = nn.Sequential(nn.Linear(3, 8), nn.Tanh())
        pass
        
        c, e = argument.personality['p3']
        layer['2'] = nn.Embedding(c, e)
        pass

        c, e = argument.personality['p4']
        layer['3'] = nn.Embedding(c, e)
        pass
        
        c, e = argument.personality['p6']
        layer["4"] = nn.Embedding(c, e)
        layer["5"] = nn.Sequential(nn.Linear(8+8+8, 64), nn.Tanh())
        layer["6"] = nn.Sequential(nn.Linear(64+16, 128), nn.Tanh())
        self.layer = nn.ModuleDict(layer).to(self.device)
        self.guide = guide
        return

    def forward(self, batch='batch'):
        
        v = dict()
        pass

        x = torch.cat([batch['p1'], batch['p2'], batch['p5']], 1)
        v[1] = self.layer['1'](x)
        pass

        v[2] = self.layer['2'](batch['p3']).squeeze(0)
        v[3] = self.layer['3'](batch['p4']).squeeze(0)
        v[4] = self.layer['4'](batch['p6']).squeeze(0)
        pass

        x = torch.cat([v[1],v[2],v[3]], 1)
        v[5] = self.layer['5'](x)
        pass

        x = torch.cat([v[4], v[5]], 1)
        v[6] = self.layer['6'](x)
        pass

        y = v[6]
        return(y)
    
    pass

def encode(tensor, method='one hot', level=None):
    
    if(method=='one hot' and level!=None):

        tensor = functional.one_hot(tensor, level)
        tensor = tensor.type(torch.FloatTensor)
        pass

    return(tensor)

def reset(shape):
    
    tensor = torch.zeros(shape)
    tensor = tensor.type(torch.FloatTensor)
    return(tensor)

class behavior(nn.Module):
    
    def __init__(self, device='cpu'):

        super(behavior, self).__init__()
        self.device = device
        pass

        layer = dict()
        guide = {
            "layer['1']-layer['25']":"b1-b25 => v[1]-v[25]",
            "layer['26'],layer['27']":'r1-r2 => v[26]-v[27]'
        }
        pass

        for index, key in enumerate(argument.behavior, 1):

            e, h, l = argument.behavior[key]
            layer[str(index)] = nn.ModuleList([nn.Embedding(e, h), nn.GRU(h, h, l)])
            continue

        self.layer = nn.ModuleDict(layer).to(self.device)
        self.guide = guide
        return

    def forward(self, batch):
        
        v = dict()
        for index, key in enumerate(argument.behavior, 1):

            if(key in ['r1']):

                s = batch['size']
                _, h, l = argument.behavior[key]
                status = reset(shape=(l, s, h)).to(self.device)
                o, _ = self.layer[str(index)][1](batch[key][0], status)
                v[index] = o
                pass

            else:

                s = batch['size']
                _, h, l = argument.behavior[key]
                # label = encode(tensor=batch[key][0], method='one hot', level=e).to(self.device)
                label = self.layer[str(index)][0](batch[key][0])
                status = reset(shape=(l, s, h)).to(self.device)
                o, _ = self.layer[str(index)][1](label, status)
                v[index] = o
                pass
            
            pass
        
        y = torch.cat([x for _, x in v.items()], 2)
        return(y)
    
    pass

def repeat(tensor, size, axis):
    
    tensor = [tensor.unsqueeze(axis) for _ in range(size)]
    tensor = torch.cat(tensor, axis)
    tensor = tensor.type(torch.FloatTensor)
    return(tensor)

class model(nn.Module):
    
    def __init__(self, device='cpu'):

        super(model, self).__init__()
        self.device = device
        pass

        layer = dict()
        guide = {
            'layer["p"]':"batch => v['p']",
            'layer["b"]':"batch => v['b']",
            'layer["b1"]-layer["b25"]':"v['p'],v['b'] => v['b1']-v['b25']",
            'layer["r1"],layer["r2"]':"v['p'],v['b'] => v['r1']-v['r2']"
        }
        pass

        layer['p'] = personality(device=self.device)
        layer['b'] = behavior(device=self.device)
        for key in argument.behavior:

            e, _, _ = argument.behavior[key]
            layer[key] = nn.Sequential(nn.Linear(343, e), nn.Tanh())
            continue

        self.layer = nn.ModuleDict(layer).to(self.device)
        self.guide = guide
        return

    def forward(self, batch):

        v = dict()
        v['p'] = self.layer['p'](batch)
        v['b'] = self.layer['b'](batch)
        pass

        s = max(batch['length'])
        v['p'] = repeat(tensor=v['p'], size=s, axis=0).to(self.device)
        x = torch.cat([v['p'], v['b']], 2)
        pass

        for key in self.layer:

            if(key in ['p', 'b']): continue

            v[key] = self.layer[key](x)
            pass

        y = v
        return(y)
    
    def cost(self, batch):

        loss = {}
        value = self.forward(batch)
        for key in argument.behavior:

            if(key in ['r1']):
                
                # _, _, _, weight = argument.behavior[key]
                digit = sample(value[key], batch[key][1], category=False)
                loss[key] = criteria(method='mse').compute(digit)
                pass

            if(key in ['b{}'.format(i) for i in range(1, 26)] + ['r2']):

                # _, _, _, weight = argument.behavior[key]
                digit = sample(value[key], batch[key][1], category=True)
                loss[key] = criteria(method='top-1 max').compute(digit)
                pass

            pass

        loss['total'] = sum([l for _, l in loss.items()])
        return(loss)

def sample(score, target, category=True):
    
    if(category):

        s = score.flatten(0,1)
        t = target.flatten(0,1)
        i = t.nonzero().flatten()
        digit = s[i,:][:,t[i]]
        pass

    else:

        s = score.flatten()
        t = target.flatten()
        i = t.nonzero().flatten()
        digit = s[i], t[i]
        pass

    return(digit)

class criteria:

    def __init__(self, method=['top-1 max', 'mse']):

        self.method = method
        return

    def compute(self, digit):

        if(self.method=='top-1 max'):

            likelihood = functional.softmax(digit, dim=1)
            difference = - (digit.diag().view(-1, 1).expand_as(digit) - digit)
            loss = torch.mean(likelihood * (torch.sigmoid(difference) + torch.sigmoid(digit ** 2)))
            pass

        if(self.method=='mse'):

            score, target = digit
            function = torch.nn.MSELoss()
            loss = function(score, target)
            pass

        return(loss)

    pass

# class metric:
    
#     def __init__(self, limit):

#         self.limit = limit
#         return

#     def compute(self, prediction, target):

#         group = [prediction, target]
#         score = []
#         for prediction, target in zip(group[0], group[1]):

#             top = min(self.limit, len(target))
#             if(top<12): prediction = prediction[:top]
#             if(top==12): target = target[:top]
#             match = [1*(str(p)==str(t)) for p, t in zip(prediction, target)]
#             precision = []
#             for i, _ in enumerate(match):
                
#                 p = sum(match[:i+1]) if(match[i]==1) else 0
#                 precision += [p/(i+1)]
#                 pass

#             score += [sum(precision) / top]
#             pass

#         score = numpy.mean(score)
#         return(score)

#     pass
