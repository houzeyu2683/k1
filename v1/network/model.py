
import torch
from torch import nn
from torch.nn import functional

class argument:

    personality = {
        'p1':[],
        'p2':[],
        'p3':[4, 8],
        'p4':[3, 8],
        'p5':[],
        'p6':[88829, 256]
    }
    behavior = {
        'b1':[13037+2, 100, 1], 
        'b2':[13227+2, 100, 1], 
        'b3':[100+2, 10, 1],
        'b4':[99+2, 10, 1], 
        'b5':[15+2, 1, 1], 
        'b6':[29+2, 3, 1],
        'b7':[29+2, 3, 1], 
        'b8':[50+2, 5, 1],
        'b9':[50+2, 5, 1],
        'b10':[8+2, 1, 1],
        'b11':[8+2, 1, 1],
        'b12':[19+2, 2, 1], 
        'b13':[19+2, 2, 1],
        'b14':[275+2, 27, 1], 
        'b15':[228+2, 22, 1], 
        'b16':[10+2, 1, 1], 
        'b17':[10+2, 1, 1],
        'b18':[5+2, 1, 1], 
        'b19':[5+2, 1, 1], 
        'b20':[56+2, 5, 1], 
        'b21':[56+2, 5, 1],
        'b22':[21+2, 2, 1], 
        'b23':[21+2, 2, 1], 
        'b24':[12149+2, 100, 1],
        'b25':[24736+2, 100, 1],
        'r1':[1, 1, 1],
        'r2':[4, 1, 1], 
    }
    pass

class code:

    def __init__(self, method='hot', device='cpu'):

        self.method = method
        self.device = device
        return

    def convert(self, tensor, level):

        if(self.method=='hot'):

            tensor = functional.one_hot(tensor, level)
            tensor = tensor.type(torch.FloatTensor).to(self.device)
            pass

        return(tensor)

    pass

class memory:

    def __init__(self, shape, device='cpu'):
    
        self.shape = shape
        self.device = device
        return
    
    def reset(self):

        tensor = torch.zeros(self.shape)
        tensor = tensor.type(torch.FloatTensor).to(self.device)
        return(tensor)

    pass

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
        layer["5"] = nn.Sequential(nn.Linear(8+8+8, 256), nn.Tanh())
        layer["6"] = nn.Sequential(nn.Linear(256+256, 128), nn.Tanh())
        self.layer = nn.ModuleDict(layer).to(self.device)
        self.guide = guide
        return

    def forward(self, batch='batch'):
        
        v = [None]
        pass

        x = torch.cat([batch['p1'], batch['p2'], batch['p5']], 1)
        v += [self.layer['1'](x)]
        pass

        v += [self.layer['2'](batch['p3']).squeeze(0)]
        v += [self.layer['3'](batch['p4']).squeeze(0)]
        v += [self.layer['4'](batch['p6']).squeeze(0)]
        pass

        x = torch.cat([v[1],v[2],v[3]], 1)
        v += [self.layer['5'](x)]
        pass

        x = torch.cat([v[4], v[5]], 1)
        v += [self.layer['6'](x)]
        pass

        y = v[6]
        return(y)
    
    pass

class behavior(nn.Module):
    
    def __init__(self, device='cpu'):

        super(behavior, self).__init__()
        self.device = device
        pass

        layer = dict()
        guide = {
            '1-25':"b1-b25 => v[1]-v[25]",
            '26-27':'r1-r2 => v[26]-v[27]'
        }
        pass

        for index, key in enumerate(argument.behavior, 1):

            i, h, l = argument.behavior[key]
            layer[str(index)] = nn.GRU(i, h, l)
            continue

        self.layer = nn.ModuleDict(layer).to(self.device)
        self.guide = guide
        return

    def forward(self, batch):
        
        v = [None]
        for index, key in enumerate(argument.behavior, 1):

            if(key=='r1'):

                i, h, l = argument.behavior[key]
                s = batch['size']
                status = memory(shape=(l, s, h), device=self.device)
                o, _ = self.layer[str(index)](batch[key][0], status.reset())
                v += [o]
                pass

            else:

                i, h, l = argument.behavior[key]
                s = batch['size']
                label = code(method='hot', device=self.device)
                status = memory(shape=(l, s, h), device=self.device)
                o, _ = self.layer[str(index)](label.convert(batch[key][0], i), status.reset())
                v += [o]
                pass
            
            pass
        
        y = torch.cat(v[1:], 2)
        return(y)
    
    pass

def repeat(tensor, size, axis, device='cpu'):
    
    tensor = [tensor.unsqueeze(axis) for _ in range(size)]
    tensor = torch.cat(tensor, axis)
    tensor = tensor.type(torch.FloatTensor).to(device)
    return(tensor)

class model(nn.Module):
    
    def __init__(self, device='cpu'):

        super(model, self).__init__()
        self.device = device
        pass

        layer = dict()
        guide = {
            "p":"batch => v['p']",
            "b":"batch => v['b']",
            'b1-b25':"v['p'],v['b'] => v['b1']-v['b25']",
            'r1-r2':"v['p'],v['b'] => v['r1']-v['r2']"
        }
        pass

        layer['p'] = personality(device=self.device)
        layer['b'] = behavior(device=self.device)
        for key in argument.behavior:

            i, _, _ = argument.behavior[key]
            layer[key] = nn.Sequential(nn.Linear(128+512, i), nn.Tanh())
            continue

        self.layer = nn.ModuleDict(layer).to(self.device)
        self.guide = guide
        return

    def forward(self, batch):

        v = dict()
        v['p'] = self.layer['p'](batch)
        v['b'] = self.layer['b'](batch)
        pass

        v['p'] = repeat(tensor=v['p'], size=max(batch['length']), axis=0, device=self.device)
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
        v = self.forward(batch)
        for k in argument.behavior:

            if(k in ['r1']):
                
                criteria = torch.nn.MSELoss()
                loss[k] = criteria(v[k], batch[k][1])
                pass

            else:

                criteria = torch.nn.CrossEntropyLoss()
                s = v[k].flatten(0,1)
                t = batch[k][1].flatten(0,1)
                pass

                i = t.nonzero().flatten().tolist()
                s = s[i,:]
                t = t[i]
                pass
                
                score = s[:,t].to(self.device)
                target = torch.arange(len(i)).to(self.device)
                loss[k] = criteria(score, target)
                pass

            pass

        loss['total'] = sum([l for _, l in loss.items()])
        return(loss)

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
