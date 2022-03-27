
import torch
import torchvision
from torch import nn

class constant:

    version = '1.0.0'
    embedding = {
        'club_member_status' : 4,
        "fashion_news_frequency" : 5,
        "postal_code" : 352899,
        "article_code" : 105545, 
        'position' : 2000
    }
    pass

class mask:

    def padding(x="(length, batch)", value=0):

        y = (x==value).transpose(0,1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    def sequence(x="(length, batch)", recourse=False):

        length = len(x)
        if(not recourse): y = torch.full((length,length), bool(False))
        if(recourse): y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    pass

class position:

    def encode(x='(length, batch)'):

        y = x.clone()
        for i in range(len(y)): y[i,:] = i
        return(y)

    pass

class vector(nn.Module):

    def __init__(self):

        super(vector, self).__init__()
        layer = dict()
        layer['FN+Active+age(1)'] = nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Dropout(0.2))
        layer['club_member_status(1)'] = nn.Embedding(constant.embedding['club_member_status'], 32)
        layer['fashion_news_frequency(1)'] = nn.Embedding(constant.embedding['fashion_news_frequency'], 32)
        layer["postal_code(1)"] = nn.Embedding(constant.embedding['postal_code'], 128)
        layer["core(1)"] = nn.Sequential(nn.Linear(64+32+32+128, 64), nn.ReLU(), nn.Dropout(0.2))
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, b='batch'):
        
        v = [
            self.layer['FN+Active+age(1)'](torch.cat([b['FN'], b['Active'], b['age']],1)),
            self.layer["club_member_status(1)"](b["club_member_status"]).squeeze(0),
            self.layer["fashion_news_frequency(1)"](b["fashion_news_frequency"]).squeeze(0),
            self.layer["postal_code(1)"](b["postal_code"]).squeeze(0)
        ]
        v = torch.cat(v, 1)
        y = self.layer['core(1)'](v)
        return(y)

    pass

class sequence(nn.Module):

    def __init__(self, ):

        super(sequence, self).__init__()
        layer = dict()
        pass

        layer['article_code(1)'] = nn.Embedding(constant.embedding['article_code'], 256)
        layer['article_code(2)'] = nn.Embedding(constant.embedding['position'], 256)
        encoder = nn.TransformerEncoderLayer(256, 4)
        layer['article_code(3)'] = nn.TransformerEncoder(encoder_layer=encoder, num_layers=1, norm=None)
        pass

        encoder = nn.TransformerEncoderLayer(1, 1)
        layer['price(1)'] = nn.TransformerEncoder(encoder_layer=encoder, num_layers=1, norm=None)
        pass

        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, b='batch'):

        v = dict()
        n, h, f = [], 'history', 'future'
        pass

        n += ["article_code"]
        v[n[0]] = dict({h:None, f:None})
        e = self.layer['article_code(1)'](b[n[0]][h])
        p = self.layer['article_code(2)'](position.encode(b[n[0]][h]))
        v[n[0]][h] = self.layer['article_code(3)'](src=e+p, mask=None, src_key_padding_mask=mask.padding(b[n[0]][h], 0))
        e = self.layer['article_code(1)'](b[n[0]][f])
        p = self.layer['article_code(2)'](position.encode(b[n[0]][f]))
        v[n[0]][f] = e + p
        pass

        n += ["price"]
        v[n[1]] = dict({h:None, f:None})
        e = b[n[1]][h].unsqueeze(-1)
        v[n[1]][h] = self.layer['price(1)'](src=e, mask=None, src_key_padding_mask=mask.padding(b[n[1]][h], 0))
        e = b[n[1]][f].unsqueeze(-1)
        v[n[1]][f] = e
        pass

        y = v
        return(y)
    
    pass

class fusion(nn.Module):

    def __init__(self):

        super(fusion, self).__init__()
        layer = dict()
        layer['vector'] = vector()
        layer['sequence'] = sequence()
        encoder = nn.TransformerEncoderLayer(64+256+1, 1)
        layer['core(1)'] = nn.TransformerEncoder(encoder_layer=encoder, num_layers=1, norm=None)
        layer['core(2)'] = nn.Sequential(nn.Linear(64+256+1, 256), nn.ReLU(), nn.Dropout(0.2))
        layer['core(3)'] = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2))
        self.layer = nn.ModuleDict(layer)    
        return
    
    def forward(self, b='batch'):

        h, _ = 'history', 'future' 
        n = []
        vector = self.layer['vector'](b)
        sequence = self.layer['sequence'](b)
        pass

        target = 'article_code'
        length = len(sequence[target][h])
        v = torch.cat([vector.unsqueeze(0) for _ in range(length)], 0)
        pass

        n += ['article_code']
        v = torch.cat([sequence[n[0]][h], v], 2) 
        pass

        n += ['price']
        v = torch.cat([sequence[n[1]][h], v], 2) 
        pass

        v = self.layer['core(1)'](src=v, mask=None, src_key_padding_mask=mask.padding(b[target][h]))
        v = self.layer['core(2)'](v)
        v = self.layer['core(3)'](v) + v
        memory = v
        pass

        y = vector, sequence, memory
        return(y)

    pass

class suggestion(nn.Module):

    def __init__(self):

        super(suggestion, self).__init__()
        layer = dict()
        layer['fusion'] = fusion()
        decoder = nn.TransformerDecoderLayer(256, 2)
        layer['core(1)'] = nn.TransformerDecoder(decoder_layer=decoder, num_layers=2, norm=None)
        layer['core(2)'] = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2))
        layer['core(3)'] = nn.Sequential(nn.Linear(256, constant.embedding['article_code']), nn.ReLU())
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, b='batch'):

        _, sequence, memory = self.layer['fusion'](b)
        target = 'article_code'
        _, f = 'history', 'future'
        v = self.layer['core(1)'](
            tgt = sequence[target][f][:-1,:,:], 
            memory = memory, 
            tgt_mask = mask.sequence(b[target][f][:-1,:], True), 
            memory_mask = None, 
            tgt_key_padding_mask = mask.padding(b[target][f][:-1,:], 0), 
            memory_key_padding_mask = None
        )
        v = self.layer['core(2)'](v) + v
        s = self.layer['core(3)'](v)
        pass

        r = [i.squeeze(1).argmax(1) for i in s.split(1,1)]
        y = s, r
        return(y)

# class suggestion(nn.Module):

#     def __init__(self):

#         super(suggestion, self).__init__()
#         layer = dict()
#         layer['vector'] = vector()
#         layer['sequence'] = sequence()
#         layer['default(f1)'] = nn.Sequential(nn.Linear(320, 256), nn.ReLU(), nn.Dropout(0.2))
#         encoder = nn.TransformerEncoderLayer(256, 2)
#         layer['default(a1)'] = nn.TransformerEncoder(encoder_layer=encoder, num_layers=2, norm=None)
#         decoder = nn.TransformerDecoderLayer(256, 2)
#         layer['default(a2)'] = nn.TransformerDecoder(decoder_layer=decoder, num_layers=2, norm=None)
#         layer['default(f2)'] = nn.Sequential(nn.Linear(256, constant.embedding['article_code']), nn.ReLU(), nn.Softmax(2))
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, b='batch'):

#         # cache = dict()
#         v = self.layer['vector'](b)
#         s = self.layer['sequence'](b)
#         pass

#         l = range(len(s))
#         v = torch.cat([v.unsqueeze(0) for _ in l], 0)
#         m = self.layer['default(f1)'](torch.cat([s, v], 2))
#         m = self.layer['default(a1)'](m)
#         pass

#         f = b['sequence(article_code)']['future'][:-1,:]
#         f = self.layer['default(a2)'](
#             tgt = self.layer['sequence'].layer['article_code(e1)'](f), 
#             memory = m, 
#             tgt_mask = mask.sequence(f, True), 
#             memory_mask = None, 
#             tgt_key_padding_mask = mask.padding(f, 0), 
#             memory_key_padding_mask = None
#         )
#         pass

#         y = self.layer['default(f2)'](f)
#         return(y)

#     pass

# class model(nn.Module):

#     def __init__(self):

#         super(model, self).__init__()
#         layer = dict()
#         layer['suggestion'] = suggestion()
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='batch'):

#         cache = dict()
#         pass
    
#         cache['day(1)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(2)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(3)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(4)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(5)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(6)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(7)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(8)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(9)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(10)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         cache['day(11)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass
    
#         cache['day(12)'] = self.layer['suggestion'](x)
#         u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
#         x['sequence(price)']['history'] = torch.cat(u, 0)
#         u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
#         x['sequence(article_code)']['history'] = torch.cat(u, 0)
#         pass

#         ##  
#         for s in ['next(price)', 'next(article_code)', 'embedding(article_code)']:

#             l = [
#                 cache['day(1)'][s],
#                 cache['day(2)'][s],
#                 cache['day(3)'][s],
#                 cache['day(4)'][s],
#                 cache['day(5)'][s],
#                 cache['day(6)'][s],
#                 cache['day(7)'][s],
#                 cache['day(8)'][s],
#                 cache['day(9)'][s],
#                 cache['day(10)'][s],
#                 cache['day(11)'][s],
#                 cache['day(12)'][s],                
#             ]
#             cache[s] = torch.stack(l)
#             pass

#         return(cache)

#     pass

# 'day(1)', 'day(2)', 'day(3)', 'day(4)', 'day(5)', 'day(6)', 'day(7)', 'day(8)', 'day(9)', 'day(10)', 'day(11)', 'day(12)'
#         v = self.layer['f1'](x[0].unsqueeze(-1))
#         _, (h, _) = self.layer['r1'](v, x[1])
#         h = h.permute(1,0,2).flatten(1,-1)
#         y = self.layer['f2'](h)  
#         return(y)

#     def status(self):

#         x = [torch.randn((3, 7)), (torch.randn((4, 7, 128)), torch.randn((4, 7, 128)))]
#         y = self.forward(x)
#         status = '1'
#         return(status, y)

#     pass

##  序列模組.
# class sequence(nn.Module):

#     def __init__(self, level=20):

#         super(sequence, self).__init__()
#         self.limit = 2000
#         self.level = level
#         pass

#         if(self.level>1):

#             layer = dict()
#             layer['e1'] = nn.Embedding(self.level, 256)
#             layer['p1'] = nn.Embedding(self.limit, 256)
#             layer['r1'] = nn.LSTM(256, 128, 4)
#             layer['f1'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
#             pass
        
#         else:

#             layer = dict()
#             layer['f1'] = nn.Sequential(nn.Linear(1,256))
#             layer['r1'] = nn.LSTM(256, 128, 4)
#             layer['f2'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
#             pass
        
#         layer['o1'] = nn.Sequential(nn.Linear(512, self.level))
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='[s, (h, c)]'):

#         if(self.level>1):

#             p = x[0].clone()
#             for i in range(len(p)): p[i,:] = i 
#             v = self.layer['e1'](x[0]) + self.layer['p1'](p)
#             _, (h, _) = self.layer['r1'](v, x[1])
#             h = h.permute(1,0,2).flatten(1,-1)
#             y = self.layer['f1'](h)
#             pass
        
#         else:

#             v = self.layer['f1'](x[0].unsqueeze(-1))
#             _, (h, _) = self.layer['r1'](v, x[1])
#             h = h.permute(1,0,2).flatten(1,-1)
#             y = self.layer['f2'](h)              
#             pass

#         return(y)

#     def status(self):

#         if(self.level>1):

#             x = [torch.randint(0, 20, (3, 7)), (torch.randn((4, 7, 128)), torch.randn((4, 7, 128)))]
#             y = self.forward(x)
#             status = '1'
#             pass

#         else:

#             x = [torch.randn((3, 7)), (torch.randn((4, 7, 128)), torch.randn((4, 7, 128)))]
#             y = self.forward(x)
#             status = '1'
#             pass

#         return(status, y)

#     pass

# class model(nn.Module):

#     def __init__(self):

#         super(model, self).__init__()
#         layer = dict()
#         layer['r1'] = row()
#         for i, (_,v) in enumerate(constant.sequence.items(), 1): layer["s"+str(i)] = sequence(level=v)
#         layer['a1'] = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(512, 4), num_layers=4, norm=None
#         )
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='batch'):

#         h, c = self.layer['r1']([x[0],x[1]])
#         s = []
#         for i, (_,_) in enumerate(constant.sequence.items(), 1): 
            
#             s += [self.layer["s"+str(i)]([x[2][i-1][0], (h,c)])]
#             pass

#         a = self.layer['a1'](torch.stack(s))
#         o = []
#         for i, (_,_) in enumerate(constant.sequence.items(), 1): 
            
#             o += [self.layer["s"+str(i)].layer['o1'](a[i-1,:,:]).squeeze()]
#             pass        
        
#         y = o
#         return(y)
