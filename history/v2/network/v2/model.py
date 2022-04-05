
import torch
import torchvision
from torch import nn

class constant:

    version = '2.0.0'
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

    def forward(self, b='batch', ):

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
        origin = sequence[target][f][:-1,:,:]
        v = self.layer['core(1)'](
            tgt = origin, 
            memory = memory, 
            tgt_mask = mask.sequence(b[target][f][:-1,:], True), 
            memory_mask = None, 
            tgt_key_padding_mask = mask.padding(b[target][f][:-1,:], 0), 
            memory_key_padding_mask = None
        )
        v = self.layer['core(2)'](v) + v
        upgrade = v
        pass

        positive = upgrade, sequence[target][f][1:,:,:], 2*(b[target][f][1:,:]==b[target][f][1:,:])-1
        pass

        likelihood = self.layer['core(3)'](upgrade)
        prediction = torch.cat([p.unsqueeze(1) for p in [i.squeeze(1).argmax(1) for i in likelihood.split(1,1)]], 1)
        hit = 2*(prediction==b[target][f][1:,:])-1
        e = self.layer['fusion'].layer['sequence'].layer['article_code(1)'](prediction)
        p = self.layer['fusion'].layer['sequence'].layer['article_code(2)'](position.encode(prediction))
        negative = upgrade, e+p, hit
        pass

        y = likelihood, prediction, positive, negative
        return(y)

    pass
