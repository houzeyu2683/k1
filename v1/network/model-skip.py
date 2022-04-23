
import torch
import torchvision
from torch import nn

class constant:

    version = '1.0.0'
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
        layer['FN+Active+age(1)'] = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Dropout(0.2))
        layer['club_member_status(1)'] = nn.Embedding(4+10, 8)
        layer['fashion_news_frequency(1)'] = nn.Embedding(5+10, 8)
        layer["postal_code(1)"] = nn.Embedding(352899+10, 256)
        layer["core(1)"] = nn.Sequential(nn.Linear(16+8+8+256, 512), nn.ReLU(), nn.Dropout(0.2))
        layer["core(2)"] = nn.Sequential(nn.Linear(512, 512), nn.Tanh(), nn.Dropout(0.2))
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, batch='batch'):
        
        v = [
            self.layer['FN+Active+age(1)'](torch.cat([batch['FN'], batch['Active'], batch['age']],1)),
            self.layer["club_member_status(1)"](batch["club_member_status"]).squeeze(0),
            self.layer["fashion_news_frequency(1)"](batch["fashion_news_frequency"]).squeeze(0),
            self.layer["postal_code(1)"](batch["postal_code"]).squeeze(0)
        ]
        v = torch.cat(v, 1)  ##  (batch, feature)
        v = self.layer['core(1)'](v)
        y = self.layer['core(2)'](v) + v
        return(y)

    pass

class sequence(nn.Module):

    def __init__(self, ):

        super(sequence, self).__init__()
        layer = dict()
        pass

        key = "article_id_code"
        layer['{}(1)'.format(key)] = nn.Embedding(105542+10, 512)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(512, 256, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(256, 128, 1)
        pass

        key = "product_code"
        layer['{}(1)'.format(key)] = nn.Embedding(47224+10, 512)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(512, 256, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(256, 128, 1)
        pass

        key = "prod_name"
        layer['{}(1)'.format(key)] = nn.Embedding(45875+10, 512)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(512, 256, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(256, 256, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(256, 128, 1)
        pass

        key = "product_type_no"
        layer['{}(1)'.format(key)] = nn.Embedding(132+10, 128)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(128, 64, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(64, 32, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(64, 32, 1)
        pass

        key = "product_type_name"
        layer['{}(1)'.format(key)] = nn.Embedding(131+10, 128)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(128, 64, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(64, 32, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(64, 32, 1)
        pass

        key = "product_group_name"
        layer['{}(1)'.format(key)] = nn.Embedding(19+10, 32)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(32, 16, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(16, 8, 1)
        pass

        key = "graphical_appearance_no"
        layer['{}(1)'.format(key)] = nn.Embedding(30+10, 32)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(32, 16, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(16, 8, 1)
        pass

        key = "graphical_appearance_name"
        layer['{}(1)'.format(key)] = nn.Embedding(30+10, 32)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(32, 16, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(16, 16, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(16, 8, 1)
        pass

        key = "colour_group_code"
        layer['{}(1)'.format(key)] = nn.Embedding(50+10, 64)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(64, 32, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(32, 16, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(32, 16, 1)
        pass

        key = "colour_group_name"
        layer['{}(1)'.format(key)] = nn.Embedding(50+10, 64)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(64, 32, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(32, 16, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(32, 16, 1)
        pass

        key = "perceived_colour_value_id"
        layer['{}(1)'.format(key)] = nn.Embedding(8+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass
    
        key = "perceived_colour_value_name"
        layer['{}(1)'.format(key)] = nn.Embedding(8+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass
    
        key = "perceived_colour_master_id"
        layer['{}(1)'.format(key)] = nn.Embedding(20+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass    
    
        key = "perceived_colour_master_name"
        layer['{}(1)'.format(key)] = nn.Embedding(20+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass        
    
        key = "department_no"
        layer['{}(1)'.format(key)] = nn.Embedding(299+10, 128)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(128, 64, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(64, 64, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(64, 32, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(64, 32, 1)
        pass        
    
        key = "department_name"
        layer['{}(1)'.format(key)] = nn.Embedding(250+10, 64)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(64, 32, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(32, 16, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(32, 16, 1)
        pass        
    
        key = "index_code"
        layer['{}(1)'.format(key)] = nn.Embedding(10+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass        
    
        key = "index_name"
        layer['{}(1)'.format(key)] = nn.Embedding(10+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass        
    
        key = "index_group_no"
        layer['{}(1)'.format(key)] = nn.Embedding(5+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass        

        key = "index_group_name"
        layer['{}(1)'.format(key)] = nn.Embedding(5+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass        

        key = "section_no"
        layer['{}(1)'.format(key)] = nn.Embedding(57+10, 64)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(64, 32, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(32, 16, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(32, 16, 1)
        pass    

        key = "section_name"
        layer['{}(1)'.format(key)] = nn.Embedding(56+10, 64)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(64, 32, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(32, 32, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(32, 16, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(32, 16, 1)
        pass    

        key = "garment_group_no"
        layer['{}(1)'.format(key)] = nn.Embedding(21+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass    

        key = "garment_group_name"
        layer['{}(1)'.format(key)] = nn.Embedding(21+10, 16)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(16, 8, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(8, 8, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(8, 4, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(8, 4, 1)
        pass    

        key = "detail_desc"
        layer['{}(1)'.format(key)] = nn.Embedding(43405+10, 256)
        layer['{}(2)'.format(key)] = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        layer['{}(3)'.format(key)] = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.ReLU())
        layer['{}(4)'.format(key)] = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.ReLU())
        layer['{}(5)'.format(key)] = nn.Sequential(nn.Conv1d(128, 128, 7, padding=3), nn.ReLU())
        layer['{}(6)'.format(key)] = nn.Sequential(nn.Conv1d(128, 64, 1), nn.ReLU())
        layer['{}(7)'.format(key)] = nn.GRU(128, 64, 1)
        pass    

        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, batch='batch'):

        y = dict()
        pass

        key = "article_id_code"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "product_code"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "prod_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "product_type_no"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "product_type_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "product_group_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "graphical_appearance_no"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "graphical_appearance_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "colour_group_code"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "colour_group_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "perceived_colour_value_id"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "perceived_colour_value_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "perceived_colour_master_id"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "perceived_colour_master_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "department_no"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "department_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "index_code"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "index_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "index_group_no"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "index_group_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "section_no"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "section_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "garment_group_no"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "garment_group_name"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        key = "detail_desc"
        v1 = self.layer['{}(1)'.format(key)](batch[key]['history']).permute(1,2,0)
        v2 = self.layer['{}(2)'.format(key)](v1)
        v3 = self.layer['{}(3)'.format(key)](v2) + v2
        v4 = self.layer['{}(4)'.format(key)](v3) + v3
        v5 = self.layer['{}(5)'.format(key)](v4) + v4
        v6 = self.layer['{}(6)'.format(key)](v5)
        v5 = v5.permute(2,0,1)
        h = v6[:,:,-1:].permute(2,0,1)
        v7, _ = self.layer['{}(7)'.format(key)](v5, h)
        y.update({key:v7})
        pass

        y = torch.cat([y[k] for k in y], 2) ## (length, batch, embedding:688)
        return(y)
    
    pass

class fusion(nn.Module):

    def __init__(self):

        super(fusion, self).__init__()
        layer = dict()
        pass
        
        key = 'vector'
        layer['{}'.format(key)] = vector()
        key = 'sequence'
        layer['{}'.format(key)] = sequence()
        key = 'memory'
        layer['{}(1)'.format(key)] = nn.GRU(688, 512, 1)
        pass

        self.layer = nn.ModuleDict(layer)    
        return
    
    def forward(self, batch='batch'):

        key = "vector"
        vector = self.layer['{}'.format(key)](batch).unsqueeze(0) ##(length:1, batch, embedding:512)
        pass

        key = 'sequence'
        sequence = self.layer["{}".format(key)](batch)
        pass

        key = 'memory'
        y, _ = self.layer["{}(1)".format(key)](sequence, vector)
        pass

        return(y)

    pass

class suggestion(nn.Module):

    def __init__(self):

        super(suggestion, self).__init__()
        layer = dict()
        pass
        
        key = 'fusion'
        layer["{}".format(key)] = fusion()
        pass
        
        key = "product_code"#'article_id_code'
        layer["{}(1)".format(key)] = nn.Embedding(47224+10, 512)# nn.Embedding(105542+10, 512)
        decoder = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        layer["{}(2)".format(key)] = nn.TransformerDecoder(decoder_layer=decoder, num_layers=2, norm=None)
        layer["{}(3)".format(key)] = nn.Sequential(nn.Linear(512, 105542+10), nn.ReLU())
        pass

        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, batch='batch'):

        memory = self.layer['fusion'](batch)
        pass

        key = "product_code"#'article_id_code'
        token = batch[key]['future'][:-1, :]
        encode = self.layer["{}(2)".format(key)](
            tgt = self.layer["{}(1)".format(key)](token),
            memory = memory, 
            tgt_mask = mask.sequence(token, True),
            memory_mask = None,
            tgt_key_padding_mask = mask.padding(token, 0),
            memory_key_padding_mask = None
        )
        likelihood = self.layer["{}(3)".format(key)](encode)
        pass

        prediction = [v.squeeze(1).argmax(1).tolist() for v in likelihood.split(1,1)]
        y = (likelihood, prediction)
        return(y)

    pass

class model(nn.Module):
    
    def __init__(self):

        super(model, self).__init__()
        layer = dict()
        pass
        
        key = 'suggestion'
        layer["{}".format(key)] = suggestion()
        pass

        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, batch='batch'):

        key = 'suggestion'
        likelihood, prediction = self.layer['{}'.format(key)](batch)
        y = (likelihood, prediction)
        return(y)

    pass