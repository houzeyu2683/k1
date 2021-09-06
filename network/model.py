




import torch, torchvision
from torch import nn


class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()
        pass
        
        self.layer = nn.Sequential( 
            *list(torchvision.models.resnet18(True).children())[:-1],
            nn.Flatten(1,-1),
            nn.Linear(in_features=512, out_features=10),
            nn.Softmax(dim=1)
        )
        return

    def forward(self, batch):

        image, target = batch
        score = self.layer(image)
        return(score, target)

    pass

# import torchvision
# from torch import nn
# layer = nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1])
# import torch
# x = torch.randn(size=(1,3,64,64))
# layer(x).shape
# batch[0]
# image['residual'](batch[0])
# import torch
# from torch import nn

# index = torch.randint(low=0, high=100, size=(13, 4))
# target = torch.randint(low=0, high=100, size=(3, 4))
# embed_layer = nn.Embedding(num_embeddings=100, embedding_dim=6)
# embed = embed_layer(index)
# embed.shape

# embed_target = embed_layer(target)


# encoder_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=6, nhead=1), num_layers=1)
# encode = encoder_layer(embed)


# decoder_layer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=6, nhead=1), num_layers=1)
# decode = decoder_layer(embed_target, encode)
# decode.shape

# ##
# ##
# import torch, torchvision, pickle
# import torch.nn as nn


# ##
# ##
# path='SOURCE/PICKLE/VOCABULARY.pickle'
# with open(path, 'rb') as paper:

#     vocabulary = pickle.load(paper)
#     pass


# ##
# ##
# class mask:

#     def encode(text):

#         if(text.is_cuda):

#             device = "cuda"
#             pass

#         else:

#             device = 'cpu'
#             pass

#         length = text.shape[0]
#         mask = torch.zeros((length, length), device=device).type(torch.bool)
#         return mask
    
#     def decode(text):

#         if(text.is_cuda):

#             device = "cuda"
#             pass

#         else:

#             device = 'cpu'
#             pass

#         length = text.shape[0]
#         mask = (torch.triu(torch.ones((length, length), device=device)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def pad(text):

#         mask = (text == vocabulary['<pad>']).transpose(0, 1)      
#         return mask


# ##
# ##
# class model(torch.nn.Module):

#     def __init__(self):
        
#         super(model, self).__init__()
#         pass

#         self.size = {
#             "vocabulary" : len(vocabulary.itos),
#             "embedding" : 256
#         }
#         pass

#         embedding = nn.ModuleDict({
#             "01" : nn.Embedding(self.size['vocabulary'], self.size['embedding'])
#         })

#         encoding = nn.ModuleDict({
#             "02" : nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.size['embedding'], nhead=2), num_layers=4)
#         })

#         sequence = nn.ModuleDict({
#             "03" : nn.GRU(self.size['embedding'], self.size['embedding'], 1),
#             "04" : nn.GRU(self.size['embedding'], self.size['embedding'], 1)
#         })

#         classification = nn.ModuleDict({
#             "05" : nn.Sequential(nn.Linear(self.size['embedding'], 3))
#         })

#         layer = {
#             "embedding": embedding,
#             "encoding" : encoding,
#             "sequence" : sequence,
#             "classification" : classification
#         }
#         self.layer = nn.ModuleDict(layer)
#         pass
    
#     def forward(self, batch):
        
#         ##
#         index, _ = batch

#         ##
#         cell = {}
#         cell['01'] = self.layer['embedding']['01'](index)
#         cell['02'] = self.layer['encoding']['02'](
#             cell['01'],
#             mask.encode(index),
#             mask.pad(index)
#         )
#         cell['03'], memory = self.layer['sequence']['03'](cell['02'])
#         cell['04'], memory = self.layer['sequence']['04'](cell['03'], memory)

#         cell['05'] = self.layer['classification']['05'](cell['04'][-1,:,:])
#         return cell['05']        

        # # cell['01'], _ = self.layer['encoding']['02'](cell['01'].transpose(0,1).unsqueeze(dim=2))
        # # cell['02'] = self.layer['image']['02'](cell['01']).squeeze()
        # # index = torch.as_tensor(cell['02'] * self.size['vocabulary'], dtype=torch.long)
        # # cell['03'] = self.layer['token']['03'](cell['00'])
        # # length = (cell['03'] * (512-3)).int().flatten().tolist()

        # # ##
        # # for column, row in enumerate(length):

        # #     index[0, column] = self.vocabulary['<bos>']
        # #     index[row, column] = self.vocabulary['<eos>']
        # #     index[row+1:, column] = self.vocabulary['<pad>']
        # #     pass 
        
        # ##
        # # cell['04'] = self.layer['token']['04'](
        # #     self.layer['token']['embedding'](index), 
        # #     mask.encode(index), 
        # #     mask.pad(index)            
        # # )

        # ##
        # # text = dictionary.convert(text, vocabulary=self.vocabulary)
        # cell['05'] = self.layer['token']['05'](
        #     self.layer['token']['embedding'](token),
        #     cell['04'],
        #     mask.decode(token),
        #     None,
        #     mask.pad(token, vocabulary=self.vocabulary),
        #     None
        # )
        # cell['06'] = self.layer['token']['06'](cell['05'])
        # return(cell['06'])

    # def convert(self, image, size=128):

    #     ##
    #     if(image.is_cuda):

    #         device = 'cuda'
    #         pass

    #     else:
     
    #         device = 'cpu'
    #         pass
        
    #     ##
    #     cell = {}
    #     cell['00'] = self.layer['image']['00'](image).squeeze()
    #     cell['01'], _ = self.layer['image']['01'](cell['00'].transpose(0,1).unsqueeze(dim=2))
    #     cell['02'] = self.layer['image']['02'](cell['01']).squeeze()
    #     index = torch.as_tensor(cell['02'] * self.size['vocabulary'], dtype=torch.long)
    #     cell['03'] = self.layer['token']['03'](cell['00'])
    #     length = (cell['03'] * (512-3)).int().flatten().tolist()

    #     ##
    #     for column, row in enumerate(length):

    #         index[0, column] = self.vocabulary['<bos>']
    #         index[row, column] = self.vocabulary['<eos>']
    #         index[row+1:, column] = self.vocabulary['<pad>']
    #         pass 
        
    #     ##
    #     cell['04'] = self.layer['token']['04'](
    #         self.layer['token']['embedding'](index), 
    #         mask.encode(index), 
    #         mask.pad(index, vocabulary=self.vocabulary)            
    #     )

    #     batch = len(image)
    #     sequence = torch.ones(1, batch).fill_(self.vocabulary['<bos>']).type(torch.long).to(device)

    #     for _ in range(size):

    #         code = self.layer['token']['05'](
    #             self.layer['token']['embedding'](sequence), 
    #             cell['04'], 
    #             mask.decode(sequence), 
    #             None, 
    #             None
    #         )
    #         probability = self.layer['token']['06'](code.transpose(0, 1)[:, -1])
    #         _, prediction = torch.max(probability, dim = 1)
    #         sequence = torch.cat([sequence, prediction.unsqueeze(dim=0)], dim=0)
    #         pass

    #     output = []
    #     for i in range(batch):

    #         character = "".join([self.vocabulary.itos[token] for token in sequence[:,i]])
    #         character = "InChI=1S/" + character
    #         character = character.replace("<bos>", "").replace("<eos>", "").replace('<pad>', "")
    #         output += [character]
    #         pass

    #     return output

    #     output = []
    #     for item in range(batch):
            
    #         memory = midden['encoder memory'][:,item:item+1,:]

    #     # print("midden['encoder memory']")
    #     # print(midden['encoder memory'].shape)
    #         ##  Generate sequence.
    #         sequence = torch.ones(1, 1).fill_(vocabulary['<bos>']).type(torch.long).to(device)
    #         for i in range(length):

    #             midden['decoder output'] = self.layer['text decoder'](
    #                 self.layer['text to embedding'](sequence), 
    #                 memory, 
    #                 mask.decode(sequence), 
    #                 None, 
    #                 None
    #             )
    #             print("midden['decoder output'] ")
    #             print(midden['decoder output'].shape)
    #             probability = self.layer['text to vacabulary'](midden['decoder output'].transpose(0, 1)[:, -1])
    #             _, prediction = torch.max(probability, dim = 1)
    #             index = prediction.item()
    #             sequence = torch.cat([sequence, torch.ones(1, 1).type_as(midden['image to index 03']).fill_(index)], dim=0)
    #             pass

    #             if index == vocabulary['<eos>']:
                    
    #                 break
            
    #         character = "InChI=1S/" + "".join([vocabulary.itos[tok] for tok in sequence]).replace("<bos>", "").replace("<eos>", "")
    #         output += [character]
    #         pass

    #     return output




# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     src = src.to(device)
#     src_mask = src_mask.to(device)

#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
#     for i in range(max_len-1):
#         memory = memory.to(device)
#         memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)

#         tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
#         print('tgt_mask----')
#         print(tgt_mask)
#         print(tgt_mask.shape)


#         out = model.decode(ys, memory, tgt_mask)
#         out = out.transpose(0, 1)
#         print("output===")
#         print(out.shape)
#         print(out)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim = 1)
#         next_word = next_word.item()

#         ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=0)
#         if next_word == EOS_IDX:
#           break
#     return ys

# def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
#     model.eval()
#     tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)]+ [EOS_IDX]
#     num_tokens = len(tokens)
#     src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
#     return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")

# # batch = torch.randn((8, 3, 224, 224)), torch.randint(0, 141, (10, 8))
# # image, text = batch
# # m = model()
# # x = m(batch)

# # x.shape



# z
# nn.Linear(24, 3)(y[-1,:,:])

# vocabulary = data.process.vocabulary.load(path='SOURCE/PICKLE/VOCABULARY.pickle')

# image = torch.randn((8,3,224,224))

# L01 = nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1])
# L02 = nn.Sequential(nn.GRU(1, 141, 1))
# L03 = nn.Softmax(dim=2)
# L04 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

# M01 = L01(image).squeeze()
# M02, _ = L02(M01.transpose(0,1).unsqueeze(dim=2))
# M03 = L03(M02).argmax(dim=2)

# M04 = (L04(M01) * 512).int().flatten().tolist() # seq length
# for column, row in enumerate(M04):
#     M03[0, column] = vocabulary['<bos>']
#     M03[row, column] = vocabulary['<eos>']
#     M03[row+1:, column] = vocabulary['<pad>']
#     pass 

# L05 = nn.Embedding(141, 256)
# M05 = L05(M03)
# M05.shape

# L06 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=2), num_layers=4)
# M06 = L06(M05, mask.encode(M03), mask.pad(M03, vocabulary=vocabulary))



# def convert(text):

#     output = []
    
#     for item in text:
#         item = [vocabulary['<bos>']] + [vocabulary[i] for i in item] + [vocabulary['<eos>']]
#         item = torch.tensor(item, dtype=torch.long)
#         output += [item]
#     output = torch.nn.utils.rnn.pad_sequence(output, padding_value=vocabulary['<pad>'])
#     return(output)

# text = [["H", "2", "o"], ["H", "2", "o"], ["C", "20"], ["C", "20"], ["C", "20"], ["C", "20"], ["C", "20"], ["H", "2", "o"]]
# L07 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=256, nhead=2), num_layers=4)
# M07 = L07(L05(convert(text)), M06, mask.decode(convert(text)), None, mask.pad(convert(text), vocabulary=vocabulary))

# L08 =  nn.Sequential(nn.Linear(256, 141), nn.Softmax(dim=2))
# M08 =  L08(M07)
# M08


        # ##  Decoder, encode to index of text.
        # midden['decoder output'] = self.layer['text decoder'](
        #     self.layer['text to embedding'](text), 
        #     midden['encoder memory'], 
        #     mask.decode(text), 
        #     None, 
        #     mask.pad(text), 
        #     None
        # )
        # output = self.layer['text to vacabulary'](midden['decoder output'])
        # # print("self.generator(outs)-----")
        # # print(self.generator(outs).shape)
        # return output

