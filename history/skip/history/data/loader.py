


##
##  Packages.
import torch, pickle
from torch.utils.data import DataLoader


##
##  Class for loader of dataset.
class loader:

    def __init__(self, train=None, check=None, exam=None, test=None, batch=32):

        if(train):
            
            self.train = DataLoader(train, batch_size=batch, shuffle=True , drop_last=True)
            pass

        if(check):

            self.check = DataLoader(check , batch_size=batch, shuffle=False, drop_last=False)
            pass

        if(exam):
            
            self.exam = DataLoader(exam, batch_size=batch, shuffle=False , drop_last=False)
            pass

        if(test):

            self.test  = DataLoader(test , batch_size=batch, shuffle=False, drop_last=False)
            pass

        pass

    # def sample(self, collection):

    #     batch = {
    #         'text':[],
    #         "author":[],
    #         "target":[]
    #     }
    #     for _, (text, author, target) in enumerate(collection):

    #         batch['text'] += [torch.tensor(text, dtype=torch.long)]
    #         batch['author'] += [torch.tensor(author, dtype=torch.long)]
    #         batch['target'] += [torch.tensor(target, dtype=torch.long)]
    #         pass

    #     batch['text'] = torch.nn.utils.rnn.pad_sequence(batch['text'], padding_value=self.vocabulary['<pad>'])
    #     batch['author'] = torch.nn.utils.rnn.pad_sequence(batch['author'], padding_value=self.vocabulary['<pad>'])
    #     batch['target'] = torch.tensor(batch['target'])
    #     output = list(batch.values())
    #     return(output)
            # target = torch.tensor(target, dtype=torch.long)            
            # batch['target'] += [target]
            # pass

            # index = [self.vocabulary[i] for i in token]
            # # index = torch.tensor(index, dtype=torch.long)
            # batch['index'] += index

            # point = len(index)
            # batch['point'] += [point]
            # pass

        # batch['index']   = torch.nn.utils.rnn.pad_sequence(batch['index'], padding_value=vocabulary['<pad>'])
        # batch['target']  = torch.tensor(batch['target'])
    #     batch['index'] = torch.tensor(batch['index'], dtype=torch.long)
    #     batch['target'] = torch.tensor(batch['target'], dtype=torch.long)
    #     batch['point'] = torch.tensor(batch['point'][:-1]).cumsum(dim=0)
    #     return(batch['index'], batch['target'], batch['point'])

    # pass


# index = [vocabulary['<bos>']] + [vocabulary[i] for i in token] + [vocabulary['<eos>']]
    # def load(self, what='vocabulary', path=None):

    #     if(what=='vocabulary'):

    #         with open(path, 'rb') as paper:

    #             vocabulary = pickle.load(paper)
    #             pass

    #         self.vocabulary = vocabulary
    #         pass

    #     pass
