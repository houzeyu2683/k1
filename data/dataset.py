

import torch
import PIL.Image, PIL.ImageStat
from torchvision import transforms as kit

class dataset(torch.utils.data.Dataset):

    def __init__(self, table, process=None):

        self.table = table
        self.process = process
        pass

    def __len__(self):

        length = len(self.table)
        return(length)

    def __getitem__(self, index):

        item  = self.table.iloc[index, :]

        if(self.process):

            case = (
                self.process.image(item),
                self.process.target(item)
            )                
            pass
        
        else:

            case = item
            pass

        return(case)

    def get(self, index):

        return self.__getitem__(index)

    pass





##
##  Packages.

# from skimage.feature import hog


##
##
# class image:

#     def __init__(self):

#         return

#     def learn(self, item):

#         image   = PIL.Image.open(item).convert("RGB")
#         pass

#         mu      = (0.5,0.5,0.5)
#         sigma   = (0.5,0.5,0.5)
#         size    = (64, 64)
#         pipeline = [
#             kit.RandomRotation((-60, 60)),
#             kit.Resize(size),
#             kit.ToTensor(),
#             kit.Normalize(mean = mu, std = sigma)
#         ]
#         action = kit.Compose(pipeline)
#         pass

#         image = action(image).type(torch.float)
#         return(image)

#     def guide(self, item):

#         image   = PIL.Image.open(item).convert("RGB")
#         pass

#         mu      = (0.5,0.5,0.5)
#         sigma   = (0.5,0.5,0.5)
#         size    = (64, 64)
#         pipeline = [
#             kit.RandomRotation((-60, 60)),
#             kit.Resize(size),
#             kit.ToTensor(),
#             kit.Normalize(mean = mu, std = sigma)
#         ]
#         action = kit.Compose(pipeline)
#         pass

#         image = action(image).type(torch.float)
#         return(image)

#     pass




# import torch


# class target:

#     def __init__(self):

#         pass

#     def learn(self, item):

#         alpha = int(item)
#         alpha = torch.tensor(alpha, dtype=torch.long)
#         return(alpha)

#     def guide(self, item):

#         alpha = int(item)
#         alpha = torch.tensor(alpha, dtype=torch.long)
#         return(alpha)
    
#     pass