
import numpy

class metric:

    def __init__(self, prediction=[['c', 'c'], ['a', 'b', 'a']], target=[['a'], ['a']]):

        self.prediction = prediction
        self.target = target
        self.top = 12
        self.description = 'MAP@12'
        return

    def evaluate(self):

        score = numpy.mean([self.compute(p, t) for p, t in zip(self.prediction, self.target)])
        return(score) 

    def compute(self, prediction=['a'], target=['a', 'a']):
        
        if(len(prediction)>self.top): prediction = prediction[:self.top]
        score = 0.0
        hit   = 0.0
        for i, p in enumerate(prediction):

            if p in target and p not in prediction[:i]:

                hit += 1.0
                score += hit / (i+1.0)
                pass

            pass

        if not target: return 0.0
        score = score / min(len(target), self.top)
        return(score)

    pass


# prediction = ['a', 'c', 'b', 'c', 'c', 'c', 'a', 'a', 'd', 'c', 'a', 'a']
# target=['a', 'c', 'b', 'c', 'c', 'c', 'a', 'c', 'd', 'c', 'a', 'a', 'a', 'a']
# target=['a', 'c', 'b', 'c']

# x = [['a', 'c'], ['a1', 'c']]
# y = [['a', 'c'], ['a', 'c']]
# g = (x,y)

# for j,k in zip(x,y):

#     print(j)
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
#             match = [1*(p==t) for p, t in zip(prediction, target)]
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
# metric().evaluate()
    # def apk(actual, predicted, k=10):

    #     """
    #     Computes the average precision at k.
    #     This function computes the average prescision at k between two lists of
    #     items.
    #     Parameters
    #     ----------
    #     actual : list
    #             A list of elements that are to be predicted (order doesn't matter)
    #     predicted : list
    #                 A list of predicted elements (order does matter)
    #     k : int, optional
    #         The maximum number of predicted elements
    #     Returns
    #     -------
    #     score : double
    #             The average precision at k over the input lists
    #     """
    #     if len(predicted)>k: predicted = predicted[:k]
    #     score = 0.0
    #     num_hits = 0.0

    #     for i,p in enumerate(predicted):

    #         if p in actual and p not in predicted[:i]:

    #             num_hits += 1.0
    #             score += num_hits / (i+1.0)
    #             pass

    #         pass

    #     if not actual: return 0.0
    #     score = score / min(len(actual), k)
    #     return(score) 

    # def mapk(actual, predicted, k=10):

    #     """
    #     Computes the mean average precision at k.
    #     This function computes the mean average prescision at k between two lists
    #     of lists of items.
    #     Parameters
    #     ----------
    #     actual : list
    #             A list of lists of elements that are to be predicted 
    #             (order doesn't matter in the lists)
    #     predicted : list
    #                 A list of lists of predicted elements
    #                 (order matters in the lists)
    #     k : int, optional
    #         The maximum number of predicted elements
    #     Returns
    #     -------
    #     score : double
    #             The mean average precision at k over the input lists
    #     """
    #     score = numpy.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    #     return(score) 

    # pass