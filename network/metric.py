
import numpy

class metric:

    def __init__(self, prediction=[['c', 'c'], ['a', 'b', 'a']], target=[['a', 'c', 'c'], ['a', 'a']]):

        self.prediction = prediction
        self.target = target
        self.top = 12
        self.description = 'MAP@12'
        return

    def evaluate(self):

        score = numpy.mean([self.compute(p, t) for p, t in zip(self.prediction, self.target)])
        return(score) 

    def compute(self, prediction=['a', 'b', 'a'], target=['a', 'a']):
        
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