import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.metrics import Evaluator

'''
todo 找到了一样的方法,里面的方法名称都一样, 应该是同一个,但不知道他为什么换个名字 
thanks to https://github.com/jakc4103/DFQ/tree/master 
and https://stackoverflow.com/questions/68465208/python-how-to-install-utils-metrics-module
'''


# from utils.metric_tool import SegEvaluator

# Self-Adaptive Weighted Loss Function
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(WeightedBCEWithLogitsLoss, self).__init__()

    def forward(self, input, target):
        # evaluator = SegEvaluator(1)
        evaluator = Evaluator(1)
        evaluator.reset()
        pred = torch.where(torch.sigmoid(input) > 0.5, 1, 0)
        evaluator.add_batch(gt_image=target.cpu().numpy(), pre_image=pred.cpu().numpy())
        w_00, w_11 = evaluator.loss_weight()
        weight1 = torch.zeros_like(target)
        weight1 = torch.fill_(weight1, w_00)
        weight1[target > 0] = w_11
        loss = F.binary_cross_entropy_with_logits(input, target, weight=weight1, reduction="mean")

        return loss

# model = WeightedBCEWithLogitsLoss()
# print(model)
# test_data1 = test_label = torch.randint(0, 2, (2, 1, 256, 256)).cuda()
# test_data2 = test_label = torch.randint(0, 2, (2, 1, 256, 256)).cuda()
# print(model(test_data1, test_data2))
