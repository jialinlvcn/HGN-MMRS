from sklearn.metrics import cohen_kappa_score


class AverageMeter:
    def __init__(self, string=""):
        self.name = string
        self.reset()

    def reset(self):
        self.avg = 0
        self.rsum = 0
        self.count = 0
        self.gt = []
        self.pred = []

    def update(self, pred, gt):
        self.gt.extend(gt.tolist())
        self.pred.extend(pred.tolist())
        n = len(gt)
        val = (pred == gt).sum().item()
        self.count += n
        self.rsum += val
        self.avg = self.rsum / self.count * 100

    def __str__(self):
        return rf"the accuracy of {self.name} is {self.avg * 100} \%"

    def cale_kappa(self):
        return cohen_kappa_score(self.gt, self.pred)


class LossMeter:
    def __init__(self, string=""):
        self.name = string
        self.reset()

    def reset(self):
        self.avg = 0
        self.rsum = 0
        self.count = 0

    def update(self, lossrt, n):
        self.count += n
        self.rsum += lossrt.cpu().item()
        self.avg = self.rsum / self.count

    def __str__(self):
        return rf"the accuracy of {self.name} is {self.avg * 100} \%"
