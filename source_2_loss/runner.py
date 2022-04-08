import torch
from tqdm import tqdm
#import hyper paramer from config
from . import config


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)


# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def train_epoch(
    models=None,
    optimizers=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
    loss_meter=None,
    score_meter=None,
    epoch = None,
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    total_loss = 0
    module_loss = 0

    models["backbone"].to(device).train()
    models["module"].to(device).train()

    with tqdm(dataloader, desc="Train") as iterator:

        for sample in iterator:
            #print(sample["x"].shape)
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]
            
            optimizers["backbone"].zero_grad()
            optimizers["module"].zero_grad()
            outputs, features = models["backbone"].forward(x)
            target_loss = criterion(outputs, y)
                        
            if epoch >= config.EPOCH_LOSS:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()
                
            
            #print(target_loss.shape)
            
            #print(features[0].shape, features[1].shape, features[2].shape, features[3].shape, len(features))
            # predict loss
            pred_loss = models["module"](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=config.MARGIN)
            loss            = m_backbone_loss + config.WEIGHT * m_module_loss
            total_loss+=loss.cpu().detach().numpy()
            module_loss+=m_module_loss.cpu().detach().numpy()
            
            
            loss.backward()
            optimizers["backbone"].step()
            optimizers["module"].step()

            loss_meter.update(loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({criterion.name: loss_meter.avg})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs, total_loss/n, module_loss/n


def valid_epoch(
    models=None,
    criterion=None,
    metric=None,
    dataloader=None,
    device="cpu",
):

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    
    uncertainty = torch.tensor([]).to(device)
    models["backbone"].to(device).eval()
    models["module"].to(device).eval()

    with tqdm(dataloader, desc="Valid") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]

            with torch.no_grad():
                outputs, features = models["backbone"].forward(x)
                #print(features[0].shape, features[1].shape, features[2].shape, features[3].shape, len(features))
                # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                uncertainty = torch.cat((uncertainty, pred_loss), 0)
                
                target_loss = criterion(outputs, y)

            loss_meter.update(target_loss.cpu().detach().numpy(), n=n)
            score_meter.update(metric(outputs, y).cpu().detach().numpy(), n=n)

            logs.update({criterion.name: loss_meter.avg[0]})
            logs.update({metric.name: score_meter.avg})
            iterator.set_postfix_str(format_logs(logs))
    return logs, uncertainty.cpu()
