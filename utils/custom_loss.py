import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from utils.utils import one_hot

class LossConstructor(_WeightedLoss):
    def __init__(self, source_classes, sink_classes, num_classes, source_loss, others_loss,
                 weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='none',
                 confidence=0.0, alpha=1.0,
                 use_cuda=True):
        super(LossConstructor, self).__init__(weight, size_average, reduce, reduction)
        assert len(source_classes)>=1

        self.source_classes = source_classes
        self.sink_classes = sink_classes
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        all_classes = np.arange(num_classes)
        self.other_classes =  [cl for cl in all_classes if cl not in source_classes]
        self.confidence=confidence
        self.alpha = alpha

        # Select source loss:
        if source_loss == "none":
            self.source_loss_fn = empty_loss
        elif source_loss == "neg_ce":
            self.source_loss_fn = neg_ce
        elif source_loss == "ce_sink":
            self.source_loss_fn = ce_sink
        elif source_loss == "logit_dec":
            self.source_loss_fn = logit_dec
        elif source_loss == "bounded_logit_dec":
            self.source_loss_fn = bounded_logit_dec
        elif source_loss == "bounded_logit_source_sink":
            self.source_loss_fn = bounded_logit_source_sink
        elif source_loss == "ce_source_sink":
            self.source_loss_fn = ce_source_sink
        elif source_loss == "lt1":
            self.source_loss_fn = lt1
        elif source_loss == "lt2":
            self.source_loss_fn = lt2
        else:
            raise ValueError()

        # Select others loss:
        if others_loss == "none":
            self.others_loss_fn = empty_loss
        elif others_loss == "ce":
            self.others_loss_fn = ce
        elif others_loss == "logit_inc":
            self.others_loss_fn = logit_inc
        elif others_loss == "bounded_logit_inc":
            self.others_loss_fn = bounded_logit_inc
        elif others_loss == "bounded_logit_inc_sink":
            self.others_loss_fn = bounded_logit_inc_sink
        elif others_loss == "l1":
            self.others_loss_fn = l1
        elif others_loss == "l2":
            self.others_loss_fn = l2
        else:
            raise ValueError()

    def forward(self, perturbed_logit, clean_logit, gt):
        # Consider only sample that are correctly classified
        clean_class = torch.argmax(clean_logit, dim=-1)
        correct_cl_mask = clean_class == gt
        perturbed_logit = perturbed_logit[correct_cl_mask]
        clean_class = clean_class[correct_cl_mask]

        # Sepeare into source and others
        source_classes_idxs = [i in self.source_classes for i in clean_class]
        source_classes_mask = torch.Tensor(source_classes_idxs)==True
        if torch.sum(source_classes_mask)>0:
            perturbed_logit_source = perturbed_logit[source_classes_mask]
            clean_class_source = clean_class[source_classes_mask]
            source_loss = self.source_loss_fn(perturbed_logit=perturbed_logit_source, clean_class=clean_class_source,
                                            source_classes=self.source_classes,
                                            sink_classes=self.sink_classes,
                                            num_classes=self.num_classes,
                                            confidence=self.confidence,
                                            use_cuda=self.use_cuda)
        else:
            source_loss = torch.tensor([])
            if self.use_cuda:
                source_loss = source_loss.cuda()

        other_classes_idxs = (~np.array(source_classes_idxs)).tolist()
        other_classes_mask = torch.Tensor(other_classes_idxs)==True
        if torch.sum(other_classes_mask)>0:
            perturbed_logit_others = perturbed_logit[other_classes_mask]
            clean_class_others = clean_class[other_classes_mask]
            others_loss = self.others_loss_fn(perturbed_logit=perturbed_logit_others, clean_class=clean_class_others,
                                            source_classes=self.source_classes,
                                            sink_classes=self.sink_classes,
                                            num_classes=self.num_classes,
                                            confidence=self.confidence,
                                            use_cuda=self.use_cuda)
        else:
            others_loss = torch.tensor([])
            if self.use_cuda:
                others_loss = others_loss.cuda()

        loss = torch.cat((source_loss, self.alpha*others_loss), 0)
        if len(loss) == 0:
            loss = torch.tensor([0.], requires_grad=True)
        return torch.mean(loss)


def empty_loss(perturbed_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0, use_cuda=False):
    loss = torch.tensor([], requires_grad=True)
    if use_cuda:
        loss = loss.cuda()
    return loss


def ce(perturbed_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0, use_cuda=False):
    if len(sink_classes)>0:
        pass
    loss = F.cross_entropy(perturbed_logit, clean_class,
                            weight=None, ignore_index=-100, reduction='none')
    return loss

def ce_sink(perturbed_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0, use_cuda=False):
    assert len(sink_classes) == 1
    sink_class = torch.ones_like(clean_class)*sink_classes[0]
    loss = F.cross_entropy(perturbed_logit, sink_class,
                            weight=None, ignore_index=-100, reduction='none')
    return loss

def neg_ce(perturbed_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0, use_cuda=False):
    loss = -F.cross_entropy(perturbed_logit, clean_class,
                            weight=None, ignore_index=-100, reduction='none')
    return loss


def logit_dec(perturbed_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()
    loss = (one_hot_labels * perturbed_logit).sum(1)
    return loss


def logit_inc(perturbed_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()
    loss = -(one_hot_labels * perturbed_logit).sum(1)
    return loss


def bounded_logit_dec(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()

    class_logits = (one_hot_labels * perturbed_logit).sum(1)
    not_class_logits = ((1. - one_hot_labels) * perturbed_logit - one_hot_labels * 10000.).max(1)[0]
    loss = torch.clamp(class_logits - not_class_logits, min=-confidence)
    return loss


def bounded_logit_inc(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()
    class_logits = (one_hot_labels * perturbed_logit).sum(1)
    not_class_logits = ((1. - one_hot_labels) * perturbed_logit - one_hot_labels * 10000.).max(1)[0]
    loss = torch.clamp(not_class_logits - class_logits, min=-confidence)
    return loss


def bounded_logit_source_sink(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()

    loss = torch.tensor([])
    if use_cuda:
        loss = loss.cuda()

    for source_cl, sink_cl in zip(source_classes, sink_classes):
        # Filter all idxs which belong to the source class
        source_cl_idxs = [i == source_cl for i in clean_class]
        source_cl_mask = torch.Tensor(source_cl_idxs)==True
        if torch.sum(source_cl_mask)>0:
            clean_class_source_cl = clean_class[source_cl_mask]
            one_hot_labels_source_cl = one_hot_labels[source_cl_mask]
            perturbed_logit_source_cl = perturbed_logit[source_cl_mask]

            # source loss: Decrease the Source part
            class_logits_source_cl = (one_hot_labels_source_cl * perturbed_logit_source_cl).sum(1)
            not_class_logits_source_cl = ((1. - one_hot_labels_source_cl) * perturbed_logit_source_cl - one_hot_labels_source_cl * 10000.).max(1)[0].detach()
            # source_cl_loss = torch.clamp(class_logits_source_cl - not_class_logits_source_cl, min=-confidence)
            source_cl_loss = torch.clamp(class_logits_source_cl - not_class_logits_source_cl, min=0)

            # sink loss: Increase the Sink part
            target_sink_class = torch.ones_like(clean_class_source_cl) * sink_cl
            one_hot_labels_sink_cl = one_hot(target_sink_class.cpu(), num_classes=num_classes)
            if use_cuda:
                one_hot_labels_sink_cl = one_hot_labels_sink_cl.cuda()
            class_logits_sink_cl = (one_hot_labels_sink_cl * perturbed_logit_source_cl).sum(1)
            not_class_logits_sink_cl = ((1. - one_hot_labels_sink_cl) * perturbed_logit_source_cl - one_hot_labels_sink_cl * 10000.).max(1)[0].detach()
            sink_cl_loss = torch.clamp(not_class_logits_sink_cl - class_logits_sink_cl, min=-confidence)

            loss_source_cl = (source_cl_loss + sink_cl_loss)/2.

            loss = torch.cat((loss, loss_source_cl), 0)

    assert len(loss) == len(clean_class) # Can be deleted after a few tries

    return loss
    
def ce_source_sink(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0, use_cuda=False):
    assert len(sink_classes) == 1
    sink_class = torch.ones_like(clean_class)*sink_classes[0]
    inc_sink_loss = F.cross_entropy(perturbed_logit, sink_class,
                            weight=None, ignore_index=-100, reduction='none')
    
    dec_source_loss = -F.cross_entropy(perturbed_logit, clean_class,
                            weight=None, ignore_index=-100, reduction='none')
    
    loss = (inc_sink_loss + dec_source_loss)/2.
    
    return loss


def lt1(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()

    loss = torch.tensor([])
    if use_cuda:
        loss = loss.cuda()

    for source_cl, sink_cl in zip(source_classes, sink_classes):
        # Filter all idxs which belong to the source class
        source_cl_idxs = [i == source_cl for i in clean_class]
        source_cl_mask = torch.Tensor(source_cl_idxs)==True
        if torch.sum(source_cl_mask)>0:
            clean_class_source_cl = clean_class[source_cl_mask]
            one_hot_labels_source_cl = one_hot_labels[source_cl_mask]
            perturbed_logit_source_cl = perturbed_logit[source_cl_mask]

            # source loss: Decrease the Source part
            class_logits_source_cl = (one_hot_labels_source_cl * perturbed_logit_source_cl).sum(1)
            not_class_logits_source_cl = ((1. - one_hot_labels_source_cl) * perturbed_logit_source_cl - one_hot_labels_source_cl * 10000.).max(1)[0].detach()
            # source_cl_loss = torch.clamp(class_logits_source_cl - not_class_logits_source_cl, min=-confidence)
            source_cl_loss = torch.clamp(class_logits_source_cl - not_class_logits_source_cl, min=0)

            loss_source_cl = source_cl_loss

            loss = torch.cat((loss, loss_source_cl), 0)

    assert len(loss) == len(clean_class) # Can be deleted after a few tries

    return loss

def lt2(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()

    loss = torch.tensor([])
    if use_cuda:
        loss = loss.cuda()

    for source_cl, sink_cl in zip(source_classes, sink_classes):
        # Filter all idxs which belong to the source class
        source_cl_idxs = [i == source_cl for i in clean_class]
        source_cl_mask = torch.Tensor(source_cl_idxs)==True
        if torch.sum(source_cl_mask)>0:
            clean_class_source_cl = clean_class[source_cl_mask]
            one_hot_labels_source_cl = one_hot_labels[source_cl_mask]
            perturbed_logit_source_cl = perturbed_logit[source_cl_mask]

            # sink loss: Increase the Sink part
            target_sink_class = torch.ones_like(clean_class_source_cl) * sink_cl
            one_hot_labels_sink_cl = one_hot(target_sink_class.cpu(), num_classes=num_classes)
            if use_cuda:
                one_hot_labels_sink_cl = one_hot_labels_sink_cl.cuda()
            class_logits_sink_cl = (one_hot_labels_sink_cl * perturbed_logit_source_cl).sum(1)
            not_class_logits_sink_cl = ((1. - one_hot_labels_sink_cl) * perturbed_logit_source_cl - one_hot_labels_sink_cl * 10000.).max(1)[0].detach()
            sink_cl_loss = torch.clamp(not_class_logits_sink_cl - class_logits_sink_cl, min=-confidence)

            loss_source_cl = sink_cl_loss

            loss = torch.cat((loss, loss_source_cl), 0)

    return loss

def bounded_logit_inc_sink(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()
    class_logits = (one_hot_labels * perturbed_logit).sum(1)
    not_class_logits = ((1. - one_hot_labels) * perturbed_logit - one_hot_labels * 10000.).max(1)[0]
    loss = torch.clamp(not_class_logits - class_logits, min=-confidence)
    return loss

def l1(perturbed_logit, clean_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0, use_cuda=False):
    loss = torch.sum(torch.abs(perturbed_logit - clean_logit), dim=-1)
    return loss

def l2(perturbed_logit, clean_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0, use_cuda=False):
    loss = torch.sum((perturbed_logit - clean_logit)**2, dim=-1)
    return loss
