from mmseg.registry import METRICS
from mmengine.evaluator import BaseMetric
import numpy as np

@METRICS.register_module()
class CrackMetric(BaseMetric):
    """自定义裂缝分割指标，兼容新版 MMEngine"""
    default_prefix = 'crack'

    def __init__(self, **kwargs):
        """
        kwargs 可能包含：
        - metric
        - iou_metrics
        - topk
        - prefix
        - collect_device
        所以必须全部接收并忽略或传给父类
        """
        super().__init__(**kwargs)
        self.results = []

    def process(self, data_batch, predictions):
        preds = predictions[0].argmax(dim=1).cpu().numpy()  # [N,H,W]
        gts = data_batch['data_samples'][0].gt_sem_seg.data.cpu().numpy()

        for pred, gt in zip(preds, gts):
            self.results.append(dict(
                tp=np.sum((pred == 1) & (gt == 1)),
                fp=np.sum((pred == 1) & (gt == 0)),
                fn=np.sum((pred == 0) & (gt == 1)),
            ))

    def compute_metrics(self, results):
        tp = sum(x['tp'] for x in results)
        fp = sum(x['fp'] for x in results)
        fn = sum(x['fn'] for x in results)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return dict(precision=precision, recall=recall, F1=f1)
