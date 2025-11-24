import numpy as np
from mmseg.registry import METRICS
from mmseg.evaluation.metrics import IoUMetric


@METRICS.register_module()
class CrackMetric(IoUMetric):
    """Compute Crack Precision, Recall, F1, IoU for binary segmentation."""

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)

        # 二值分割：背景=0，裂缝=1
        tp = 0
        fp = 0
        fn = 0

        for result in results:
            pred = result['pred_sem_seg'].data[0]
            gt = result['gt_sem_seg'].data[0]

            tp += np.logical_and(pred == 1, gt == 1).sum()
            fp += np.logical_and(pred == 1, gt == 0).sum()
            fn += np.logical_and(pred == 0, gt == 1).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        metrics.update({
            'Crack Precision': precision,
            'Crack Recall': recall,
            'Crack F1': f1
        })

        return metrics
