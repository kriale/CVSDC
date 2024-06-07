import motmetrics as mm


class MotMetricsService:
    def __init__(self, predictions_dict, groundtruth_dict):
        self.predictions_dict = predictions_dict
        self.groundtruth_dict = groundtruth_dict

    def calculate_mot_metrics(self):
        accumulator = mm.MOTAccumulator(auto_id=True)

        for frame_id in self.groundtruth_dict:
            if frame_id not in self.predictions_dict:
                break

            y_true = self.groundtruth_dict.get(frame_id, [])
            y_pred = self.predictions_dict.get(frame_id, [])
            if y_true and y_pred:
                gt_ids = [gt[0] for gt in y_true]
                pred_ids = [pred[0] for pred in y_pred]
                gt_boxes = [gt[1:5] for gt in y_true]
                pred_boxes = [pred[1:5] for pred in y_pred]
                distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
                accumulator.update(
                    gt_ids,  # Ground truth objects in this frame
                    pred_ids,  # Detector hypotheses in this frame
                    distances  # Distances from object to hypotheses
                )

        mh = mm.metrics.create()
        summary = mh.compute(
            accumulator,
            metrics=['num_frames', 'mota', 'motp'],
            name='acc'
        )
        return summary['mota'].iloc[0], summary['motp'].iloc[0]