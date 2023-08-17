import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv import Config, DictAction
from mmcv.ops import nms

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate froc curve from detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def calculate_tp(dataset,
                results,
                score_thr=0,
                nms_iou_thr=None):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(dataset.CLASSES)
    maxiou_confidence = np.array([])
    num_gt_bboxes=0
    print(len(dataset),';',len(results))
    assert len(dataset) == len(results)
    num_img = len(results)
    prog_bar = mmcv.ProgressBar(len(results)) # 进度条
    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        per_img_num_gt_bboxes=len(gt_bboxes)
        num_gt_bboxes = num_gt_bboxes + per_img_num_gt_bboxes
        #labels = ann['labels']
        # print('\ngt_bboxes:',gt_bboxes)
        # print('\nlabels:',labels)
        # print('\n')
        if len(res_bboxes[0]):
            per_img_maxiou_confidence=analyze_per_img_dets(gt_bboxes, res_bboxes,
                                score_thr, nms_iou_thr)
            #print('\nper_img_maxiou_confidence:',per_img_maxiou_confidence)
            maxiou_confidence = np.append(maxiou_confidence, per_img_maxiou_confidence)     
        prog_bar.update()
    print('\nnmaxiou_confidence.shape:',maxiou_confidence.shape,'\n')
    maxiou_confidence = maxiou_confidence.reshape(-1, 2)
    maxiou_confidence = maxiou_confidence[np.argsort(-maxiou_confidence[:, 1])] # 按置信度从大到小排序
    
    #print('\nmaxiou_confidence:',maxiou_confidence,'\n')
    print('\nnmaxiou_confidence.shape:',maxiou_confidence.shape,'\n')

    return maxiou_confidence, num_gt_bboxes


def analyze_per_img_dets(gt_bboxes,                  
                         result,
                         score_thr=0,                         
                         nms_iou_thr=None):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """

    for det_label, det_bboxes in enumerate(result):
        # print('\ndet_label:',det_label, '\n')
        # print('\ndet_bbox:',det_bboxes, '\n')
        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes[:, :4],
                det_bboxes[:, -1],
                nms_iou_thr,
                score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        confidence=det_bboxes[:, -1]
        # maxiou = np.max(ious, axis=1)
        maxiou_index=np.argmax(ious,axis=0)
        maxiou=np.zeros(len(det_bboxes))
        maxiou[maxiou_index]=np.max(ious,axis=0)
        maxiou_confidence=np.array([maxiou,confidence])
        # maxiou = np.max(ious, axis=0)
        # max_confidence=confidence[np.argmax(ious, axis=0)]
        # maxiou_confidence=np.array([maxiou,max_confidence])
        maxiou_confidence=maxiou_confidence.T
        #maxiou_confidence=np.concatenate((maxiou,confidence),axis=)


    return maxiou_confidence

def thres(maxiou_confidence, threshold = 0.5):
    """
    将大于阈值的最大交并比记为1, 反正记为0
    :param maxiou_confidence: np.array, 存放所有检测框对应的最大交并比和置信度
    :param threshold: 阈值
    :return tf_confidence: np.array, 存放所有检测框对应的tp或fp和置信度
    """
    maxious = maxiou_confidence[:, 0]
    confidences = maxiou_confidence[:, 1]
    true_or_flase = (maxious > threshold)
    tf_confidence = np.array([true_or_flase, confidences])
    tf_confidence = tf_confidence.T
    tf_confidence = tf_confidence[np.argsort(-tf_confidence[:, 1])]
    return tf_confidence

def result_list(tf_confidence, num_groundtruthbox,num_img):
    """
    从上到下截取tf_confidence, 计算并画图
    :param tf_confidence: np.array, 存放所有检测框对应的tp或fp和置信度
    :param num_groundtruthbox: int, 标注框的总数
    """
    fp_list = []
    fp_per_img_list=[]
    recall_list = []
    precision_list = []
    auc = 0
    mAP = 0
    for num in range(len(tf_confidence)):
        arr = tf_confidence[:(num + 1), 0] # 截取, 注意要加1
        tp = np.sum(arr)
        fp = np.sum(arr == 0)
        fp_per_img=fp/num_img
        recall = tp / num_groundtruthbox
        precision = tp / (tp + fp)
        auc = auc + recall
        mAP = mAP + precision

        fp_list.append(fp)
        fp_per_img_list.append(fp_per_img)
        recall_list.append(recall)
        precision_list.append(precision)
    
    auc = auc / len(fp_list)
    mAP = mAP * max(recall_list) / len(recall_list)
    index=next((n for (n,i) in enumerate(fp_per_img_list) if i>2),-1)
    score=recall_list[index]
    return fp_per_img_list, recall_list, score

def tocsv(fp_per_img_list, recall_list, score, save_dir):
    import pandas as pd
    out_list=[]
    for i in range(len(fp_per_img_list)):
        out_list.append([fp_per_img_list[i],recall_list[i],score])

    column_name = ['fps','recall','score']
    xml_df = pd.DataFrame(out_list, columns=column_name)
    xml_df.to_csv(save_dir, index=None)

def plot(fp_per_img_list, recall_list, score, save_dir):    
    plt.figure()
    plt.title('FROC performance')
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Recall')
    plt.ylim(0, 1.1)
    plt.plot(fp_per_img_list, recall_list, label = 'Score: ' + str(score))

    plt.xlim(0,3.2)
    plt.legend()

    plt.savefig(save_dir, format='png', dpi=300)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    results = mmcv.load(args.prediction_path)
    assert isinstance(results, list)
    if isinstance(results[0], list):
        pass
    elif isinstance(results[0], tuple):
        results = [result[0] for result in results]
    else:
        raise TypeError('invalid type of prediction results')

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    dataset = build_dataset(cfg.data.test)

    maxiou_confidence, num_gt_bboxes = calculate_tp(dataset, results,
                                    args.score_thr,
                                    args.nms_iou_thr)
    tf_confidence = thres(maxiou_confidence, args.tp_iou_thr)
    num_img = len(results)
    fp_per_img_list, recall_list, score = result_list(tf_confidence, num_gt_bboxes,num_img)

    tocsv(fp_per_img_list, recall_list, score, args.save_dir) #create csv file
    # plot(fp_per_img_list, recall_list, score, args.save_dir) #plot froc curve

if __name__ == '__main__':
    main()
