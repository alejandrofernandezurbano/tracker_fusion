import argparse
import glob
from pathlib import Path
import open3d
import tracker_fusion.open3d_vis_utils as V
import numpy as np
import torch
#print(torch.__version__)
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
#from pcdet.models import *
from pcdet.utils import common_utils
print('Done Det3D')
loaded_model = None
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / '.bin')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):

        #points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        points = self.ext


        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/chimuelo/fusion_ws/src/tracker_fusion/tracker_fusion/cfgs/kitti_models/pointrcnn_iou.yaml',help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/chimuelo/Documents/detection3d/OpenPCDet/data/kitti/training/velodyne',help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def model3d(points):
    global loaded_model
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=points, logger=None
    )

    # Cargar el modelo solo si aún no se ha cargado
    if loaded_model is None:
        logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename='/home/chimuelo/fusion_ws/src/tracker_fusion/tracker_fusion/model/pointrcnn_iou_7875.pth', logger=logger, to_cpu=False)
        model.cuda()
        model.eval()
        loaded_model = model
    else:
        model = loaded_model
   
    with torch.no_grad():

        idx, data_dict =next(enumerate(demo_dataset))
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict) 
             
        pred_labels =  pred_dicts[0]['pred_labels']
        pred_dicts_filtered = {
        'pred_boxes': pred_dicts[0]['pred_boxes'][pred_labels == 2],
        'pred_scores': pred_dicts[0]['pred_scores'][pred_labels == 2],
        'pred_labels': pred_dicts[0]['pred_labels'][pred_labels == 2]
        }

        return pred_dicts

if __name__ == '__main__':
    main()
