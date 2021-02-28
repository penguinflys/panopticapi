
import torch
import os
import logging
from collections import OrderedDict
from detectron2.data import MetadataCatalog,build_detection_train_loader
from detectron2.data import DatasetCatalog
from detectron2.data.transforms import augmentation
# from detectron2.data import detection_utils as utils
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# from detectron2.data import MetadataCatalog
# from detectron2.data.common import MapDataset
# from detectron2.data.build import build_batch_data_loader
# from detectron2.data.common import AspectRatioGroupedDataset
# from detectron2.data.datasets import register_coco_panoptic_separated

from detectron2.engine import launch
# from detectron2.engine import default_setup
from detectron2.engine import DefaultTrainer
from detectron2.engine import default_argument_parser
from detectron2.engine import hooks
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator,inference_on_dataset, print_csv_format
# from detectron2.evaluation import SemSegEvaluator
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results
from detectron2.data.build import build_detection_test_loader

import sys
sys.path.append("/data-input/ISPRS_2D_MultiTasks")
import isprs.transformer as T
from isprs.models import RotatedCOCOEvaluatorWithMask
from isprs.models import ISPRSSemSegEvaluator
# from isprs.models import RROIHeadsWithMasks
# from isprs.models import RotatedMaskRCNNConvUpsampleHead
# from isprs.models import build_resnest_fpn_backbone
from isprs.models.hooks.validation_hook import LossEvalHook
from isprs.mapper import ISPRSCOCOStyleMapperAxisAligned, ISPRSCOCOStyleMapperRotated
# train loader
# from config import add_isprs_config

# from isprs.config import get_isprsloader_config, deep_update, add_isprs_config
# from utils import get_isprs_instance_meta, get_isprs_panoptic_seperated_meta
from isprs.utils import register_isprs_train_panoptic, register_isprs_val_panoptic, register_isprs_test_panoptic
# TODO define hooks and something else to train the network
from isprs.utils import register_isprs_train_instance, register_isprs_val_instance, register_isprs_test_instance
from isprs.mapper import ISPRSOnlineTrainMapper
from isprs.transformer import *
from isprs.utils import setup

class ISPRSTrainer(DefaultTrainer):
    """
    ISPRS Trainer
    """
    # TODO build trainloader
    @classmethod
    def build_train_loader(cls, cfg, mapper=None):
        if cfg.ISPRS.MODE != "COCO":
            dataset_name = cfg.DATASETS.TRAIN[0]
            if mapper is None:
                mapper = ISPRSOnlineTrainMapper(cfg, True)
            
            func = DatasetCatalog[dataset_name]
            isprs = func(cfg, cfg.SOLVER.IMS_PER_BATCH, mapper=mapper,dataset_name = dataset_name)
            return isprs
        elif cfg.ISPRS.LABEL.BOXMODE == "ROTATED":
            # mapper = ISPRSCOCOStyleMapperRotated(cfg,is_train = True, augmentations = [T.ISPRSRandomRotation([0,360])] )
            mapper = ISPRSCOCOStyleMapperRotated(cfg,is_train = True)
            return build_detection_train_loader(cfg,mapper=mapper)
        else:
            mapper = ISPRSCOCOStyleMapperAxisAligned(cfg,is_train = True)
            return build_detection_train_loader(cfg,mapper=mapper)
    @classmethod
    def MapDataset(cls, func, dataset):
        return func(next(dataset))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "isprs_panoptic_seg"]:
           evaluators_list.append(
               ISPRSSemSegEvaluator(
                   dataset_name,
                   distributed=True,
                   output_dir=output_folder,
                   num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
               )
           )
        if evaluator_type in ["isprs_instance", "isprs_panoptic_seg","isprs_rpn"]:
            if cfg.ISPRS.LABEL.BOXMODE == "ROTATED":
             evaluators_list.append(
                 RotatedCOCOEvaluatorWithMask(dataset_name, cfg, True, output_folder)
                 )
            else:
             evaluators_list.append(
                 COCOEvaluator(dataset_name, cfg, True, output_folder)
                 )
        if evaluator_type == "isprs_panoptic_seg":
            evaluators_list.append(
                COCOPanopticEvaluator(dataset_name, output_folder)
            )
        
        return DatasetEvaluators(evaluators_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        """
        This is test during training procedure.
        Temperaroly use Standard Tests
        """
        print("test with test dataset")
        logger = logging.getLogger("detectron2.trainer")
        #TODO define Training procedure
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)

        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(
                    cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        hooks = super().build_hooks()
        if self.cfg.ISPRS.LABEL.BOXMODE == "ROTATED":
            eval_mapper = ISPRSCOCOStyleMapperRotated
        else:
            eval_mapper = ISPRSCOCOStyleMapperAxisAligned
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self._trainer.model,
            
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                eval_mapper(self.cfg,True)
            )
        ))
        return hooks

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
def main(args):
    cfg = setup(args)

    # TODO REGISTRATION OF DATASET

    register_isprs_train_instance(cfg)
    register_isprs_val_instance(cfg)
    register_isprs_test_instance(cfg)
    register_isprs_train_panoptic(cfg)
    register_isprs_val_panoptic(cfg)
    register_isprs_test_panoptic(cfg)
    
    if args.eval_only:
        model = ISPRSTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = ISPRSTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(ISPRSTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res



    trainer = ISPRSTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()




if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
