from yacs.config import CfgNode as CN

import datetime
import copy
import logging
import os
import time

import torch

from dataset.utils import DataLoader, tqdm_cycle
from dataset.customized import CustomizedFewshotDataset

from experiments import cfg_from_args
from models import build_model
from models.nn import Measure
from tools.dataset_catalog import DatasetCatalog
from utils import Checkpointer, mkdir
from utils import setup_logger, SummaryWriter, Metric
from utils import start_up, ArgumentParser, data_parallel
from utils import to_cuda
from visualization import build_visualizer


def test(cfg, args):
    logger = logging.getLogger("falcon_logger")
    logger.info("Setting up dependencies.")

    logger.info("Setting up models.")
    gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    model = build_model(cfg).to("cuda")
    model = data_parallel(model, gpu_ids)

    logger.info("Setting up utilities.")
    output_dir = cfg.OUTPUT_DIR

    checkpointer = Checkpointer(cfg, model, None, None)
    iteration = checkpointer.load(args.iteration)
    summary_writer = SummaryWriter(output_dir)

    model.eval()
    temp_sets = [] 
    for dataset_name, temp_set in DatasetCatalog(cfg).get(cfg.DATASETS.TEST, args, as_tuple=True):
        temp_sets.append(temp_set)
    temp_set = temp_sets[0]
    #    start_testing_time = time.time()
    #    last_batch_time = time.time()
    #    test_loader = DataLoader(test_set, cfg)
    #    test_metrics = Metric(delimiter="  ", summary_writer=summary_writer)
    #    visualizer = build_visualizer(cfg.VISUALIZATION, test_set, summary_writer)
    #    logger.info(f"Start testing on {test_set} with mode {test_set.mode}.")
    temp_ = CN(dict(NAME="gqa_fewshot", SPLIT="test", ROOT="/scratch/weiweigu/data/customized_set", OPTS=[], **cfg.CATALOG))
    word_vocab = copy.deepcopy(temp_set.word_vocab)
    names = copy.deepcopy(temp_set.names)
    named_entries = copy.deepcopy(temp_set.named_entries_)
    kinds = copy.deepcopy(temp_set.kinds_)
    use_text = copy.deepcopy(temp_set.use_text)
    
    test_set = CustomizedFewshotDataset(temp_,word_vocab, names, named_entries, kinds, use_text, args)
    test_loader = DataLoader(test_set, cfg) 

    with torch.no_grad():
        model.eval()
        #evaluated = test_set.init_evaluate(args.mode)
        for i, inputs in enumerate(tqdm_cycle(test_loader)):
            #data_time = time.time() - last_batch_time
            inputs = to_cuda(inputs)
            outputs = model(inputs)
            model.callback(inputs, outputs)
            new_concept_embedding = model.box_registry[105]
            #Can only test against train embedding
            # metal, glass as concepts for materials
            # red, yellow as concepts for colors
            # bicycle as unrelated concept
            metal_embedding = model.box_registry[50]
            red_embedding = model.box_registry[70]
            glass_embedding = model.box_registry[28]
            bicycle_embedding = model.box_registry[0]
            yellow_embedding = model.box_registry[104]
            mdl_cfg = cfg.MODEL
            measure = Measure(mdl_cfg)
            
            # Intersections
            metal_new_intersection = measure.intersection(metal_embedding, new_concept_embedding)
            glass_new_intersection = measure.intersection(glass_embedding, new_concept_embedding)
            red_new_intersection = measure.intersection(red_embedding, new_concept_embedding)
            yellow_new_intersection = measure.intersection(yellow_embedding, new_concept_embedding)
            bicycle_new_intersection = measure.intersection(bicycle_embedding, new_concept_embedding)
            # IoUs
            metal_new_iou = measure.iou(metal_embedding, new_concept_embedding)
            glass_new_iou = measure.iou(glass_embedding, new_concept_embedding)
            red_new_iou = measure.iou(red_embedding, new_concept_embedding)
            yellow_new_iou = measure.iou(yellow_embedding, new_concept_embedding)
            bicycle_new_iou = measure.iou(bicycle_embedding, new_concept_embedding)
            # new entails concepts
            metal_new_entailment = measure.entailment(metal_embedding, new_concept_embedding)
            glass_new_entailment = measure.entailment(glass_embedding, new_concept_embedding)
            red_new_entailment = measure.entailment(red_embedding, new_concept_embedding)
            yellow_new_entailment = measure.entailment(yellow_embedding, new_concept_embedding)
            bicycle_new_entailment = measure.entailment(bicycle_embedding, new_concept_embedding)
            # concepts entail new
            new_metal_entailment = measure.entailment( new_concept_embedding,metal_embedding)
            new_glass_entailment = measure.entailment( new_concept_embedding,glass_embedding)
            new_red_entailment = measure.entailment(new_concept_embedding, red_embedding)
            new_yellow_entailment = measure.entailment(new_concept_embedding, yellow_embedding)
            new_bicycle_entailment = measure.entailment(new_concept_embedding, bicycle_embedding)
            breakpoint()
            #test_set.callback(i)
            #test_set.batch_evaluate(inputs, outputs, evaluated)

            #batch_time = time.time() - last_batch_time
            #last_batch_time = time.time()
            #test_metrics.update(batch_time=batch_time, data_time=data_time)

            #if i % 5 == 0:
            #    visualizer.visualize(inputs, outputs, model, iteration + i)

        #metrics = test_set.evaluate_metric(evaluated)
        #visualizer.visualize(evaluated, model, iteration)
    #        test_set.save(output_dir, evaluated, iteration, metrics)
    #        test_metrics.update(**metrics)
    #        test_metrics.log_summary(test_set.tag, iteration)
    #        checkpointer.save(9999999, False)
    #        logger.warning(test_metrics.delimiter.join([f"iter: {iteration}", f"{test_metrics}"]))

        #total_training_time = time.time() - start_testing_time
        logger.info(f"Total testing time: {datetime.timedelta(seconds=total_training_time)}")


def main():
    parser = ArgumentParser()
    args = parser.parse_args()
    cfg = cfg_from_args(args)
    output_dir = mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("falcon_logger", os.path.join(output_dir, "test_log.txt"))
    start_up()

    logger.info(f"Running with args:\n{args}")
    logger.info(f"Running with config:\n{cfg}")
    test(cfg, args)


if __name__ == "__main__":
    main()
