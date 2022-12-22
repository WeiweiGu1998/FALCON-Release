import random
from collections import Counter
from itertools import chain
from multiprocessing import Pool

import torch
from tqdm import tqdm
# noinspection PyUnresolvedReferences
from tqdm.contrib.itertools import product as tqdm_product

from dataset.meta_dataset import MetaDataset, MetaBuilderDataset
from torchvision.transforms import functional as TF
from dataset.utils import FixedResizeTransform, WordVocab, ProgramVocab
from dataset.utils import sample_with_ratio
from utils import load, join, nonzero


class CustomizedFewshotDataset(MetaDataset):

    @property
    def question_concepts(self):
        return torch.tensor([q['concept_index'] for q in self.questions])

    @property
    def transform_fn(self):
        return FixedResizeTransform

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.root="/scratch/weiweigu/data/customized_set"
        self.questions = self._build_questions()
        #self.split_specs = self.concept2splits[self.question_concepts]
        #self.indices_split = self.select_split(self.split_specs)

    def get_image(self, image_index):
        return TF.to_tensor(read_image(join(self.root, f"images/{image_index}.jpg")))

    def get_stacked_scenes(self, image_index):
        return {"image": self.transform(self.get_image(image_index))}

    def _build_questions(self):
        return load(join(self.root, "custom_fewshot.json"))
