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
from utils import load, join, nonzero, read_image


class CustomizedFewshotDataset(MetaDataset):

    @property
    def question_concepts(self):
        return torch.tensor([q['concept_index'] for q in self.questions])

    @property
    def transform_fn(self):
        return FixedResizeTransform

    def __init__(self, cfg, word_vocab, names, named_entries, kinds, use_text, args):
        super().__init__(cfg, args)
        self.root="/home/local/ASUAD/weiweigu/data/customized_set"
        self.word_vocab = word_vocab
        self.names = names
        self.named_entries_ = named_entries
        self.kinds_ = kinds
        self.use_text = use_text
        self.questions = self._build_questions()
        self.split_specs = torch.Tensor([2]) 
        self.indices_split = self.select_split(self.split_specs)
        self.image_filenames = [f"{i}.jpg" for i in range(29)]

    def get_image(self, image_index):
        return TF.to_tensor(read_image(join(self.root, f"images/{image_index}.jpg")))

    def get_stacked_scenes(self, image_index):
        return {"image": self.transform(self.get_image(image_index))}

    def _build_questions(self):
        return load(join(self.root, "custom_fewshot.json"))

    def get_annotated_image(self, image_index, mask_index=None):
        return self.get_image(image_index)
