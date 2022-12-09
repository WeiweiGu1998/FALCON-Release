import random
from collections import Counter
from itertools import chain
from multiprocessing import Pool

import torch
from tqdm import tqdm
from torchvision.transforms import functional as TF
from dataset.utils import FixedResizeTransform, WordVocab, ProgramVocab
from utils import load, join, read_image, file_cached, IdentityDict

# noinspection PyUnresolvedReferences
from tqdm.contrib.itertools import product as tqdm_product

from dataset.meta_dataset import MetaDataset
from dataset.utils import sample_with_ratio


class CustomFewshotDataset(MetaDataset):
    _concept_knowledge = "knowledge/"

    def __init__(self, cfg, args):
        super().__init__(cfg, args)

    @property
    def transform_fn(self):
        return FixedResizeTransform

    def _build_images(self):
        pass

    def get_image(self, image_index):
        return TF.to_tensor(read_image(join(self.root, "images", self.image_filenames[image_index])))

    def get_stacked_scenes(self, image_index):
        assert not torch.is_tensor(image_index)
        return {'image': self.transform(self.get_image(image_index))}

    def exist_question(self, candidate):
        return f"Is there a {self.names[candidate]} object?"

    def exist_statement(self, candidate):
        return f"There is a {self.names[candidate]} object."

    def filter_question(self, candidate, filters):
        return f"Is the {' '.join(self.names[f] for f in filters)} object a {self.names[candidate]} object? "

    def filter_statement(self, candidate, filters):
        return f"The {' '.join(self.names[f] for f in filters)} object is a {self.names[candidate]} object."

    def metaconcept_text(self, supports, relations, concept_index):
        other_names = list(self.names[s] for e, s in zip(relations, supports) if e != 0)
        return f"{', '.join(other_names + [self.names[concept_index]])} describes the same property of an " \
               f"object.".capitalize()
