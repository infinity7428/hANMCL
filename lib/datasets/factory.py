from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.coco_split import coco_split
from datasets.pascal_split import pascal_split
from datasets.imagenet import imagenet
from datasets.episode import episode

# ft
for year in ['seed1', 'seed2','seed3', 'seed4','seed5', 'seed6','seed7', 'seed8','seed9', 'seed10','seed0']:
  for split in ['1shots','2shots','3shots','5shots','10shots','30shots']:
    for n in ['coco']:
        name = 'coco_ft_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco_split(split, year))
    for n in ['pascal']:
        name = 'pascal_ft_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_split(split, year))
    

# coco 20 evaluation
for year in ['set1', 'set2']:
  for split in ['20']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# coco 60 training
for year in ['set1']:
  for split in ['60']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# voc 5 evaluation
for year in ['set1']:
  for split in ['5']:
    name = 'pascal_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: pascal_split(split, year))    
for year in ['set2']:
  for split in ['5']:
    name = 'pascal_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: pascal_split(split, year))   
for year in ['set3']:
  for split in ['5']:
    name = 'pascal_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: pascal_split(split, year))   

# voc 15 training
for year in ['set1']:
  for split in ['15']:
    name = 'pascal_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: pascal_split(split, year))    
for year in ['set2']:
  for split in ['15']:
    name = 'pascal_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: pascal_split(split, year))  
for year in ['set3']:
  for split in ['15']:
    name = 'pascal_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: pascal_split(split, year))  

    
# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
