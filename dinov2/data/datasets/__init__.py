# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .image_net import ImageNet
from .image_net_22k import ImageNet22k
from .pathology import PathologyDataset
from .pathology_on_the_fly import PathologyOnTheFlyDataset
from .knn import KNNDataset
from .foundation import PathologyFoundationDataset
from .image_folder import ImageFolderWithNameDataset
from .regions import SlideIDsDataset, SlideRegionDataset
