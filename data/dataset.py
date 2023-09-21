from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import nltk
import torch
import os
from torchvision import transforms


class CustomCocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, cfg, mode, vocabulary):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.cfg = cfg
        self.mode = mode

        if self.mode == "train":
            self.root = cfg.DATASET.RESIZED_IMAGE_DIR
            self.coco_data = COCO(cfg.DATASET.ANNOTATIONS_DIR)

        elif self.mode == "val":
            self.root = cfg.DATASET.RESIZED_VAL_IMAGE_DIR
            self.coco_data = COCO(cfg.DATASET.VAL_ANNOTATIONS_DIR)

        self.indices = list(self.coco_data.anns.keys())
        self.vocabulary = vocabulary

    def __getitem__(self, idx):
        """Returns one data pair (image and caption)."""
        coco_data = self.coco_data
        vocabulary = self.vocabulary
        annotation_id = self.indices[idx]
        caption = coco_data.anns[annotation_id]["caption"]
        image_id = coco_data.anns[annotation_id]["image_id"]
        image_path = coco_data.loadImgs(image_id)[0]["file_name"]

        image = Image.open(os.path.join(self.root, image_path)).convert("RGB")

        if self.mode == "train":
            transform_list = transforms.Compose(
                [
                    transforms.RandomCrop(
                        (self.cfg.TRAIN.IMG_SIZE, self.cfg.TRAIN.IMG_SIZE)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )

            image = transform_list(image)

        elif self.mode == "val":
            # without the horizontal flip
            transform_list = transforms.Compose(
                [
                    transforms.RandomCrop(
                        (self.cfg.TRAIN.IMG_SIZE, self.cfg.TRAIN.IMG_SIZE)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )

            image = transform_list(image)

        # Convert caption (string) to word ids.
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocabulary("<start>"))
        caption.extend([vocabulary(token) for token in word_tokens])
        caption.append(vocabulary("<end>"))
        ground_truth = torch.Tensor(caption)
        return image, ground_truth

    def __len__(self):
        # return len(self.indices)
        return 8


def collate_function(data_batch):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    imgs, caps = zip(*data_batch)

    # Merge images (from list of 3D tensors to 4D tensor).
    # Originally, imgs is a list of <batch_size> number of RGB images with dimensions (3, 256, 256)
    # This line of code turns it into a single tensor of dimensions (<batch_size>, 3, 256, 256)
    imgs = torch.stack(imgs, 0)

    # Merge captions (from list of 1D tensors to 2D tensor), similar to merging of images donw above.
    cap_lens = [len(cap) for cap in caps]
    tgts = torch.zeros(len(caps), max(cap_lens)).long()
    for i, cap in enumerate(caps):
        end = cap_lens[i]
        tgts[i, :end] = cap[:end]
    return imgs, tgts, cap_lens
