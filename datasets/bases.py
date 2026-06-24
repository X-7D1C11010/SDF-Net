from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
# 解除 PIL 的像素限制，避免 DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    max_retries = 3
    retry_count = 0
    
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    
    while retry_count < max_retries:
        try:
            img = Image.open(img_path)
            # 检查图像是否有效
            img.verify()
            img = Image.open(img_path)  # 重新打开以获取可操作的图像对象
            return img
        except IOError as e:
            retry_count += 1
            if retry_count < max_retries:
                print(
                    "IOError incurred when reading '{}'. Retry {}/{}.".format(
                        img_path, retry_count, max_retries
                    )
                )
            else:
                print(
                    "Failed to read '{}' after {} retries. Skipping this file.".format(
                        img_path, max_retries
                    )
                )
                raise e
        except Exception as e:
            print(
                "Unexpected error reading '{}': {}. Skipping this file.".format(
                    img_path, str(e)
                )
            )
            raise e


def sar32bit2RGB(img):
    nimg = np.array(img, dtype=np.float32)
    nimg = nimg / nimg.max() * 255
    nimg_8 = nimg.astype(np.uint8)
    cv_img = cv2.cvtColor(nimg_8, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(cv_img)
    return pil_img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        if train is not None:
            (
                num_train_pids,
                num_train_imgs,
                num_train_cams,
                num_train_views,
            ) = self.get_imagedata_info(train)
        (
            num_query_pids,
            num_query_imgs,
            num_query_cams,
            num_train_views,
        ) = self.get_imagedata_info(query)
        (
            num_gallery_pids,
            num_gallery_imgs,
            num_gallery_cams,
            num_train_views,
        ) = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        if train is not None:
            print(
                "  train    | {:5d} | {:8d} | {:9d}".format(
                    num_train_pids, num_train_imgs, num_train_cams
                )
            )
        print(
            "  query    | {:5d} | {:8d} | {:9d}".format(
                num_query_pids, num_query_imgs, num_query_cams
            )
        )
        print(
            "  gallery  | {:5d} | {:8d} | {:9d}".format(
                num_gallery_pids, num_gallery_imgs, num_gallery_cams
            )
        )
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, pair=False):
        self.dataset = dataset
        self.transform = transform
        self.pair = pair

    def __len__(self):
        return len(self.dataset)

    def get_image(self, img_path):
        img = read_image(img_path).convert("RGB")
        img_size = img.size
        img_size = [img_size[0] * 0.75, img_size[1] * 0.75]
        img_size = (
            (img_size[0] / 93 - 0.434) / 0.031,
            (img_size[1] / 427 - 0.461) / 0.031,
            img_size[1] / img_size[0],
        )
        if self.transform is not None:
            img = self.transform(img)
        return img, img_size

    def __getitem__(self, index):
        if self.pair:
            imgs = []
            for img in self.dataset[index]:
                img_path, pid, camid = img
                im, img_size = self.get_image(img_path)
                imgs.append((im, pid, camid, img_path.split("/")[-1], img_size))
            return imgs
        else:
            img_path, pid, camid, trackid = self.dataset[index]
            img, img_size = self.get_image(img_path)
            return img, pid, camid, trackid, img_path.split("/")[-1], img_size
