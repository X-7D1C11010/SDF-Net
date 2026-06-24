# encoding: utf-8
import glob
import os.path as osp
from .bases import BaseImageDataset


class MergedDataset(BaseImageDataset):
    """
    合并的多数据集 - 支持光学和SAR跨模态ReID
    """

    def __init__(self, root="", verbose=True, pid_begin=0, is_train=True, **kwargs):
        super(MergedDataset, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        self.is_train = is_train

        self._check_before_run()
        self.pid_begin = pid_begin

        if self.is_train:
            train, train_pair = self._process_dir_train(self.train_dir, relabel=True)
        else:
            train, train_pair = [], []

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Merged Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)
            if train_pair is not None:
                print("Number of RGB-SAR pair: {}".format(len(train_pair)))
                print("  ----------------------------------------")

        self.train = train
        self.train_pair = train_pair
        self.query = query
        self.gallery = gallery

        if self.is_train:
            (
                self.num_train_pids,
                self.num_train_imgs,
                self.num_train_cams,
                self.num_train_vids,
            ) = self.get_imagedata_info(self.train)
            (
                self.num_train_pair_pids,
                self.num_train_pair_imgs,
                self.num_train_pair_cams,
                self.num_train_pair_vids,
            ) = self.get_imagedata_info_pair(self.train_pair)
        else:
            self.num_train_pids = 0
            self.num_train_imgs = 0
            self.num_train_cams = 0
            self.num_train_vids = 0
            self.num_train_pair_pids = 0
            self.num_train_pair_imgs = 0
            self.num_train_pair_cams = 0
            self.num_train_pair_vids = 0

        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
        ) = self.get_imagedata_info(self.query)

        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
        ) = self.get_imagedata_info(self.gallery)

    def get_imagedata_info_pair(self, data):
        pids, cams, tracks = [], [], []

        for img in data:
            for _, pid, camid, trackid in img:
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

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if self.is_train and not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg")) + \
                    glob.glob(osp.join(dir_path, "*.jpeg")) + \
                    glob.glob(osp.join(dir_path, "*.png"))

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = self._extract_pid(img_path)
            pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        
        for img_path in sorted(img_paths):
            pid = self._extract_pid(img_path)
            camid = self._extract_camid(img_path)
            
            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_dir_train(self, dir_path, relabel=False):
        opt_dir = osp.join(dir_path, "opt")
        sar_dir = osp.join(dir_path, "sar")

        opt_paths = []
        if osp.exists(opt_dir):
            opt_paths = glob.glob(osp.join(opt_dir, "*.jpg")) + \
                        glob.glob(osp.join(opt_dir, "*.jpeg")) + \
                        glob.glob(osp.join(opt_dir, "*.png"))

        sar_paths = []
        if osp.exists(sar_dir):
            sar_paths = glob.glob(osp.join(sar_dir, "*.jpg")) + \
                        glob.glob(osp.join(sar_dir, "*.jpeg")) + \
                        glob.glob(osp.join(sar_dir, "*.png"))

        all_paths = opt_paths + sar_paths

        pid_container = set()
        pid2sar = {}
        pid2opt = {}

        for img_path in sorted(all_paths):
            pid = self._extract_pid(img_path)
            pid_container.add(pid)
            
            if self._is_sar(img_path):
                if pid not in pid2sar:
                    pid2sar[pid] = [img_path]
                else:
                    pid2sar[pid].append(img_path)
            else:
                if pid not in pid2opt:
                    pid2opt[pid] = [img_path]
                else:
                    pid2opt[pid].append(img_path)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(all_paths):
            pid = self._extract_pid(img_path)
            camid = self._extract_camid(img_path)
            
            if relabel:
                pid = pid2label[pid]
            
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        dataset_pair = []
        for pid in pid2opt.keys():
            if pid not in pid2sar:
                continue
            for opt_path in pid2opt[pid]:
                for sar_path in pid2sar[pid]:
                    dataset_pair.append([
                        (opt_path, self.pid_begin + pid2label[pid], 0, 1),
                        (sar_path, self.pid_begin + pid2label[pid], 1, 1),
                    ])

        return dataset, dataset_pair

    def _extract_pid(self, img_path):
        import hashlib
        import re
        filename = osp.basename(img_path)
        name_without_ext = osp.splitext(filename)[0]
        normalized = name_without_ext.lower()
        
        patterns = [
            r'^(\d+)_s\d+c\d+_(opt|sar)$',
            r'^(\d+)_s\d+c\d+$',
            r'^(\d+)[_-]',
            r'^(\d+)$',
            r'(?:pid|id)[_-]?(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        
        normalized = re.sub(r'[_-]?(opt|rgb|visible|vis|sar|ir)$', '', normalized)
        stable_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return int(stable_hash[:12], 16) % 1000000

    def _extract_camid(self, img_path):
        if self._is_sar(img_path):
            return 1
        return 0

    def _is_sar(self, img_path):
        filename = osp.basename(img_path).lower()
        return 'sar' in filename or img_path.startswith(osp.join(self.train_dir, "sar"))
