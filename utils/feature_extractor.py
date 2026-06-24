import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm
from model import make_model


class FeatureExtractor:
    def __init__(self, cfg, weight_path=None):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.num_classes = 0
        self.camera_num = 0
        self._initialize_model(weight_path)

    def _initialize_model(self, weight_path=None):
        self.num_classes = 90020
        self.camera_num = 2

        self.model = make_model(self.cfg, num_class=self.num_classes, camera_num=self.camera_num)

        if weight_path is None:
            weight_path = self.cfg.TEST.WEIGHT

        if weight_path:
            print(f"Loading pretrained weights from: {weight_path}")
            self.model.load_param(weight_path)

        self.model.to(self.device)
        self.model.eval()
        print(f"FeatureExtractor initialized. Device: {self.device}")

    def extract(self, dataloader, progress_callback=None):
        features = []
        pids = []
        camids_list = []
        img_paths = []
        total_batches = len(dataloader)
        start_time = time.time()

        print(f"    → Starting feature extraction: {total_batches} batches, batch_size={dataloader.batch_size}")
        sys.stdout.flush()

        with torch.no_grad():
            for n_iter, batch in enumerate(dataloader):
                img, pid, camid_batch, cam_label, target_view, imgpath, img_wh = batch

                img = img.to(self.device, non_blocking=True)
                cam_label = cam_label.to(self.device, non_blocking=True)
                img_wh = img_wh.to(self.device, non_blocking=True)

                feat = self.model(img, cam_label=cam_label, img_wh=img_wh)

                features.append(feat.detach().cpu())
                pids.extend(pid)
                camids_list.extend(camid_batch.tolist() if isinstance(camid_batch, torch.Tensor) else camid_batch)
                img_paths.extend(imgpath)

                del img, cam_label, img_wh, feat
                torch.cuda.empty_cache()

                if progress_callback:
                    progress_callback(n_iter + 1, f"{(n_iter+1)*dataloader.batch_size} 张图像")
                else:
                    if (n_iter + 1) % 5 == 0 or (n_iter + 1) == total_batches:
                        elapsed = time.time() - start_time
                        progress = (n_iter + 1) / total_batches * 100
                        processed = (n_iter + 1) * dataloader.batch_size
                        speed = processed / elapsed if elapsed > 0 else 0
                        eta = (total_batches - n_iter - 1) / (n_iter + 1) * elapsed if n_iter > 0 else 0

                        bar_len = 30
                        filled = int(bar_len * (n_iter + 1) / total_batches)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        print(f"    → [{bar}] {n_iter+1}/{total_batches} ({progress:.1f}%) "
                              f"| {processed} images | {speed:.1f} img/s | ETA: {eta:.0f}s")
                        sys.stdout.flush()

        print(f"    → Concatenating {len(features)} feature tensors...")
        sys.stdout.flush()
        features_tensor = torch.cat(features, dim=0)
        del features
        torch.cuda.empty_cache()
        
        features_np = features_tensor.numpy()
        features_np = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
        
        pids = np.array(pids)
        camids = np.array(camids_list)
        total_time = time.time() - start_time
        print(f"    → Feature extraction done in {total_time:.2f}s, shape: {features_np.shape}, normalized: True")
        sys.stdout.flush()

        return features_np, pids, camids, img_paths

    def extract_single(self, img_tensor, cam_label=None, img_wh=None):
        with torch.inference_mode():
            img_tensor = img_tensor.to(self.device, non_blocking=True)
            if cam_label is not None:
                cam_label = cam_label.to(self.device, non_blocking=True)
            if img_wh is not None:
                img_wh = img_wh.to(self.device, non_blocking=True)

            feat = self.model(img_tensor, cam_label=cam_label, img_wh=img_wh)

        return feat.cpu().numpy()

    def normalize_features(self, features):
        if isinstance(features, torch.Tensor):
            return nn.functional.normalize(features, dim=1, p=2)
        else:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            return features / norms