import sys
import time

import numpy as np
import torch
import torch.nn as nn

from model import make_model


class FeatureExtractor:
    """Label-independent feature extractor built on the SDF-Net backbone."""

    def __init__(self, cfg, weight_path=None, num_classes=1, camera_num=2):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.model = self._initialize_model(weight_path)

    def _initialize_model(self, weight_path=None):
        model = make_model(self.cfg, num_class=self.num_classes, camera_num=self.camera_num)

        if weight_path is None:
            weight_path = self.cfg.TEST.WEIGHT

        if weight_path:
            print(f"Loading model weights from: {weight_path}")
            model.load_param(weight_path)
        else:
            print("No model weight path was provided; using current model initialization.")

        model.to(self.device)
        model.eval()
        print(f"FeatureExtractor ready on {self.device}.")
        return model

    def extract(self, dataloader, progress_callback=None, normalize=True):
        features = []
        pids = []
        camids = []
        img_paths = []
        total_batches = len(dataloader)
        total_images = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
        start_time = time.time()

        print(
            f"    Starting feature extraction: {total_batches} batches, "
            f"batch_size={getattr(dataloader, 'batch_size', 'unknown')}"
        )
        sys.stdout.flush()

        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                img, pid, camid, cam_label, _target_view, imgpath, img_wh = batch
                img = img.to(self.device, non_blocking=True)
                cam_label = cam_label.to(self.device, non_blocking=True)
                img_wh = img_wh.to(self.device, non_blocking=True)

                feat = self.model(img, cam_label=cam_label, img_wh=img_wh)
                if normalize:
                    feat = nn.functional.normalize(feat, dim=1, p=2)

                features.append(feat.detach().cpu())
                pids.extend(np.asarray(pid).tolist())
                camids.extend(np.asarray(camid).tolist())
                img_paths.extend(list(imgpath))

                current = batch_idx + 1
                if progress_callback:
                    processed = (
                        min(current * dataloader.batch_size, total_images)
                        if total_images and getattr(dataloader, "batch_size", None)
                        else current
                    )
                    progress_callback(current, f"processed={processed}")
                elif current % 5 == 0 or current == total_batches:
                    elapsed = time.time() - start_time
                    speed = current / elapsed if elapsed > 0 else 0
                    print(
                        f"    Batch {current}/{total_batches} | "
                        f"{speed:.2f} batch/s | elapsed={elapsed:.1f}s"
                    )
                    sys.stdout.flush()

        if not features:
            raise RuntimeError("No features were extracted. Check the dataloader.")

        features_tensor = torch.cat(features, dim=0)
        features_np = features_tensor.numpy()
        if normalize:
            features_np = self.normalize_features(features_np)

        elapsed = time.time() - start_time
        print(f"    Feature extraction done in {elapsed:.2f}s, shape={features_np.shape}")
        sys.stdout.flush()
        return features_np, np.asarray(pids), np.asarray(camids), img_paths

    def extract_single(self, img_tensor, cam_label=None, img_wh=None, normalize=True):
        with torch.inference_mode():
            img_tensor = img_tensor.to(self.device, non_blocking=True)
            if cam_label is not None:
                cam_label = cam_label.to(self.device, non_blocking=True)
            if img_wh is not None:
                img_wh = img_wh.to(self.device, non_blocking=True)

            feat = self.model(img_tensor, cam_label=cam_label, img_wh=img_wh)
            if normalize:
                feat = nn.functional.normalize(feat, dim=1, p=2)

        return feat.detach().cpu().numpy()

    @staticmethod
    def normalize_features(features):
        if isinstance(features, torch.Tensor):
            return nn.functional.normalize(features, dim=1, p=2)

        features = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.clip(norms, a_min=1e-12, a_max=None)
