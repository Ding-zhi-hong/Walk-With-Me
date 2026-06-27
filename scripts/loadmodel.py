#!/usr/bin/env python3
"""
CityWalkerFeat 模型预加载 + 推理接口
- 预先加载模型到GPU，保持实例化状态
- 提供统一推理接口，接收自定义输入数据返回预测结果
"""

import sys
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime
# ========== 关键修复：添加CityWalker根目录到sys.path ==========
CITYWALKER_ROOT = "/home/robot/CityWalker"
if CITYWALKER_ROOT not in sys.path:
    sys.path.insert(0, CITYWALKER_ROOT)

# 现在可以正常导入
from pl_modules.citywalker_feat_module import CityWalkerFeatModule

# ===========================
# 全局配置（与1.py保持一致）
# ===========================
# 模型权重路径
DINOv2_CKPT = "/home/robot/CityWalker/checkpoints/dinov2_vitb14_pretrain.pth"
CITYWALKER_CKPT = "/home/robot/CityWalker/checkpoints/CityWalker_2000hr.ckpt"

# 固定参数
# step_scale: 步长缩放因子
#   训练时每个视频按自身平均步长归一化(DPVO任意尺度)，模型学到的输出是"归一化单位"
#   推理时乘以 step_scale 得到真实米数。
#   建议设置为机器人1秒内的典型位移量。例如:
#   - 四足机器人慢走 0.5m/s → step_scale ≈ 0.5
#   - 四足机器人快走 1.0m/s → step_scale ≈ 1.0
#   - 如预测轨迹偏大(机器人冲过头) → 减小 step_scale
#   - 如预测轨迹偏小(机器人不动)  → 增大 step_scale
STEP_SCALE = 0.60
IMG_SIZE = (350, 630)  # H, W
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===========================
# 必要的工具类（复用1.py）
# ===========================
class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, DictNamespace(**v))
            else:
                setattr(self, k, v)

sys.modules["__main__"].DictNamespace = DictNamespace

# 离线DINOv2封装
class DINOv2ViTB14Offline(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        # 本地DINOv2仓库路径
        dinov2_repo = "/home/robot/.cache/torch/hub/facebookresearch_dinov2_main"
        self.dinov2 = torch.hub.load(
            dinov2_repo,
            "dinov2_vitb14",
            pretrained=False,
            source="local"
        )
        print(f"[INFO] Loading local DINOv2 weights from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.dinov2.load_state_dict(ckpt, strict=False)
        self.dinov2.eval()
        self.head = nn.Identity()

    def forward(self, x):
        return self.dinov2(x)

# ===========================
# CityWalker模型封装类
# ===========================
class CityWalkerInferencer:
    def __init__(self):
        self.device = DEVICE
        self.step_scale = STEP_SCALE
        self.img_size = IMG_SIZE
        self.model = None
        self._load_config()
        self._load_model()

    def _load_config(self):
        """加载模型配置（与1.py完全一致）"""
        from omegaconf import OmegaConf
        self.cfg = OmegaConf.create({
            "project": {"result_dir": "results"},
            "training": {
                "normalize_step_length": True,
                "direction_loss_weight": 5.0,
                "feature_loss_weight": 0.1,
                "batch_size": 1,
            },
            "data": {"type": "citywalk_feat"},
            "model": {
                "type": "citywalker_feat",
                "do_rgb_normalize": True,
                "do_resize": True,
                "obs_encoder": {
                    "type": "dinov2_vitb14",
                    "context_size": 5,
                    "crop": [350, 630],
                    "resize": [350, 630],
                    "freeze": True,
                },
                "cord_embedding": {
                    "type": "input_target",
                    "num_freqs": 6,
                    "include_input": True,
                },
                "decoder": {
                    "type": "attention",
                    "len_traj_pred": 5,
                    "num_heads": 8,
                    "num_layers": 16,
                    "ff_dim_factor": 4,
                },
                "encoder_feat_dim": 768,
                "output_coordinate_repr": "euclidean",
            },
            "validation": {"num_visualize": 0},
            "testing": {"num_visualize": 0},
        })

    def _load_model(self):
        """加载完整模型（DINOv2 + CityWalker）"""
        print(f"[INFO] Loading CityWalkerFeat model (offline DINOv2) to {self.device}...")
        
        # 初始化CityWalker模块
        self.model = CityWalkerFeatModule(self.cfg)
        
        # 替换离线DINOv2编码器
        self.model.model.obs_encoder = DINOv2ViTB14Offline(DINOv2_CKPT)
        print("[INFO] DINOv2 encoder replaced (offline mode)")
        
        # 加载CityWalker权重
        ckpt = torch.load(CITYWALKER_CKPT, map_location="cpu")
        if "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        
        # 移至GPU并设置eval模式
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[SUCCESS] Model loaded and ready on {self.device}")

    def _preprocess_images(self, images):
        """
        预处理图像（5张：history_0~3 + current）
        :param images: 列表，长度为5，每项可以是：
           - str: 图像文件路径（向后兼容）
           - np.ndarray: BGR 图像 numpy 数组 (H, W, 3)
        :return: 预处理后的tensor (1, 5, 3, H, W)
        """
        if len(images) != 5:
            raise ValueError(f"需要5张图像（history_0~3 + current），但收到{len(images)}张")

        imgs = []
        for img_data in images:
            if isinstance(img_data, str):
                # 从文件路径读取
                if not os.path.exists(img_data):
                    raise FileNotFoundError(f"图像文件不存在: {img_data}")
                img = cv2.imread(img_data)
            else:
                # 直接使用 numpy 数组（跳过磁盘 I/O）
                img = img_data

            # 仅在尺寸不匹配时 resize（避免冗余缩放）
            h, w = img.shape[:2]
            target_w, target_h = self.img_size[1], self.img_size[0]
            if w != target_w or h != target_h:
                img = cv2.resize(img, (target_w, target_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))  # (3, H, W)

        # 拼接为 (1, 5, 3, H, W)
        return torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)

    def _preprocess_poses(self, history_poses_model, target_pose_model):
        """
        预处理位姿数据（模型坐标系：right, forward）
        :param history_poses_model: 列表，4个历史位姿+[0,0]（共5个），格式[[right, fwd], ...]
        :param target_pose_model: 目标位姿 [right, fwd]
        :return: 预处理后的坐标tensor (1, 6, 2)
        """
        if len(history_poses_model) == 4:
            history_poses_model = history_poses_model + [[0.0, 0.0]]
        if len(history_poses_model) != 5:
            raise ValueError(f"history_poses需包含5个点位，但收到{len(history_poses_model)}个")

        # 归一化（除以 step_scale）
        history_norm = [[x/self.step_scale, y/self.step_scale] for x, y in history_poses_model]
        target_norm = [target_pose_model[0]/self.step_scale, target_pose_model[1]/self.step_scale]

        # 构造坐标tensor
        coords = torch.zeros((1, 6, 2), device=self.device)
        for i in range(5):
            coords[0, i, 0] = history_norm[i][0]
            coords[0, i, 1] = history_norm[i][1]
        coords[0, 5, 0] = target_norm[0]
        coords[0, 5, 1] = target_norm[1]

        return coords

    def predict(self, images, history_poses, target_pose, save_results=False, save_dir=None):
        """
        核心推理接口
        :param images: 列表，5张图像，每项可以是：
           - str: 图像文件路径
           - np.ndarray: BGR 图像数组 (H, W, 3)
           顺序: [history_0, history_1, history_2, history_3, current]
        :param history_poses: 列表，4个历史位姿 [[前向, 左侧], ...]（ROS约定，单位米）
        :param target_pose: 列表，目标位姿 [前向, 左侧]（ROS约定，单位米）
        :param save_results: 是否保存预测结果
        :param save_dir: 保存目录（save_results=True时必填）
        :return: 字典，包含预测结果
            {
                "waypoints": 预测的路径点 (5个)，np.array (5,2)，ROS约定(前向, 左侧)，单位米
                "arrival_confidence": 到达置信度 (0~1)
                "history_poses": 输入的历史位姿（ROS约定）
                "target_pose": 输入的目标位姿（ROS约定）
                "step_scale": 步长缩放因子
            }
        """
        # ════════════════════════════════════════════════════════════
        #  坐标系转换: ROS(前向, 左侧) → 模型(右侧, 前向)
        #
        #  模型训练数据来自DPVO视觉里程计(OpenCV相机标准):
        #    DPVO X = 右侧(pose[:,0]), DPVO Y = 下方(丢弃), DPVO Z = 前向(pose[:,2])
        #  代码取 body-frame 变换后的列 [:, [0, 2]] = (右侧, 前向)
        #  详见 CityWalker 论文 Figure 3: φ_action 从正y轴起算 = 前向
        #
        #  ROS 里程计 body-frame: (前向, 左侧, 竖直)
        #  转换: 右侧 = -左侧,  前向 = 前向
        # ════════════════════════════════════════════════════════════
        history_model = [(-left, fwd) for fwd, left in history_poses]  # (前,左)→(右,前)
        target_model = [-target_pose[1], target_pose[0]]               # [前,左]→[右,前]

        # 1. 数据预处理
        imgs = self._preprocess_images(images)
        # 补全history_poses（4个历史 + [0,0]）
        history_model_full = history_model + [[0.0, 0.0]]
        coords = self._preprocess_poses(history_model_full, target_model)

        # 2. 推理（无梯度）
        with torch.no_grad():
            wp_pred, arrive_pred, _, _ = self.model.model(imgs, coords, future_obs=None)

        # 3. 反归一化: 模型输出 (右侧, 前向)
        waypoints_model = wp_pred[0].cpu().numpy() * self.step_scale  # (5, 2), (right, forward)

        # ════════════════════════════════════════════════════════════
        #  坐标系转换: 模型(右侧, 前向) → ROS(前向, 左侧)
        #  前向 = 前向,  左侧 = -右侧
        # ════════════════════════════════════════════════════════════
        waypoints = np.zeros_like(waypoints_model)
        waypoints[:, 0] = waypoints_model[:, 1]  # fwd
        waypoints[:, 1] = -waypoints_model[:, 0] # left = -right

        arrive_confidence = torch.sigmoid(arrive_pred[0, 0]).item()

        # 4. 构造返回结果（全部用ROS约定）
        result = {
            "waypoints": waypoints,
            "arrival_confidence": arrive_confidence,
            "history_poses": history_poses,
            "target_pose": target_pose,
            "step_scale": self.step_scale
        }

        # 5. 保存结果（可选）
        if save_results:
            if not save_dir:
                raise ValueError("save_results=True时必须指定save_dir")
            os.makedirs(save_dir, exist_ok=True)
            # 保存txt/npy/json
            with open(os.path.join(save_dir, "predicted_waypoints.txt"), "w") as f:
                f.write("# CityWalker predicted waypoints\n")
                f.write(f"# step_scale: {self.step_scale}\n")
                f.write(f"# arrival_confidence: {arrive_confidence:.4f}\n")
                for i, (x, y) in enumerate(history_poses):
                    f.write(f"# history_{i}: {x:.3f} {y:.3f} 0\n")
                for i, (x, y) in enumerate(waypoints):
                    f.write(f"t+{i+1} {x:.6f} {y:.6f} 0.0\n")
            np.save(os.path.join(save_dir, "predicted_waypoints.npy"), waypoints)
            with open(os.path.join(save_dir, "prediction.json"), "w") as f:
                json.dump({
                    "step_scale": self.step_scale,
                    "arrival_confidence": arrive_confidence,
                    "history_poses": history_poses,
                    "target_pose": target_pose,
                    "predicted_waypoints": waypoints.tolist()
                }, f, indent=2)
            print(f"[INFO] 预测结果已保存至: {save_dir}")

        return result

    def print_result(self, result):
        """格式化打印预测结果"""
        print("\n" + "="*60)
        print("  CityWalkerFeat 推理结果")
        print("="*60)
        print(f"\n📌 设备        : {self.device}")
        print(f"📌 步长缩放    : {result['step_scale']}")
        print(f"📌 到达置信度  : {result['arrival_confidence']:.4f}")

        print(f"\n📥 输入历史位姿（4个相对位姿，ROS坐标: 前向, 左侧，米）:")
        for i, (x, y) in enumerate(result['history_poses']):
            tag = f"odom ID {3+i}" if i < 4 else "current"
            print(f"   Step {i:>2} ({tag:>10}):  forward = {x:+.4f},  left = {y:+.4f}")

        print(f"\n📥 输入目标位姿  :  forward = {result['target_pose'][0]:+.4f},  left = {result['target_pose'][1]:+.4f}")

        print(f"\n📤 预测路径点（ROS坐标: 前向, 左侧，米）:")
        for i, (fx, ly) in enumerate(result['waypoints']):
            print(f"   t+{i+1}:  forward = {fx:+.4f},  left = {ly:+.4f}")

        total_disp = np.sqrt(result['waypoints'][-1, 0]**2 + result['waypoints'][-1, 1]**2)
        dist_to_target = np.sqrt(
            (result['waypoints'][-1, 0] - result['target_pose'][0])**2 +
            (result['waypoints'][-1, 1] - result['target_pose'][1])**2
        )
        print(f"\n📏 总预测位移   : {total_disp:.4f} m")
        print(f"📏 到目标距离   : {dist_to_target:.4f} m")
        print("="*60)


        

# ===========================
# 初始化模型（全局单例）
# ===========================
# 预加载模型（只加载一次，后续直接调用predict）
citywalker_inferencer = None

def init_model():
    """初始化模型（全局调用一次）"""
    global citywalker_inferencer
    if citywalker_inferencer is None:
        citywalker_inferencer = CityWalkerInferencer()
    return citywalker_inferencer


# ===========================
# 测试示例（可删除）
# ===========================
if __name__ == "__main__":
    # 1. 初始化模型（预加载到GPU）
    inferencer = init_model()


    