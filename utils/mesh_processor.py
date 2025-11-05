import copy
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from skimage import measure

from utils.dicom_processor import DICOMProcessor


logger = logging.getLogger(__name__)


def mesh_to_pointcloud(mesh: trimesh.Trimesh, num_points: int) -> o3d.geometry.PointCloud:
    """균등 샘플링으로 Trimesh를 Open3D 포인트클라우드로 변환한다."""
    if num_points <= 0:
        raise ValueError("num_points must be a positive integer")

    sampled_points = mesh.sample(num_points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(sampled_points, dtype=np.float64))
    return point_cloud


def icp_align_pointcloud(
    source_pcd: o3d.geometry.PointCloud,
    template_xyz: np.ndarray,
    max_iter: int = 100,
    max_dist: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Open3D ICP를 수행하여 변환 행렬과 정렬된 좌표를 반환한다."""
    if template_xyz.size == 0:
        raise ValueError("template_xyz must contain at least one point")

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(template_xyz.astype(np.float64))

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )

    aligned_pcd = copy.deepcopy(source_pcd)
    aligned_pcd.transform(result.transformation)
    return result.transformation.astype(np.float64), np.asarray(aligned_pcd.points, dtype=np.float64)


def reorder_by_template(
    aligned_xyz: np.ndarray,
    template_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """KDTree로 최근접 정점을 찾아 템플릿 순서에 맞게 재정렬한다."""
    if template_xyz.size == 0:
        raise ValueError("template_xyz must contain at least one point")

    if aligned_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    tree = cKDTree(aligned_xyz)
    distances, indices = tree.query(template_xyz, k=1)
    reordered = aligned_xyz[indices]
    return reordered.astype(np.float32), distances.astype(np.float32)


def estimate_normals_from_pointcloud(verts: np.ndarray, knn: int = 30) -> np.ndarray:
    """점 구름에서 법선을 추정한다."""
    if verts.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(verts)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(knn, len(verts)))
    )
    return np.asarray(point_cloud.normals, dtype=np.float32)


def compute_lung_vertex_labels(
    lung_verts: np.ndarray,
    nodule_verts: np.ndarray,
    distance_threshold: float = 1e-3,
) -> np.ndarray:
    """KD-Tree를 사용하여 결절 라벨을 생성한다."""
    if lung_verts.size == 0:
        return np.empty((0,), dtype=np.int32)

    labels = np.zeros(len(lung_verts), dtype=np.int32)

    if nodule_verts.size == 0:
        return labels

    tree = cKDTree(nodule_verts)
    distances, _ = tree.query(lung_verts, k=1)
    labels[distances < distance_threshold] = 1
    return labels


def zscore_safe(values: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """분산이 0인 경우에도 안전하게 Z-Score를 계산한다."""
    mean = np.mean(values, axis=axis, keepdims=True)
    std = np.std(values, axis=axis, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (values - mean) / std


def build_features_from_xyz(xyz: np.ndarray) -> np.ndarray:
    """좌표 기반 7차원 특징 (xyz + 중심거리 + z-score)을 생성한다."""
    if xyz.size == 0:
        return np.empty((0, 7), dtype=np.float32)

    center = xyz.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(xyz - center, axis=1, keepdims=True)
    dist_min = distances.min()
    dist_max = distances.max()
    norm = (distances - dist_min) / (dist_max - dist_min + 1e-8)

    xyz_z = zscore_safe(xyz, axis=0)
    return np.concatenate([xyz, norm.astype(np.float32), xyz_z.astype(np.float32)], axis=1)


def _sample_with_positive_priority(
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    target_vertices: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """양성 정점을 우선적으로 포함하는 샘플링 전략."""
    if len(positive_indices) == 0 and len(negative_indices) == 0:
        return np.empty((0,), dtype=int)

    if len(positive_indices) == 0:
        if len(negative_indices) >= target_vertices:
            return rng.choice(negative_indices, target_vertices, replace=False)
        return rng.choice(negative_indices, target_vertices, replace=True)

    if len(positive_indices) >= target_vertices:
        return rng.choice(positive_indices, target_vertices, replace=False)

    remaining = target_vertices - len(positive_indices)
    if len(negative_indices) == 0:
        extra = rng.choice(positive_indices, remaining, replace=True)
        return np.concatenate([positive_indices, extra])

    if len(negative_indices) >= remaining:
        sampled_neg = rng.choice(negative_indices, remaining, replace=False)
    else:
        sampled_neg = rng.choice(negative_indices, remaining, replace=True)

    return np.concatenate([positive_indices, sampled_neg])


def resample_features(
    features: np.ndarray,
    target_vertices: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """라벨 없이 정점 특징을 타깃 개수에 맞춰 리샘플링한다."""
    if rng is None:
        rng = np.random.default_rng()

    num_points = len(features)
    if num_points == 0:
        return features, np.empty((0,), dtype=int)

    if num_points == target_vertices:
        indices = np.arange(num_points)
    elif num_points > target_vertices:
        indices = rng.choice(num_points, target_vertices, replace=False)
    else:
        shortage = target_vertices - num_points
        extra = rng.choice(num_points, shortage, replace=True)
        indices = np.concatenate([np.arange(num_points), extra])

    return features[indices], indices


def resample_features_with_labels(
    features: np.ndarray,
    labels: np.ndarray,
    target_vertices: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """양성 정점을 우선 포함하는 리샘플링."""
    if rng is None:
        rng = np.random.default_rng()

    if len(features) == 0:
        return features, labels, np.empty((0,), dtype=int)

    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    selected = _sample_with_positive_priority(positive, negative, target_vertices, rng)

    return features[selected], labels[selected], selected


def preprocess_mesh_single(
    mesh: trimesh.Trimesh,
    target_vertices: int = 2674,
    rng_seed: int = 2025,
    normal_knn: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """단일 메쉬를 (좌표, 특징) 형태로 전처리한다."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    normals = estimate_normals_from_pointcloud(verts, knn=normal_knn)

    if normals.shape[0] != verts.shape[0]:
        normals = np.resize(normals, verts.shape)

    features_with_normals = np.concatenate([verts, normals], axis=1)
    rng = np.random.default_rng(rng_seed)
    resampled_features, indices = resample_features(features_with_normals, target_vertices, rng)
    xyz = resampled_features[:, :3].astype(np.float32)
    features_7 = build_features_from_xyz(xyz)

    return xyz, features_7.astype(np.float32)


def process_lung_nodule_pair(
    lung_mesh_path: str,
    nodule_mesh_path: str,
    distance_threshold: float = 1e-3,
    normal_knn: int = 30,
    fix_vertices: bool = False,
    target_vertices: int = 2674,
    rng_seed: int = 2025,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """폐 메쉬와 결절 메쉬 쌍을 특징/라벨로 변환한다."""
    lung_mesh = trimesh.load(lung_mesh_path, process=False)
    lung_verts = np.asarray(lung_mesh.vertices, dtype=np.float32)

    normals = estimate_normals_from_pointcloud(lung_verts, knn=normal_knn)
    if normals.shape[0] != lung_verts.shape[0]:
        normals = np.zeros_like(lung_verts, dtype=np.float32)

    if os.path.exists(nodule_mesh_path):
        nodule_mesh = trimesh.load(nodule_mesh_path, process=False)
        nodule_verts = np.asarray(nodule_mesh.vertices, dtype=np.float32)
    else:
        nodule_verts = np.empty((0, 3), dtype=np.float32)

    labels = compute_lung_vertex_labels(lung_verts, nodule_verts, distance_threshold)
    features = np.concatenate([lung_verts, normals], axis=1).astype(np.float32)

    if fix_vertices and features.shape[0] != target_vertices:
        rng = np.random.default_rng(rng_seed)
        features, labels, _ = resample_features_with_labels(
            features,
            labels.astype(np.int8),
            target_vertices,
            rng,
        )

    lung_verts = features[:, :3].astype(np.float32)
    return lung_verts, features.astype(np.float32), labels.astype(np.int32)


def process_lung_nodule_directories(
    lung_dir: str,
    nodule_dir: str,
    out_feature_dir: str,
    out_label_dir: str,
    distance_threshold: float = 1e-3,
    normal_knn: int = 30,
    fix_vertices: bool = False,
    target_vertices: int = 2674,
    rng_seed: int = 2025,
    feature_suffix: str = "_features.npy",
    label_suffix: str = "_vertex_labels.npy",
) -> List[str]:
    """폐/결절 PLY 폴더를 매칭하여 특징과 라벨을 저장한다."""
    os.makedirs(out_feature_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    lung_files = sorted([f for f in os.listdir(lung_dir) if f.endswith(".ply")])
    nodule_files = sorted([f for f in os.listdir(nodule_dir) if f.endswith(".ply")])

    def _key(filename: str) -> str:
        base, _ = os.path.splitext(filename)
        return base.split('-')[0]

    lung_map: Dict[str, str] = {_key(f): f for f in lung_files}
    nodule_map: Dict[str, str] = {_key(f): f for f in nodule_files}
    common_keys = sorted(set(lung_map.keys()) & set(nodule_map.keys()))

    processed_keys: List[str] = []

    for key in common_keys:
        lung_mesh_path = os.path.join(lung_dir, lung_map[key])
        nodule_mesh_path = os.path.join(nodule_dir, nodule_map[key])

        _, features, labels = process_lung_nodule_pair(
            lung_mesh_path,
            nodule_mesh_path,
            distance_threshold=distance_threshold,
            normal_knn=normal_knn,
            fix_vertices=fix_vertices,
            target_vertices=target_vertices,
            rng_seed=rng_seed,
        )

        base_name = os.path.splitext(lung_map[key])[0]
        feature_path = os.path.join(out_feature_dir, f"{base_name}{feature_suffix}")
        label_path = os.path.join(out_label_dir, f"{base_name}{label_suffix}")

        np.save(feature_path, features.astype(np.float32))
        np.save(label_path, labels.astype(np.int8))
        processed_keys.append(base_name)

    return processed_keys


def resample_lung_feature_directory(
    feature_dir: str,
    label_dir: str,
    out_feature_dir: str,
    out_label_dir: str,
    target_vertices: int = 2674,
    rng_seed: int = 2025,
    fix_vertices: bool = True,
    feature_suffix: str = "_features.npy",
    label_suffix: str = "_vertex_labels.npy",
) -> List[Tuple[str, int]]:
    """저장된 특징/라벨을 리샘플링하여 (N,7) 특징을 생성한다."""
    os.makedirs(out_feature_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    rng = np.random.default_rng(rng_seed)
    processed: List[Tuple[str, int]] = []

    files = sorted([f for f in os.listdir(feature_dir) if f.endswith(feature_suffix)])

    for filename in files:
        base_name = filename[: -len(feature_suffix)]
        feature_path = os.path.join(feature_dir, filename)
        label_path = os.path.join(label_dir, f"{base_name}{label_suffix}")

        if not os.path.exists(label_path):
            continue

        raw_features = np.load(feature_path)
        raw_labels = np.load(label_path).astype(np.int8).squeeze()

        xyz = raw_features[:, :3].astype(np.float32)
        features_7 = build_features_from_xyz(xyz)

        labels_to_save = raw_labels

        if fix_vertices and features_7.shape[0] != target_vertices:
            features_7, labels_to_save, _ = resample_features_with_labels(
                features_7,
                labels_to_save,
                target_vertices,
                rng,
            )

        np.save(
            os.path.join(out_feature_dir, f"{base_name}{feature_suffix}"),
            features_7.astype(np.float32),
        )
        np.save(
            os.path.join(out_label_dir, f"{base_name}{label_suffix}"),
            labels_to_save.astype(np.int8),
        )

        processed.append((base_name, int(labels_to_save.sum())))

    return processed


class MeshProcessor:
    """3D 메쉬 처리"""

    def __init__(self, template_xyz: Optional[np.ndarray] = None):
        self.dicom_processor = DICOMProcessor()
        self.template_xyz = None if template_xyz is None else np.asarray(template_xyz, dtype=np.float32)

    def create_mesh(self, image, slices, tmpdir):
        """DICOM에서 3D 메쉬 생성"""
        # 리샘플링
        resampled, _ = self.dicom_processor.resample(
            image, slices, new_spacing=[1, 1, 1]
        )
        
        # 폐 마스크 세그멘테이션
        segmented = self.dicom_processor.segment_lung_mask(
            resampled, fill_lung_structures=False
        )
        
        # Marching Cubes로 메쉬 생성
        verts, faces, _, _ = measure.marching_cubes(segmented, level=0.5)

        # 좌표축 재배열: (z, y, x) → (x, y, z)
        verts_reordered = verts[:, [2, 1, 0]]
        
        mesh = trimesh.Trimesh(vertices=verts_reordered, faces=faces)
        
        # 메쉬 저장
        mesh_path = os.path.join(tmpdir, "generated_mesh.ply")
        mesh.export(mesh_path)
        
        return mesh, mesh_path
    
    def extract_features(
        self,
        mesh: trimesh.Trimesh,
        target_vertices: int = 2674,
        rng_seed: int = 2025,
        normal_knn: int = 30,
        icp_max_iter: int = 100,
        icp_max_distance: float = 10.0,
        distance_warning_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """메쉬에서 정점 좌표와 7차원 특징을 추출하고 ICP 정보를 반환한다."""
        if self.template_xyz is None:
            logger.warning(
                "template_xyz가 설정되지 않아 기본 전처리로 대체합니다. "
                "MeshProcessor(template_xyz=...)로 템플릿을 지정하세요."
            )
            xyz, features = preprocess_mesh_single(
                mesh,
                target_vertices=target_vertices,
                rng_seed=rng_seed,
                normal_knn=normal_knn,
            )
            identity = np.eye(4, dtype=np.float32)
            if target_vertices and xyz.shape[0] != target_vertices:
                logger.warning(
                    "템플릿 없이 처리된 정점 수가 기대값(%d)과 다릅니다: %d",
                    target_vertices,
                    xyz.shape[0],
                )
            return (
                xyz.astype(np.float32),
                features.astype(np.float32),
                identity,
                np.zeros((xyz.shape[0],), dtype=np.float32),
            )

        template_points = self.template_xyz.astype(np.float32)
        num_points = template_points.shape[0]
        point_cloud = mesh_to_pointcloud(mesh, num_points)
        transform, aligned_xyz = icp_align_pointcloud(
            point_cloud,
            template_points,
            max_iter=icp_max_iter,
            max_dist=icp_max_distance,
        )

        reordered_xyz, distances = reorder_by_template(aligned_xyz, template_points)

        if reordered_xyz.shape[0] != template_points.shape[0]:
            message = (
                "ICP 정렬 결과 정점 수(%d)가 템플릿 정점 수(%d)와 일치하지 않습니다."
                % (reordered_xyz.shape[0], template_points.shape[0])
            )
            logger.error(message)
            raise ValueError(message)

        if target_vertices and reordered_xyz.shape[0] != target_vertices:
            message = (
                "ICP 정렬 결과 정점 수(%d)가 기대 정점 수(%d)와 일치하지 않습니다."
                % (reordered_xyz.shape[0], target_vertices)
            )
            logger.error(message)
            raise ValueError(message)
        features = build_features_from_xyz(reordered_xyz)

        if distance_warning_threshold is None:
            distance_warning_threshold = icp_max_distance * 0.5

        if distances.size and np.any(distances > distance_warning_threshold):
            logger.warning(
                "ICP 정렬 후 템플릿과 %.2f 이상의 거리를 가진 정점이 존재합니다 (최대 %.2f).",
                distance_warning_threshold,
                float(distances.max()),
            )

        return (
            reordered_xyz.astype(np.float32),
            features.astype(np.float32),
            transform.astype(np.float32),
            distances.astype(np.float32),
        )


def trimesh_to_open3d(mesh: trimesh.Trimesh):
    """Trimesh를 Open3D로 변환"""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    return o3d_mesh


def open3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh):
    """Open3D를 Trimesh로 변환"""
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False
    )


def simplify_mesh(mesh: trimesh.Trimesh, target_faces=60000):
    """메쉬 단순화"""
    o3d_mesh = trimesh_to_open3d(mesh)
    simplified = o3d_mesh.simplify_quadric_decimation(target_faces)
    return open3d_to_trimesh(simplified)
