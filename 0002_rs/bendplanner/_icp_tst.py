import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

from scipy.spatial import KDTree


def best_fit_transform(A, B):
    """
    计算从点云A到点云B的最优刚体变换
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 中心化点云
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # 计算协方差矩阵
    H = np.dot(A_centered.T, B_centered)

    # 计算SVD
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转矩阵R
    R = np.dot(Vt.T, U.T)

    # 确保R是一个右手坐标系
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 计算平移向量t
    t = centroid_B - np.dot(centroid_A, R)

    return R, t


def icp(A, B, init_pose=None, max_iterations=20, tolerance=1e-6):
    """
    利用点对点ICP算法计算从点云A到点云B的最优刚体变换

    :param A: 源点云 (N x 3)
    :param B: 目标点云 (M x 3)
    :param init_pose: 初始变换矩阵 (4 x 4)
    :param max_iterations: ICP算法的最大迭代次数
    :param tolerance: 收敛容差
    :return: 最优刚体变换矩阵 (4 x 4)，最小均方误差
    """
    assert A.shape[1] == B.shape[1] == 3, "输入的点云应为(N x 3)形状"

    # 初始化
    if init_pose is None:
        T = np.eye(4)
    else:
        T = init_pose

    A_homo = np.column_stack((A, np.ones(A.shape[0]))).T  # (N x 4).T
    prev_error = 0

    # 使用KDTree查找最近邻
    kd_tree = KDTree(B)

    for i in range(max_iterations):
        A_transformed = np.dot(T, A_homo).T[:, :3]
        distances, indices = kd_tree.query(A_transformed)
        B_matched = B[indices]

        # 计算最优刚体变换
        R, t = best_fit_transform(A_transformed, B_matched)

        # 更新变换矩阵
        T_step = np.eye(4)
        T_step[:3, :3] = R
        T_step[:3, 3] = t
        T = np.dot(T_step, T)

        # 计算均方误差
        mean_error = np.sum(distances ** 2) / distances.shape[0]

        # 检查收敛条件
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

        return T, mean_error


def generate_transformed_spiral_point_clouds(num_points=1000, num_turns=2, noise_std=0.05, angle=np.pi / 6,
                                             translation=np.array([5, 5, 5])):
    t_values = np.linspace(0, num_turns * 2 * np.pi, num_points)
    x = t_values * np.cos(t_values)
    y = t_values * np.sin(t_values)
    z = t_values

    source_points = np.vstack((x, y, z)).T

    # 添加噪声
    noise = np.random.normal(0, noise_std, size=(num_points, 3))
    source_points += noise

    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])

    target_points = np.dot(source_points, R) + translation

    return source_points, target_points


def visualize_registration(source, target, R, t):
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()

    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd.points = o3d.utility.Vector3dVector(target)

    source_pcd.paint_uniform_color([1, 0, 0])  # Source点云设置为红色
    target_pcd.paint_uniform_color([0, 1, 0])  # Target点云设置为绿色

    source_transformed = np.dot(source, R.T) + t
    source_transformed_pcd = o3d.geometry.PointCloud()
    source_transformed_pcd.points = o3d.utility.Vector3dVector(source_transformed)
    source_transformed_pcd.paint_uniform_color([0, 0, 1])  # 变换后的Source点云设置为蓝色

    print("红色点云: Source")
    print("绿色点云: Target")
    print("蓝色点云: Transformed Source")

    # 显示原始点云
    o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name='Before ICP')

    # 显示配准后的点云
    o3d.visualization.draw_geometries([source_pcd, source_transformed_pcd, target_pcd], window_name='After ICP')


if __name__ == "__main__":
    # 生成点云
    angle = np.pi / 6
    t = np.array([0.5, 0.5, 0.5])
    source_points, target_points = generate_transformed_spiral_point_clouds(angle=angle, translation=t)
    # 执行ICP算法
    T, mean_error = icp(source_points, target_points, max_iterations=500)

    # 可视化
    visualize_registration(source_points, target_points, T[:3, :3], T[:3, 3])
