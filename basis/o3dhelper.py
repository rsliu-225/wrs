import numpy as np
import open3d as o3d
import copy
import basis.trimesh
import sklearn.cluster as skc

# abbreviations
# pnp panda nodepath
# o3d open3d
# pnppcd - a point cloud in the panda nodepath format

def nparray2o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals = False):
    """

    :param nx3nparray_pnts: (n,3) nparray
    :param nx3nparray_nrmls, estimate_normals: if nx3nparray_nrmls is None, check estimate_normals, or else do not work on normals
    :return:

    author: ruishuang, weiwei
    date: 20191210
    """

    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:,:3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd

def o3dpcd2nparray(o3dpcd, return_normals = False):
    """

    :param o3dpcd: open3d point cloud
    :param estimate_normals
    :return:

    author:  weiwei
    date: 20191229, 20200316
    """

    if return_normals:
        if o3dpcd.has_normals():
            return [np.asarray(o3dpcd.points), np.asarray(o3dpcd.normals)]
        else:
            o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
            return [np.asarray(o3dpcd.points), np.asarray(o3dpcd.normals)]
    else:
        return np.asarray(o3dpcd.points)

def cropnx3nparray(nx3nparray, xrng, yrng, zrng):
    """
    crop a n-by-3 nparray

    :param nx3nparray:
    :param xrng, yrng, zrng: [min, max]
    :return:

    author: weiwei
    date: 20191210
    """

    xmask = np.logical_and(nx3nparray[:,0]>xrng[0], nx3nparray[:,0]<xrng[1])
    ymask = np.logical_and(nx3nparray[:,1]>yrng[0], nx3nparray[:,1]<yrng[1])
    zmask = np.logical_and(nx3nparray[:,2]>zrng[0], nx3nparray[:,2]<zrng[1])
    mask = xmask*ymask*zmask
    return nx3nparray[mask]

def cropo3dpcd(o3dpcd, xrng, yrng, zrng):
    """
    crop a o3dpcd

    :param o3dpcd:
    :param xrng, yrng, zrng: [min, max]
    :return:

    author: weiwei
    date: 20191210
    """

    o3dpcdarray = np.asarray(o3dpcd.points)
    xmask = np.logical_and(o3dpcdarray[:,0]>xrng[0], o3dpcdarray[:,0]<xrng[1])
    ymask = np.logical_and(o3dpcdarray[:,1]>yrng[0], o3dpcdarray[:,1]<yrng[1])
    zmask = np.logical_and(o3dpcdarray[:,2]>zrng[0], o3dpcdarray[:,2]<zrng[1])
    mask = xmask*ymask*zmask
    return nparray2o3dpcd(o3dpcdarray[mask])

def o3dmesh2trimesh(o3dmesh):
    """

    :param o3dmesh:
    :return:
    """

    vertices = np.asarray(o3dmesh.vertices)
    faces = np.asarray(o3dmesh.triangles)
    face_normals = np.asarray(o3dmesh.triangle_normals)
    cvterdtrimesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=face_normals)
    return cvterdtrimesh

def __draw_registration_result(source_o3d, target_o3d, transformation):
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def registration_ptpt(src, tgt, downsampling_voxelsize=1, toggledebug = False):
    """
    registrate two point clouds using global registration + local icp
    the correspondence checker for icp is point to point

    :param src: nparray
    :param tgt: nparray
    :param downsampling_voxelsize:
    :param icp_distancethreshold:
    :param debug:
    :return: quality, homomat: quality is measured as RMSE of all inlier correspondences

    author: ruishuang, revised by weiwei
    date: 20191210
    """

    src_o3d = nparray2o3dpcd(src)
    tgt_o3d = nparray2o3dpcd(tgt)

    def __preprocess_point_cloud(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        down_radius_normal = voxel_size * 15
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_radius_normal, max_nn=30))
        radius_feature = voxel_size * 15
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, np.identity(4))

    source_down, source_fpfh = __preprocess_point_cloud(src_o3d, downsampling_voxelsize)
    target_down, target_fpfh = __preprocess_point_cloud(tgt_o3d, downsampling_voxelsize)

    distance_threshold = downsampling_voxelsize * 1.5
    if toggledebug:
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % downsampling_voxelsize)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)

    # result_global = o3d.registration.registration_fass_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
    #     o3d.registration.TransformationEstimationPointToPoint(False), 4, [
    #         o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    #     o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    result_global = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    if toggledebug:
        __draw_registration_result(source_down, target_down, result_global.transformation)

    distance_threshold = downsampling_voxelsize * 0.4
    if toggledebug:
        print(":: Point-to-point ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)

    return _registration_icp_ptpt_o3d(src_o3d, tgt_o3d, result_global.transformation, toggledebug=toggledebug)

    # def _registration_icp_ptpt_o3d(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    #     """
    #
    #     :param src:
    #     :param tgt:
    #     :param maxcorrdist:
    #     :param toggledebug:
    #     :return:
    #
    #     author: weiwei
    #     date: 20191229
    #     """
    #
    #     criteria = o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                        # converge if fitnesss smaller than this
    #                                                        relative_rmse=1e-6,  # converge if rmse smaller than this
    #                                                        max_iteration=2000)
    #     result_icp = o3d.registration.registration_icp(src, tgt, maxcorrdist, inithomomat, criteria=criteria)
    #     if toggledebug:
    #         __draw_registration_result(src, tgt, result_icp.transformation)
    #     return [result_icp.inlier_rmse, result_icp.transformation]
    #
    # result_icp = o3d.registration.registration_icp(
    #     src_o3d, tgt_o3d, distance_threshold, result_global.transformation,
    #     o3d.registration.TransformationEstimationPointToPoint(False))
    # if toggledebug:
    #     __draw_registration_result(src_o3d, tgt_o3d, result_icp.transformation)
    # return [result_icp.inlier_rmse, result_icp.transformation]

def registration_ptpln(src, tgt, downsampling_voxelsize=2, toggledebug = False):
    """
    registrate two point clouds using global registration + local icp
    the correspondence checker for icp is point to plane

    :param src:
    :param tgt:
    :param downsampling_voxelsize:
    :param icp_distancethreshold:
    :param debug:
    :return: quality, homomat: quality is measured as RMSE of all inlier correspondences

    author: ruishuang, revised by weiwei
    date: 20191210
    """

    src_o3d = nparray2o3dpcd(src)
    tgt_o3d = nparray2o3dpcd(tgt)

    def __preprocess_point_cloud(pcd, voxel_size):
        original_radius_normal = voxel_size
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=original_radius_normal, max_nn=30))
        pcd_down = pcd.voxel_down_sample(voxel_size)
        down_radius_normal = voxel_size * 2
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, np.identity(4))

    source_down, source_fpfh = __preprocess_point_cloud(src_o3d, downsampling_voxelsize)
    target_down, target_fpfh = __preprocess_point_cloud(tgt_o3d, downsampling_voxelsize)

    distance_threshold = downsampling_voxelsize * 1.5
    if toggledebug:
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % downsampling_voxelsize)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)

    result_global = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    if toggledebug:
        __draw_registration_result(source_down, target_down, result_global.transformation)

    distance_threshold = downsampling_voxelsize * 0.4
    if toggledebug:
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
    result_icp = o3d.registration.registration_icp(
        src_o3d, tgt_o3d, distance_threshold, result_global.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    if toggledebug:
        __draw_registration_result(src_o3d, tgt_o3d, result_icp.transformation)
    return [result_icp.inlier_rmse, result_icp.transformation]

def registration_icp_ptpt(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    """

    :param src:
    :param tgt:
    :param maxcorrdist:
    :param toggledebug:
    :return:

    author: weiwei
    date: 20191229
    """

    src_o3d = nparray2o3dpcd(src)
    tgt_o3d = nparray2o3dpcd(tgt)
    return _registration_icp_ptpt_o3d(src_o3d, tgt_o3d, inithomomat, maxcorrdist, toggledebug)

def removeoutlier(src_nparray, downsampling_voxelsize=2, nb_points=7, radius=3, estimate_normals = False, toggledebug=False):
    """
    downsample and remove outliers statistically

    :param src:
    :return: cleared o3d point cloud and their normals [pcd_nparray, nrmls_nparray]

    author: weiwei
    date: 20191229
    """

    src_o3d = nparray2o3dpcd(src_nparray, estimate_normals = estimate_normals)
    cl = _removeoutlier_o3d(src_o3d, downsampling_voxelsize, nb_points, radius, toggledebug)
    return o3dpcd2nparray(cl, return_normals=estimate_normals)

def _removeoutlier_o3d(src_o3d, downsampling_voxelsize=2, nb_points=7, radius=3, toggledebug=False):
    """
    downsample and remove outliers statistically

    :param src:
    :return: cleared o3d point cloud

    author: weiwei
    date: 20200316
    """

    if downsampling_voxelsize is None:
        src_o3d_down = src_o3d
    else:
        src_o3d_down = src_o3d.voxel_down_sample(downsampling_voxelsize)
    cl, _ = src_o3d_down.remove_radius_outlier(nb_points=nb_points, radius=radius)
    if toggledebug:
        src_o3d_down.paint_uniform_color([1, 0, 0])
        src_o3d.paint_uniform_color([0, 1, 0])
        cl.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([src_o3d_down, src_o3d, cl])
    return cl

def _registration_icp_ptpt_o3d(src, tgt, inithomomat=np.eye(4), maxcorrdist=2, toggledebug=False):
    """

    :param src:
    :param tgt:
    :param maxcorrdist:
    :param toggledebug:
    :return: [rmse of matched points, size of matched area, homomat]

    author: weiwei
    date: 20191229
    """

    criteria = o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,  # converge if fitnesss smaller than this
                                                       relative_rmse=1e-6, # converge if rmse smaller than this
                                                       max_iteration=2000)
    result_icp = o3d.registration.registration_icp(src, tgt, maxcorrdist, inithomomat, criteria=criteria)
    if toggledebug:
        __draw_registration_result(src, tgt, result_icp.transformation)
    return [result_icp.inlier_rmse, result_icp.fitness, result_icp.transformation]

def clusterpcd(pcd_nparray, pcd_nparray_nrmls = None):
    """
    segment pcd into clusters using the DBSCAN method

    :param pcd_nparray:
    :return:

    author: weiwei
    date: 20200316
    """

    db = skc.DBSCAN(eps=10, min_samples=50, n_jobs=-1).fit(pcd_nparray)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    pcd_nparray_list = []
    pcdnrmls_nparray_list = []
    if pcd_nparray_nrmls is None:
        pcd_nparray_nrmls = np.array([[0, 0, 1]] * pcd_nparray.shape[0])
    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            temppartialpcd = pcd_nparray[class_member_mask & core_samples_mask]
            pcd_nparray_list.append(temppartialpcd)
            temppartialpcdnrmls = pcd_nparray_nrmls[class_member_mask & core_samples_mask]
            pcdnrmls_nparray_list.append(temppartialpcdnrmls)

    return [pcd_nparray_list, pcdnrmls_nparray_list]

def reconstructsurfaces_bp(nppcd, nppcdnrmls = None, radii = [5], doseparation=True):
    """

    :param nppcd:
    :param radii:
    :param doseparation: separate the reconstructed meshes or not when they are disconnected
    :return:
    """

    if doseparation:
        # db = skc.DBSCAN(eps=10, min_samples=50).fit(nppcd)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # unique_labels = set(labels)
        # nppcdlist = []
        # nppcdnrmlslist = []
        # if nppcdnrmls is None:
        #     nppcdnrmls = np.array([[0,0,1]]*nppcd.shape[0])
        # else:
        #     nppcdnrmls = nppcdnrmls
        # for k in unique_labels:
        #     if k == -1:
        #         continue
        #     else:
        #         class_member_mask = (labels == k)
        #         temppartialpcd = nppcd[class_member_mask & core_samples_mask]
        #         nppcdlist.append(temppartialpcd)
        #         temppartialpcdnrmls = nppcdnrmls[class_member_mask & core_samples_mask]
        #         nppcdnrmlslist.append(temppartialpcdnrmls)
        nppcdlist, nppcdnrmlslist = clusterpcd(nppcd, nppcdnrmls)

        tmmeshlist = []
        for i, thisnppcd in enumerate(nppcdlist):
            o3dpcd = nparray2o3dpcd(thisnppcd, nppcdnrmlslist[i])
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3dpcd, o3d.utility.DoubleVector(radii))
            # mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            tmmesh = o3dmesh2trimesh(mesh)
            tmmeshlist.append(tmmesh)
        return tmmeshlist, nppcdlist
    else:
        if nppcdnrmls is None:
            npnrmls = np.array([[0,0,1]]*nppcd.shape[0])
        else:
            npnrmls = nppcdnrmls
        o3dpcd = nparray2o3dpcd(nppcd, npnrmls)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3dpcd, o3d.utility.DoubleVector(radii))
        # mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        tmmesh = o3dmesh2trimesh(mesh)
        return tmmesh

def mergepcd(pnppcd1, pnppcd2, rotmat2, posmat2):
    """
    merge nppcd2 and nppcd1 by rotating and moving nppcd2 using rotmat2 and posmat2

    :param pnppcd1:
    :param pnppcd2:
    :param rotmat2:
    :param posmat2:
    :return:

    author: weiwei
    date: 20200221
    """

    transformednppcd2 = np.dot(rotmat2, pnppcd2.T).T+posmat2
    mergednppcd = np.zeros((len(transformednppcd2)+len(pnppcd1),3))
    mergednppcd[:len(pnppcd1), :] = pnppcd1
    mergednppcd[len(pnppcd1):, :] = transformednppcd2

    return mergednppcd

def getobb(pnppcd):
    """

    :param pnppcd:
    :return: [center_3x1nparray, corners_8x3nparray]

    author:
    date:
    """

    # TODO get the object oriented bounding box of a point cloud using PoindCloud.get_oriented_bounding_box() and OrientedBoundinBox
    pass

