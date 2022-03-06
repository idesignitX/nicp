import render
import torchvision
import numpy as np
import torch
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops import corresponding_points_alignment, knn_points, knn_gather
from pytorch3d.structures import Meshes, Pointclouds
from tqdm import tqdm
import io3d

from local_affine import LocalAffine
from utils import batch_vertex_sample
from utils import convert_mesh_to_pcl, mesh_boundary
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_pointcloud(p1, p2, title):
    # Sample points uniformly from the surface of the mesh.
    x, y, z = p1.points_padded().clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y, c="blue")
    x, y, z = p2.points_padded().clone().detach().cpu().squeeze().unbind(1)
    ax.scatter3D(x, z, -y, c="red")
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 200)
    plt.savefig('./results/' + title + '.jpg')
    plt.show()


def non_rigid_icp_mesh2mesh(
        template_mesh: Meshes,
        target_mesh: Meshes,
        template_lm_index: torch.LongTensor,
        target_lm_index: torch.LongTensor,
        config: dict,
        device=torch.device('cuda:3')
):
    target_pcl = convert_mesh_to_pcl(target_mesh)
    pcl_normal = target_mesh.verts_normals_padded()
    return non_rigid_icp_mesh2pcl(template_mesh, target_pcl, template_lm_index, target_lm_index, config, pcl_normal,
                                  device)


def non_rigid_icp_mesh2pcl(
        template_mesh: Meshes,
        target_pcl: Pointclouds,
        template_lm_index: torch.LongTensor,
        target_lm_index: torch.LongTensor,
        config: dict,
        pcl_normal: torch.FloatTensor = None,
        device=torch.device('cuda:3'),
        out_affine=False,
        in_affine=None
):
    """
        deform template mesh to target pointclouds

        The template mesh and target pcl should be normalized with utils.normalize_mesh api.
        The mesh should look at +z axis, the x define the width of mesh, and the y define the height of mesh
    """

    # init template and target
    template_mesh = template_mesh.to(device)
    template_lm_index = template_lm_index.to(device)
    template_vertex = template_mesh.verts_padded()
    target_pcl = target_pcl.to(device)
    target_lm_index = target_lm_index.to(device)
    target_vertex = target_pcl.points_padded()

    # currently, batch NICP is not supported
    assert target_vertex.shape[0] == 1

    # create mask
    boundary_mask = mesh_boundary(template_mesh.faces_padded()[0], template_vertex.shape[1])
    boundary_mask = boundary_mask.unsqueeze(0).unsqueeze(2)
    inner_mask = torch.logical_not(boundary_mask)

    # masking abnormal points according to the normal seems to be useless, we use distance mask in our framework
    # if pcl_normal is None:
    #     # estimate normal for point cloud
    #     with torch.no_grad():
    #         pcl_normal = pointcloud_normal(target_pcl).unsqueeze(0).repeat(target_vertex.shape[0], 1, 1)

    # initial rigid align according to landmarks
    target_lm = batch_vertex_sample(target_lm_index, target_vertex)
    template_lm = batch_vertex_sample(template_lm_index, template_vertex)
    R, T, s = corresponding_points_alignment(template_lm[:16, :], target_lm[:16, :], estimate_scale=True)
    transformed_vertex = s[:, None, None] * torch.bmm(template_vertex, R) + T[:, None, :]
    # transformed_vertex = template_vertex

    # plot the effect of align
    plot_pointcloud(target_pcl, convert_mesh_to_pcl(template_mesh.update_padded(template_vertex)), 'align_before')
    plot_pointcloud(target_pcl, convert_mesh_to_pcl(template_mesh.update_padded(transformed_vertex)), 'align_after')

    # define the affine transformation model
    template_edges = template_mesh.edges_packed()
    if in_affine is None:
        local_affine_model = LocalAffine(template_vertex.shape[1], template_vertex.shape[0], template_edges).to(device)
    else:
        local_affine_model = in_affine
    optimizer = torch.optim.AdamW([{'params': local_affine_model.parameters()}], lr=1e-4, amsgrad=True)

    # train param config
    inner_iter = config['inner_iter']
    outer_iter = config['outer_iter']
    loop = range(outer_iter)
    log_iter = config['log_iter']
    milestones = [int(float(i) * outer_iter) for i in list(config['milestones'])]
    stiffness_weights = np.array(config['stiffness_weights'])
    landmark_weights = np.array(config['landmark_weights'])
    laplacian_weight = config['laplacian_weight']
    w_idx = 0
    loss_sum = dist_sum = stiff_sum = lm_sum = 0
    new_deformed_meshes = []

    # landmark_part_weights = torch.ones_like(target_lm, device=device)
    # landmark_part_weights[:, 18:27, :] = landmark_part_weights[:, 18:27, :] * 0.6
    # landmark_part_weights[:, 37:46, :] = landmark_part_weights[:, 37:46, :] * 0.5
    # landmark_part_weights[:, 49:, :] = landmark_part_weights[:, 49:, :] * 0.3
    # original 3d model
    # dummy_render = render.create_dummy_render([1, 0, 0], device = device)
    # transformed_mesh = template_mesh.update_padded(transformed_vertex)
    # images = dummy_render(transformed_mesh).squeeze()
    # torchvision.utils.save_image(images.permute(2, 0, 1) / 255, 'test_data/nicp.png')

    for i in loop:
        # init verts
        new_deformed_verts, stiffness = local_affine_model(transformed_vertex, pool_num=0, return_stiff=True)
        new_deformed_lm = batch_vertex_sample(template_lm_index, new_deformed_verts)
        new_deformed_mesh = template_mesh.update_padded(new_deformed_verts)

        # we can randomly sample the target point cloud for speed up
        target_sample_verts = target_vertex

        # use knn to get corresponding points
        knn = knn_points(new_deformed_verts, target_sample_verts)
        close_points = knn_gather(target_sample_verts, knn.idx)[:, :, 0]

        if (i == 0) and (in_affine is None):
            inner_loop = range(100)
        else:
            inner_loop = range(inner_iter)

        for _ in inner_loop:
            optimizer.zero_grad()

            # calculate vert distance and mask
            vert_distance = (new_deformed_verts - close_points) ** 2
            vert_distance_mask = torch.sum(vert_distance, dim=2) < 0.04 ** 2
            weight_mask = torch.logical_and(inner_mask, vert_distance_mask.unsqueeze(2))
            vert_distance = weight_mask * vert_distance
            landmark_distance = (new_deformed_lm - target_lm) ** 2

            # dist loss
            bsize = vert_distance.shape[0]
            vert_distance = vert_distance.view(bsize, -1)
            dist_loss = torch.sum(vert_distance) / bsize
            # landmark loss
            landmark_distance = landmark_distance.view(bsize, -1)
            landmark_loss = torch.sum(landmark_distance) * landmark_weights[w_idx] / bsize
            # stiff loss
            stiffness = stiffness.view(bsize, -1)
            stiffness_loss = torch.sum(stiffness) * stiffness_weights[w_idx] / bsize
            # laplacian loss
            laplacian_loss = mesh_laplacian_smoothing(new_deformed_mesh) * laplacian_weight

            loss = torch.sqrt(dist_loss + landmark_loss + stiffness_loss) + laplacian_loss
            loss.backward()
            optimizer.step()

            # update verts
            loss_sum += loss.item()
            dist_sum += dist_loss.item()
            stiff_sum += stiffness_loss.item()
            lm_sum += landmark_loss.item()
            new_deformed_verts, stiffness = local_affine_model(transformed_vertex, pool_num=0, return_stiff=True)
            new_deformed_lm = batch_vertex_sample(template_lm_index, new_deformed_verts)
            new_deformed_mesh = template_mesh.update_padded(new_deformed_verts)

        # distance = torch.mean(torch.sqrt(torch.sum((old_verts - new_deformed_verts) ** 2, dim=2)))
        if (i + 1) % log_iter == 0:
            iter_num = inner_iter * i + 100.  # first iter is 100
            print("Epoch:%3d/%d, Loss:%.4f, Dist:%.4f, Stiff:%.4f, LM:%.4f" % (
                  i + 1, outer_iter, loss_sum / iter_num, dist_sum / iter_num, stiff_sum / iter_num, lm_sum / iter_num))
            loss_sum = dist_sum = stiff_sum = lm_sum = 0

            new_deformed_verts, _ = local_affine_model(transformed_vertex, return_stiff=True)
            new_deformed_mesh = template_mesh.update_padded(new_deformed_verts)
            new_deformed_meshes.append(new_deformed_mesh)

        if i in milestones:
            w_idx += 1

    # new_deformed_verts, _ = local_affine_model(transformed_vertex, pool_num=0, return_stiff=True)
    # new_deformed_mesh = template_mesh.update_padded(new_deformed_verts)
    if out_affine:
        return new_deformed_meshes, local_affine_model
    else:
        return new_deformed_meshes
