import torch
import pytest
from chamfer3d import chamfer_distance, chamfer_forward, chamfer_backward

@pytest.fixture
def sample_point_clouds():
    xyz1 = torch.rand(32, 1024, 3).cuda()
    xyz2 = torch.rand(32, 1024, 3).cuda()
    return xyz1, xyz2

def test_chamfer_distance(sample_point_clouds):
    xyz1, xyz2 = sample_point_clouds
    dist1, dist2, idx1, idx2 = chamfer_distance(xyz1, xyz2)
    
    assert dist1.shape == (32, 1024)
    assert dist2.shape == (32, 1024)
    assert idx1.shape == (32, 1024)
    assert idx2.shape == (32, 1024)

def test_chamfer_forward(sample_point_clouds):
    xyz1, xyz2 = sample_point_clouds
    dist1 = torch.zeros(32, 1024).cuda()
    dist2 = torch.zeros(32, 1024).cuda()
    idx1 = torch.zeros(32, 1024, dtype=torch.int).cuda()
    idx2 = torch.zeros(32, 1024, dtype=torch.int).cuda()
    
    assert chamfer_forward(xyz1, xyz2, dist1, dist2, idx1, idx2) == 1
    assert dist1.shape == (32, 1024)
    assert dist2.shape == (32, 1024)
    assert idx1.shape == (32, 1024)
    assert idx2.shape == (32, 1024)

def test_chamfer_backward(sample_point_clouds):
    xyz1, xyz2 = sample_point_clouds
    dist1 = torch.zeros(32, 1024).cuda()
    dist2 = torch.zeros(32, 1024).cuda()
    idx1 = torch.zeros(32, 1024, dtype=torch.int).cuda()
    idx2 = torch.zeros(32, 1024, dtype=torch.int).cuda()
    gradxyz1 = torch.zeros_like(xyz1).cuda()
    gradxyz2 = torch.zeros_like(xyz2).cuda()
    graddist1 = torch.rand(32, 1024).cuda()
    graddist2 = torch.rand(32, 1024).cuda()
    
    chamfer_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
    assert chamfer_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2) == 1
    assert gradxyz1.shape == xyz1.shape
    assert gradxyz2.shape == xyz2.shape