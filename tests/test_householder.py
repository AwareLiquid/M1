"""Householder NDIT primitives 的单元测试（M1 里程碑）。

运行：py -3.11 -m pytest tests/test_householder.py -v
"""
import torch
import torch.nn.functional as F

from mt_lnn.householder import (
    householder_matrix,
    householder_apply,
    wy_representation,
    wy_apply,
    unitary_check,
)


def test_householder_is_unitary():
    # 随机方向 → Q 必须酉（QQ^T = I）
    torch.manual_seed(0)
    for _ in range(5):
        v = torch.randn(8)
        Q = householder_matrix(v)
        assert unitary_check(Q) < 1e-6


def test_householder_spectral_radius_one():
    # 谱半径 = 1（酉阵特征值都在单位圆上）
    torch.manual_seed(1)
    v = torch.randn(6)
    Q = householder_matrix(v)
    eig = torch.linalg.eigvals(Q).abs()
    assert (eig - 1.0).abs().max() < 1e-5, f"eig abs = {eig}"


def test_apply_matches_matrix():
    # O(D^2) 应用 == 显式矩阵乘
    torch.manual_seed(2)
    v = torch.randn(8)
    x = torch.randn(4, 8)
    fast = householder_apply(v.unsqueeze(0).expand(4, 8), x)
    slow = torch.einsum("ij,bj->bi", householder_matrix(v), x)
    assert (fast - slow).abs().max() < 1e-6


def test_wy_equals_naive_product():
    # WY 展开 == 朴素 Householder 积（逐反射应用）
    torch.manual_seed(3)
    k, D = 4, 8
    vs = torch.randn(k, D)
    x = torch.randn(3, D)
    # 朴素：P x = Q_0 (Q_1 (... (Q_{k-1} x))) —— 先应用 Q_{k-1}
    naive = x.clone()
    for j in range(k - 1, -1, -1):
        naive = householder_apply(vs[j].unsqueeze(0).expand(3, D), naive)
    # WY
    U, V = wy_representation(vs)
    fast = wy_apply(U, V, x)
    assert (fast - naive).abs().max() < 1e-5


def test_reflector_sign_property():
    # Q v = -v（反射把方向翻到负），Q w = w for w ⊥ v
    torch.manual_seed(4)
    v = torch.randn(8)
    v_unit = F.normalize(v, dim=-1)
    Q = householder_matrix(v)
    assert (Q @ v_unit + v_unit).abs().max() < 1e-6      # Q v = -v
    w = torch.randn(8)
    w = w - (w @ v_unit) * v_unit                        # 正交化
    assert (Q @ w - w).abs().max() < 1e-6                # Q w = w


def test_wy_unitary_product():
    # 多个 Householder 的积仍酉
    torch.manual_seed(5)
    k, D = 3, 6
    vs = torch.randn(k, D)
    U, V = wy_representation(vs)
    # 通过基向量应用构造 P 的显式形式
    Id = torch.eye(D)
    P = wy_apply(U, V, Id)                               # (D, D)
    assert unitary_check(P) < 1e-6
