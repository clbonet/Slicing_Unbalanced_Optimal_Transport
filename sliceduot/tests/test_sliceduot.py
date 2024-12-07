import pytest
import torch
from sliceduot.sliced_uot import sliced_unbalanced_ot, unbalanced_sliced_ot, kullback_leibler
from sliceduot.utils_sampling import generate_measure

# TODO: tests on output of sliced_emd_backprop




def test_kl_div_output_format():
    n, dim, nproj = 3, 4, 5
    a, _ = generate_measure(n, dim, slice=False)
    b, _ = generate_measure(n, dim, slice=False)
    out = kullback_leibler(a, b)
    print(out.size())
    assert out.size() == torch.Size([])
    out = kullback_leibler(a.repeat(nproj, 1), b.repeat(nproj, 1))
    assert out.size() == torch.Size([nproj])

def test_kl_div_positive_definite():
    n, dim = 3, 4
    a, _ = generate_measure(n, dim, slice=False)
    b, _ = generate_measure(n, dim, slice=False)
    out = kullback_leibler(a, b)
    assert torch.ge(out, torch.zeros_like(out))
    out = kullback_leibler(a, a)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)



def text_kl_div_stable_zero_input():
    n, dim = 3, 4
    a, _ = generate_measure(n, dim, slice=False)
    out = kullback_leibler(torch.zeros_like(a), a)
    assert torch.allclose(out, a.sum(), atol=1e-6)





@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("rho1", [1e-1, 1e0, 1e1])
@pytest.mark.parametrize("rho2", [1e-1, 1e0, 1e1])
def test_sliced_uot_nobug(p, rho1, rho2):
    n, m, dim = 3, 4, 5
    a, x = generate_measure(n, dim, slice=False)
    b, y = generate_measure(m, dim, slice=False)
    sliced_unbalanced_ot(a, b, x, y, p, num_projections=3, rho1=rho1, rho2=rho2, niter=10)
    unbalanced_sliced_ot(a, b, x, y, p, num_projections=3, rho1=rho1, rho2=rho2, niter=10)




@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("rho1", [1e-1, 1e0, 1e1])
@pytest.mark.parametrize("rho2", [1e-1, 1e0, 1e1])
def test_sliced_uot_definite(p, rho1, rho2):
    n, dim, nproj = 3, 4, 5
    a, x = generate_measure(n, dim, slice=False)
    loss, f, g, A, B, _ = sliced_unbalanced_ot(a, a, x, x, p=p, num_projections=nproj, rho1=rho1, rho2=rho2, niter=10)
    assert torch.allclose(loss, torch.zeros([1]), atol=1e-6)
    assert torch.allclose(f, torch.zeros([nproj, n]), atol=1e-6)
    assert torch.allclose(g, torch.zeros([nproj, n]), atol=1e-6)
    assert torch.allclose(A, a.repeat(nproj, 1), atol=1e-6)
    assert torch.allclose(B, a.repeat(nproj, 1), atol=1e-6)


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("rho1", [1e-1, 1e0, 1e1])
@pytest.mark.parametrize("rho2", [1e-1, 1e0, 1e1])
def test_reweighted_uot_definite(p, rho1, rho2):
    n, dim, nproj = 3, 4, 5
    a, x = generate_measure(n, dim, slice=False)
    loss, f, g, A, B, _ = unbalanced_sliced_ot(a, a, x, x, p=p, num_projections=nproj, rho1=rho1, rho2=rho2, niter=10)
    assert torch.allclose(loss, torch.zeros([1]), atol=1e-6)
    assert torch.allclose(f, torch.zeros_like(a), atol=1e-6)
    assert torch.allclose(g, torch.zeros_like(a), atol=1e-6)
    assert torch.allclose(A, a, atol=1e-6)
    assert torch.allclose(B, a, atol=1e-6)


# @pytest.mark.parametrize("p", [1, 1.5, 2])
# @pytest.mark.parametrize("rho1", [1e-1, 1e0, 1e1])
# @pytest.mark.parametrize("rho2", [1e-1, 1e0, 1e1])
# def test_sliced_uot_sensitive_same_sample(p, rho1, rho2):
#     n, m, dim = 3, 4, 5
#     nproj = 2
#     a, x = generate_measure(n, dim)
#     b, y = generate_measure(m, dim)
#     bd, yd = 0.5 * torch.hstack((b,b)), torch.hstack((y,y))
#     loss, f, g, A, B, _ = sliced_unbalanced_ot(a, b, x, y, p, num_projections=nproj, rho1=rho1, rho2=rho2, niter=10, seed_proj=42)
#     lossd, fd, gd, Ad, Bd, _ = sliced_unbalanced_ot(a, bd, x, yd, p, num_projections=nproj, rho1=rho1, rho2=rho2, niter=10, seed_proj=42)
#     assert torch.allclose(loss, lossd)
#     assert torch.allclose(torch.hstack((f,f)), fd)
#     assert torch.allclose(torch.hstack((g,g)), gd)
#     assert torch.allclose(0.5 * torch.hstack((A,A)), Ad)
#     assert torch.allclose(0.5 * torch.hstack((B,B)), Bd)


# def test_reweighted_uot_sensitive_same_sample():
#     # TODO




# def test_sliced_uot_sensitive_zero_mass():
#     # TODO


# def test_reweighted_uot_sensitive_zero_mass():
#     # TODO