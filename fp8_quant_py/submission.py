#!POPCORN leaderboard fp8_quant
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: B200-autotuned with HELION_AUTOTUNE_EFFORT=full
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (autotuned on B200)
    (1, 256, 64): helion.Config(block_sizes=[4], num_stages=1, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
    (4, 512, 128): helion.Config(block_sizes=[16], num_stages=1, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
    (16, 1024, 64): helion.Config(block_sizes=[32], num_stages=1, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
    (1, 4096, 128): helion.Config(block_sizes=[32], num_stages=1, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
    (8, 4096, 128): helion.Config(block_sizes=[32], num_stages=1, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
    # Benchmark shapes (autotuned on B200)
    (16, 4096, 128): helion.Config(block_sizes=[32], num_stages=1, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
    (256, 4096, 128): helion.Config(block_sizes=[8], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], load_eviction_policies=['', '', 'first'], num_stages=2, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
    (256, 8192, 128): helion.Config(block_sizes=[32], num_stages=1, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
    (4096, 7168, 128): helion.Config(block_sizes=[8], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], load_eviction_policies=['first', '', ''], num_stages=5, num_warps=4, pid_type='flat', advanced_controls_file="/opt/booster_pack/fp8_group_quant_6.acf"),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        data: torch.Tensor,        # [N, G] input rows
        scales_out: torch.Tensor,  # [N] output normalization factors
    ) -> torch.Tensor:
        nrows = data.size(0)
        ncols = hl.specialize(data.size(1))
        MAX_VAL = 448.0
        EPS = 1e-10

        qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)

        for rr in hl.tile(nrows):
            row = data[rr, :].to(torch.float32)
            amax = torch.amax(torch.abs(row), -1)
            amax = torch.clamp(amax, min=EPS)
            scale = amax / MAX_VAL
            qout[rr, :] = torch.clamp(row / scale[:, None], -MAX_VAL, MAX_VAL)
            scales_out[rr] = scale

        return qout

    return kernel


_KERNELS: dict = {}


def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    T, H = x.shape
    G = x_s.shape[1]
    gsz = H // G
    N = T * G

    shape = (T, H, gsz)
    if shape not in _KERNELS:
        _KERNELS[shape] = _make_kernel(SHAPE_CONFIGS[shape])
    kernel = _KERNELS[shape]

    flat_in = x.reshape(N, gsz)
    flat_s = x_s.reshape(N)

    flat_q = kernel(flat_in, flat_s)

    x_q[...] = flat_q.reshape(T, H)
    x_s[...] = flat_s.reshape(T, G)
    return x_q, x_s
