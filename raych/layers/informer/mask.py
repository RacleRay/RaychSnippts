import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """B: batch size, H: head, L: length, index: B, H, K, scores: B, H, K, L
        # K for sparse attention n top
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)  # L, L
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])  # B, H, L, L
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)                         # B, H, K, L
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


if __name__ == "__main__":
    pmask = ProbMask(128, 12, 10, 0, torch.ones((128, 12, 10, 10)))
    print(pmask.mask.numpy())