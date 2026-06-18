import torch


class VolumeBatchSampler:
    def __init__(self, coords, values, batch_size):
        self.coords = coords
        self.values = values
        self.batch_size = int(batch_size)
        self.num_points = coords.shape[0]

    def next(self):
        indices = torch.randint(0, self.num_points, (self.batch_size,))
        return self.coords[indices], self.values[indices]


class VolumePatchSampler:
    def __init__(self, coord_volume, value_volume, patch_size, batch_size):
        if coord_volume.ndim != 4 or coord_volume.shape[-1] != 3:
            raise ValueError("coord_volume must have shape [H, W, D, 3].")
        if value_volume.ndim != 3:
            raise ValueError("value_volume must have shape [H, W, D].")
        if coord_volume.shape[:3] != value_volume.shape:
            raise ValueError("coord_volume and value_volume must share the same spatial shape.")

        patch_size = tuple(int(dim) for dim in patch_size)
        if len(patch_size) != 3:
            raise ValueError("patch_size must have exactly three dimensions.")
        if any(dim <= 1 for dim in patch_size):
            raise ValueError("patch_size dimensions must be greater than 1 for finite differences.")

        self.coord_volume = coord_volume
        self.value_volume = value_volume
        self.patch_size = patch_size
        self.batch_size = int(batch_size)
        self.volume_shape = tuple(int(dim) for dim in value_volume.shape)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        for patch_dim, volume_dim in zip(self.patch_size, self.volume_shape):
            if patch_dim > volume_dim:
                raise ValueError(
                    f"patch_size {self.patch_size} exceeds volume shape {self.volume_shape}."
                )

    def next(self):
        coord_patches = []
        value_patches = []
        for _ in range(self.batch_size):
            slices = []
            for patch_dim, volume_dim in zip(self.patch_size, self.volume_shape):
                max_start = volume_dim - patch_dim
                start = int(torch.randint(0, max_start + 1, (1,)).item())
                slices.append(slice(start, start + patch_dim))
            patch_index = tuple(slices)
            coord_patches.append(self.coord_volume[patch_index])
            value_patches.append(self.value_volume[patch_index])

        coord_batch = torch.stack(coord_patches, dim=0)
        value_batch = torch.stack(value_patches, dim=0).unsqueeze(1)
        return coord_batch, value_batch


class VolumeDataBundle:
    def __init__(
        self,
        coords,
        coord_volume,
        values,
        value_volume,
        shape,
        normalization,
        raw_bytes,
        raw_volume,
        normalized_volume,
        training_volume,
        scale_max,
    ):
        self.coords = coords
        self.coord_volume = coord_volume
        self.values = values
        self.value_volume = value_volume
        self.shape = tuple(shape)
        self.normalization = normalization
        self.raw_bytes = int(raw_bytes)
        self.raw_volume = raw_volume
        self.normalized_volume = normalized_volume
        self.training_volume = training_volume
        self.scale_max = float(scale_max)


def make_volume_bundle(
    coords,
    coord_volume,
    values,
    value_volume,
    shape,
    normalization,
    raw_bytes,
    raw_volume,
    normalized_volume,
    training_volume,
    scale_max,
):
    return VolumeDataBundle(
        coords,
        coord_volume,
        values,
        value_volume,
        shape,
        normalization,
        raw_bytes,
        raw_volume,
        normalized_volume,
        training_volume,
        scale_max,
    )
