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


class VolumeDataBundle:
    def __init__(
        self,
        coords,
        values,
        shape,
        normalization,
        raw_bytes,
        raw_volume,
        normalized_volume,
        training_volume,
        scale_max,
    ):
        self.coords = coords
        self.values = values
        self.shape = tuple(shape)
        self.normalization = normalization
        self.raw_bytes = int(raw_bytes)
        self.raw_volume = raw_volume
        self.normalized_volume = normalized_volume
        self.training_volume = training_volume
        self.scale_max = float(scale_max)


def make_volume_bundle(
    coords,
    values,
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
        values,
        shape,
        normalization,
        raw_bytes,
        raw_volume,
        normalized_volume,
        training_volume,
        scale_max,
    )
