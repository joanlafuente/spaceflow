from typing import *


def _without_local_conditioning(kwargs: dict) -> dict:
    kwargs = dict(kwargs)
    kwargs.pop('cond_list', None)
    kwargs.pop('coords_dense_indices', None)
    return kwargs


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(
            model, x_t, t, neg_cond, **_without_local_conditioning(kwargs))
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
