from gns.config.settings import settings_fabric
from gns.dataset.citation import Citation

settings = settings_fabric()


class Cora(Citation):
    """
    Cora dataset alias.
    """

    def __init__(self, random_split: bool = False, normalize_x: bool = False, **kwargs):
        super().__init__(
            settings.names.cora,
            random_split=random_split,
            normalize_x=normalize_x,
            **kwargs
        )


def cora_fabric(random_split: bool = False, normalize_x: bool = False, **kwargs):
    return Cora(random_split, normalize_x, **kwargs)
