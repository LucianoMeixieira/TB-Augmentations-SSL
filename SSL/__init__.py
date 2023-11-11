'''
This file can be empty, but its presence makes SSL a package, allowing you to import modules from it.

This way, when you import your SSL package in another module, 
you can directly access the SimCLRAugmentations class
without having to specify augment_utils. 

The __all__ variable is a list that defines the public interface of a module. 
It specifies which module or attribute names will be imported when a client imports * from the package.
'''

from .byol import BYOL
from .dino import DINO
from .simclr import SimCLR
from .swav import SwAV

__all__ = [
        'BYOL',
        'DINO',
        'SimCLR',
        'SwAV',
        ]
