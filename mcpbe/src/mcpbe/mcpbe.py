from __future__ import annotations

from .mcpbe_base import MCPBEBase
from .mcpbe_agg import MCPBEAgg
from .mcpbe_break import MCPBEBreak
from .mcpbe_post import MCPBEPost


class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase):
    """Monte Carlo PBE solver:
    - Core framework in MCPBEBase (init, capacity buffers, main loop)
    - Agglomeration logic in AgglomerationMixin
    - Breakage logic (incl. multi-fragment, two-level CDF) in BreakageMixin
    - Post-processing (moments) in PostMixin

    Inherit order ensures Base methods call mixin methods via MRO.
    """
    pass

