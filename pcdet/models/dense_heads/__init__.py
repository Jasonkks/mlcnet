from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_head_box_merge import PointHeadBoxMerge

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'PointHeadBoxMerge': PointHeadBoxMerge,
    'AnchorHeadMulti': AnchorHeadMulti,
}
