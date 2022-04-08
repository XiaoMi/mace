
from enum import Enum

from common import mace_check, ModuleName


class SkelMode(Enum):
    NON_DOMAINS = 0
    DOMAINS = 1


class Skel(Enum):
    V60 = 'V60'
    V65 = 'V65'
    V66 = 'V66'
    V68 = 'V68'
    V69 = 'V69'


class DspType(Enum):
    ADSP = 'ADSP'
    CDSP = 'CDSP'
    HTP = 'HTP'


class SocSkelTable:

    def __init__(self, soc_id, mode, skel, dsp_type):
        self.soc_id = soc_id
        self.mode = mode
        self.skel = skel
        self.dsp_type = dsp_type


SocSkelInfo = [
    SocSkelTable(246, SkelMode.NON_DOMAINS, None, None),
    SocSkelTable(291, SkelMode.NON_DOMAINS, None, None),
    SocSkelTable(292, SkelMode.DOMAINS, Skel.V60, DspType.ADSP),
    SocSkelTable(305, SkelMode.NON_DOMAINS, None, None),
    SocSkelTable(310, SkelMode.NON_DOMAINS, None, None),
    SocSkelTable(311, SkelMode.NON_DOMAINS, None, None),
    SocSkelTable(312, SkelMode.NON_DOMAINS, None, None),
    SocSkelTable(317, SkelMode.DOMAINS, Skel.V60, DspType.CDSP),
    SocSkelTable(318, SkelMode.DOMAINS, Skel.V60, DspType.CDSP),
    SocSkelTable(319, SkelMode.DOMAINS, Skel.V60, DspType.ADSP),
    SocSkelTable(321, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(324, SkelMode.DOMAINS, Skel.V60, DspType.CDSP),
    SocSkelTable(325, SkelMode.DOMAINS, Skel.V60, DspType.CDSP),
    SocSkelTable(326, SkelMode.DOMAINS, Skel.V60, DspType.CDSP),
    SocSkelTable(327, SkelMode.DOMAINS, Skel.V60, DspType.CDSP),
    SocSkelTable(336, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(337, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(339, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(341, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(347, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(352, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(355, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(356, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(360, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(362, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(365, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(366, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(367, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(373, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(377, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(384, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(393, SkelMode.DOMAINS, Skel.V65, DspType.CDSP),
    SocSkelTable(394, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(400, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(407, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),
    SocSkelTable(415, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),  # Custom
    SocSkelTable(450, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),  # Custom
    SocSkelTable(457, SkelMode.DOMAINS, Skel.V66, DspType.CDSP),  # Custom
    SocSkelTable(415, SkelMode.DOMAINS, Skel.V68, DspType.HTP),  # Custom
    SocSkelTable(457, SkelMode.DOMAINS, Skel.V69, DspType.HTP),  # Custom
    SocSkelTable(0, None, None, None)
]


def get_soc_skel_info(soc_id, dsp_type=''):
    for info in SocSkelInfo:
        if dsp_type in ['', info.dsp_type] and info.soc_id == soc_id:
            return info

    mace_check(False, ModuleName.RUN, "Unsupported dsp soc %d" % soc_id)
