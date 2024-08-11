# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import copy
from dataclasses import dataclass
import fnmatch
import itertools
from pathlib import Path
from typing import List, Optional, Tuple

from codegen.cmake_config import *
from codegen.cpp_symbol_map import *


BWD_DQDKDV_PIPELINE_MAP = {
    "kr_ktr_vr" : "ck_tile::BlockFmhaBwdDQDKDVPipelineKRKTRVR",
}

BWD_DQDKDV_PIPELINE_ENUM_MAP = {
    "kr_ktr_vr" : "ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR",
}

FMHA_BWD_KERNEL_HEADER = """// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.\n
// auto generated by generate.py
#include "fmha_bwd.hpp"
"""

FMHA_BWD_DQ_DK_DV_KERNEL_BODY="""
using fmha_dtype_{F_idx} = {F_dtype};

using fmha_block_tile_{F_idx} = ck_tile::
    sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bk1}, {F_bk2}, {F_bk3}, {F_bk4}, {F_bhdq}, {F_bhdv}>;
using fmha_block_warps0_{F_idx} = ck_tile::sequence<{F_rm0}, {F_rn0}, {F_rk0}>;
using fmha_block_warps1_{F_idx} = ck_tile::sequence<{F_rm1}, {F_rn1}, {F_rk1}>;
using fmha_block_warps2_{F_idx} = ck_tile::sequence<{F_rm2}, {F_rn2}, {F_rk2}>;
using fmha_warp_tile0_{F_idx}   = ck_tile::sequence<{F_wm0}, {F_wn0}, {F_wk0}>;
using fmha_warp_tile1_{F_idx}   = ck_tile::sequence<{F_wm1}, {F_wn1}, {F_wk1}>;

// TODO: simplify Gemm0~4BlockWarps in TileFmhaBwdShape
//       G0&G2 -> GSdP
//       G1&G3 -> GdKV
//       G4    -> GdQ
using fmha_bwd_shape_{F_idx} = ck_tile::TileFmhaBwdShape<fmha_block_tile_{F_idx},
                                                         fmha_block_warps0_{F_idx},
                                                         fmha_warp_tile0_{F_idx},
                                                         fmha_block_warps1_{F_idx},
                                                         fmha_warp_tile1_{F_idx},
                                                         fmha_block_warps0_{F_idx},
                                                         fmha_warp_tile0_{F_idx},
                                                         fmha_block_warps1_{F_idx},
                                                         fmha_warp_tile1_{F_idx},
                                                         fmha_block_warps2_{F_idx},
                                                         fmha_warp_tile0_{F_idx}>;

using fmha_bwd_trait_{F_idx} = ck_tile::TileFmhaTraits<{F_spad},
                                                       {F_skpad},
                                                       {F_dpad},
                                                       {F_dvpad},
                                                       {F_bias},
                                                       {F_dbias},
                                                       false,
                                                       false,
                                                       {F_occupancy}>;
using fmha_mask_{F_idx}      = {F_mask};
using fmha_dropout_{F_idx}   = {F_dropout};

using fmha_bwd_pipeline_problem_{F_idx} = ck_tile::BlockFmhaBwdPipelineProblem<
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::QDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::KDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::VDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::GemmDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::LSEDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::AccDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::DDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::BiasDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::RandValOutputDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::ODataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::OGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::QGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::KGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::VGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::BiasGradDataType,
    fmha_bwd_shape_{F_idx},
    {F_mode},
    {F_deterministic},
    fmha_mask_{F_idx},
    fmha_dropout_{F_idx},
    fmha_bwd_trait_{F_idx}>;

using fmha_bwd_pipeline_{F_idx} = {F_pipeline}<fmha_bwd_pipeline_problem_{F_idx}>;

using fmha_bwd_dk_epilogue_{F_idx} = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfig<{F_dtype}>::AccDataType,
                                      typename FmhaBwdTypeConfig<{F_dtype}>::KGradDataType,
                                      false,
                                      false>>;

using fmha_bwd_dv_epilogue_{F_idx} = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfig<{F_dtype}>::AccDataType,
                                      typename FmhaBwdTypeConfig<{F_dtype}>::VGradDataType,
                                      false,
                                      false>>;

using fmha_bwd_dq_dk_dv_kernel_{F_idx} =
    ck_tile::FmhaBwdDQDKDVKernel<ck_tile::FmhaBwdKTilePartitioner<{F_bn0}>,
                                 fmha_bwd_pipeline_{F_idx},
                                 fmha_bwd_dk_epilogue_{F_idx},
                                 fmha_bwd_dv_epilogue_{F_idx}>;

using dq_dk_dv_trait_{F_idx} = fmha_bwd_dq_dk_dv_traits_<{F_hdim},
                                                         {F_dtype},
                                                         {F_mode},
                                                         {F_pipeline_enum},
                                                         fmha_mask_{F_idx},
                                                         fmha_dropout_{F_idx},
                                                         {F_bias},
                                                         {F_dbias},
                                                         {F_spad},
                                                         {F_skpad},
                                                         {F_dpad},
                                                         {F_dvpad},
                                                         {F_deterministic}>;

#include <iostream>

template <>
float fmha_bwd_dq_dk_dv_<dq_dk_dv_trait_{F_idx}>(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    using k_ = fmha_bwd_dq_dk_dv_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;
    auto [kargs, grids]                    = fmha_bwd_dq_dk_dv_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks                  = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
}}

template <>
void fmha_bwd_dq_dk_dv_oneshot_<dq_dk_dv_trait_{F_idx}>(const ck_tile::stream_config& s,
                                                        fmha_bwd_args a)
{{
    using k_                               = fmha_bwd_dq_dk_dv_kernel_{F_idx};
    auto [kargs, grids]                    = fmha_bwd_dq_dk_dv_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks                  = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{{s.stream_id_}});
}}

template <>
std::string fmha_bwd_dq_dk_dv_get_name_<dq_dk_dv_trait_{F_idx}>()
{{
    using k_ = fmha_bwd_dq_dk_dv_kernel_{F_idx};
    return k_::GetName();
}}
"""

FMHA_BWD_API_FILENAME="fmha_bwd_api.cpp"
FMHA_BWD_API="""
#include <iostream>

template <typename dot_do_o_trait_, typename dq_dk_dv_trait_, typename convert_dq_trait_>
float fmha_bwd_(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << fmha_bwd_dq_dk_dv_get_name_<dq_dk_dv_trait_>() << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); }},
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_dq_dk_dv_oneshot_<dq_dk_dv_trait_>(s_, a); }},
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }}
    );
}}

float fmha_bwd(fmha_bwd_traits t, fmha_bwd_args a, const ck_tile::stream_config& s){{
    float r = -1;
{F_dispatch}
    return r;
}}
"""

FMHA_BWD_API_PER_DTYPE="""    {F_if}(t.data_type.compare(\"{F_dtype}\") == 0){{
{F_hdim_case}
    }}
"""
FMHA_BWD_API_PER_HDIM_CASE="""        {F_if} (t.hdim_q <= {F_hdim} && t.hdim_v <= {F_hdim}) {{
{F_inner_dispatch}
        }}
"""

FMHA_BWD_API_INNER_DISPATCH="""            {F_if}((t.is_group_mode == {F_mode}) && ({F_mask_check}) && (t.bias_type == {F_bias_check}) && (t.has_dbias == {F_dbias}) && ({F_dropout_check}) &&
                        ({F_scheck}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck}) && (t.is_deterministic == {F_deterministic})) {{
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_spad1}, {F_dvpad}>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_pipeline_enum}, {F_mask}, {F_dropout}, {F_bias}, {F_dbias}, {F_spad0}, {F_skpad}, {F_dpad}, {F_dvpad}, {F_deterministic}>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_spad1}, {F_dpad}, {F_deterministic}>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }}
"""

@dataclass
class FmhaBwdDQDKDVApiTrait:
    pipeline      : str
    # sync with fmha_bwd_traits<>, to generate fallback calls
    hdim          : str
    dtype         : str  # data type
    mode          : str  # value from MODE_MAP
    bm0           : int  # tile size along q seqlen (block size)
    bn0           : int  # tile size along k seqlen
    bhdq          : int  # q head_dim
    bhdv          : int  # v head_dim
    mask          : str
    bias          : str
    dbias         : str
    dropout       : str
    spad          : str
    skpad         : str
    dpad          : str
    dvpad         : str
    deterministic : str

    def scheck(self, spad1 : str) -> str:
        if self.mode == 'group':
            return 'true' # always support
        elif self.spad == 't' and spad1 == 't':
            return f'a.seqlen_q % {self.bm0} != 0'
        elif self.spad == 'f' and spad1 == 't':
            return f'a.seqlen_q % {self.bm0} == 0 and a.seqlen_q % 64 != 0'
        else: # self.skpad == 'f' and skpad1 == 'f'
            return f'a.seqlen_q % 64 == 0'

    @property
    def skcheck(self) -> str:
        if self.mode == 'group':
            return 'true' # always support
        elif self.skpad == 't':
            return f'a.seqlen_k % {self.bn0} != 0'
        else:
            return f'a.seqlen_k % {self.bn0} == 0'

    @property
    def dcheck(self) -> str:
        if self.dpad == 't': return f'a.hdim_q % {self.bhdq} != 0'
        else :               return f'a.hdim_q % {self.bhdq} == 0'

    @property
    def dvcheck(self) -> str:
        if self.dvpad == 't': return f'a.hdim_v % {self.bhdv} != 0'
        else :                return f'a.hdim_v % {self.bhdv} == 0'

class FmhaBwdApiPool:
    def __init__(self, mask_impl):
        self.dq_dk_dv_pool = dict()
        self.mask_impl = mask_impl

    def register_dq_dk_dv_traits(self, trait : FmhaBwdDQDKDVApiTrait) -> None:
        # TODO: do we need to check duplication?
        if trait.dtype not in self.dq_dk_dv_pool.keys():
            self.dq_dk_dv_pool[trait.dtype] = dict()
        if trait.hdim not in self.dq_dk_dv_pool[trait.dtype].keys():
            self.dq_dk_dv_pool[trait.dtype][trait.hdim] = list()

        self.dq_dk_dv_pool[trait.dtype][trait.hdim].append(copy.copy(trait))

    @property
    def api(self) -> str:
        per_dtypes=str()
        for i, dtype in enumerate(self.dq_dk_dv_pool.keys()):
            per_hdim_case=str()
            for j, hdim in enumerate(self.dq_dk_dv_pool[dtype].keys()):
                traits=self.dq_dk_dv_pool[dtype][hdim]
                hdim_int = int(hdim)
                inners=str()
                for k, trait in enumerate(traits):
                    if_k = 'if' if k == 0 else 'else if'
                    for spad1 in ["t", "f"]:
                        if (spad1 == "f" and (trait.spad == "t" or trait.mode == "group")):
                            continue
                        if (spad1 == "t" and trait.spad == "f" and hdim_int <= 64):
                            continue
                        inners = inners + FMHA_BWD_API_INNER_DISPATCH.format(F_if=if_k, F_mode=MODE_MAP[trait.mode], F_pipeline_enum=BWD_DQDKDV_PIPELINE_ENUM_MAP[trait.pipeline], 
                                    F_mask_check=get_mask_check_map(self.mask_impl)[trait.mask], F_mask=get_mask_map(self.mask_impl)[trait.mask], F_bias_check=BIAS_CHECK_MAP[trait.bias], 
                                    F_bias=BIAS_MAP[trait.bias], F_dbias=BOOL_MAP[trait.dbias], F_dropout_check=DROPOUT_CHECK_MAP[trait.dropout], F_dropout=DROPOUT_MAP[trait.dropout],
                                    F_scheck=trait.scheck(spad1=spad1), F_skcheck=trait.skcheck, F_dcheck=trait.dcheck, F_dvcheck=trait.dvcheck, F_hdim=hdim, F_dtype=DTYPE_MAP[dtype],
                                    F_spad0=BOOL_MAP[trait.spad], F_spad1=BOOL_MAP[spad1], F_skpad=BOOL_MAP[trait.skpad], F_dpad=BOOL_MAP[trait.dpad], F_dvpad=BOOL_MAP[trait.dvpad],
                                    F_deterministic=BOOL_MAP[trait.deterministic])

                if_j = 'if' if j == 0 else 'else if'
                per_hdim_case = per_hdim_case + FMHA_BWD_API_PER_HDIM_CASE.format(F_if=if_j, F_hdim=hdim, F_inner_dispatch=inners)
            if_i = 'if' if i == 0 else 'else if'
            per_dtypes = per_dtypes + FMHA_BWD_API_PER_DTYPE.format(F_if=if_i, F_dtype=dtype, F_hdim_case=per_hdim_case)

        return FMHA_BWD_KERNEL_HEADER + FMHA_BWD_API.format(F_dispatch = per_dtypes)

# GEMM0: Q@K=S^T
# GEMM1: P^T@dO^T=dV(This was chosen as G1 to match fwd, but N1 must be equal to headdim_v)
# GEMM2: dO@V=dP^T(This was chosen as G2 because of the calculation order)
# GEMM3: dS^T@Q^T=dK(Similar to G1, but N3 must be equal to headdim_qk)
# GEMM4: dS@K^T=dQ(N4 must be equal to headdim_qk)
# Is it necessary to distinguish between K0~K4?
@dataclass
class FmhaBwdDQDKDVTileSize:
    F_bm0       : int  # tile size along q seqlen (block size)
    F_bn0       : int  # tile size along k seqlen
    F_bk0       : int  # tile size along gemm0 unroll(F_bhdq)
    F_bk1       : int  # tile size along gemm1 unroll(F_bm0)
    F_bk2       : int  # tile size along gemm2 unroll(F_bhdv)
    F_bk3       : int  # tile size along gemm3 unroll(F_bm0)
    F_bk4       : int  # tile size along gemm4 unroll(F_bn0)
    F_bhdq      : int  # q head_dim
    F_bhdv      : int  # v head_dim
    F_rm0       : int  # number of warps along q seqlen (block warps) in gemm0/gemm2
    F_rn0       : int  # number of warps along k seqlen (block warps) in gemm0/gemm2
    F_rk0       : int  # number of warps along gemm-k (not used) in gemm0/gemm2
    F_rm1       : int  # number of warps along k seqlen (block warps) in gemm1/gemm3
    F_rn1       : int  # number of warps along q seqlen (block warps) in gemm1/gemm3
    F_rk1       : int  # number of warps along gemm-k (not used) in gemm1/gemm3
    F_rm2       : int  # number of warps along k seqlen (block warps) in gemm4
    F_rn2       : int  # number of warps along q seqlen (block warps) in gemm4
    F_rk2       : int  # number of warps along gemm-k (not used) in gemm4
    F_wm0       : int  # warp size along m in gemm0/gemm2/gemm4
    F_wn0       : int  # warp size along n in gemm0/gemm2/gemm4
    F_wk0       : int  # warp size along k in gemm0/gemm2/gemm4
    F_wm1       : int  # warp size along m in gemm1/gemm3
    F_wn1       : int  # warp size along n in gemm1/gemm3
    F_wk1       : int  # warp size along k in gemm1/gemm3
    F_occupancy : int  # occupancy
    @property
    def name(self) -> str:
        return f"b{self.F_bm0}x{self.F_bn0}x{self.F_bk0}x{self.F_bk1}x{self.F_bk2}x{self.F_bk3}x{self.F_bk4}x{self.F_bhdq}x{self.F_bhdv}" +\
        f"_r{self.F_rm0}x{self.F_rn0}x{self.F_rk0}_r{self.F_rm1}x{self.F_rn1}x{self.F_rk1}_r{self.F_rm2}x{self.F_rn2}x{self.F_rk2}" +\
        f"_w{self.F_wm0}x{self.F_wn0}x{self.F_wk0}_w{self.F_wm1}x{self.F_wn1}x{self.F_wk1}_o{self.F_occupancy}"

@dataclass
class FmhaBwdDQDKDVKernel:
    F_idx           : int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim          : int  # hdim
    F_dtype         : str  # data type
    F_tile          : FmhaBwdDQDKDVTileSize
    F_spad          : str  # true/false
    F_skpad         : str  #
    F_dpad          : str  #
    F_dvpad         : str  #
    F_bias          : str  #
    F_dbias         : str  #
    F_dropout       : str  #
    F_mask          : str  # value from MASK_MAP
    F_mode          : str  # value from MODE_MAP
    F_deterministic : str  #
    F_pipeline      : str  #
    mask_impl       : str  #

    @property
    def template(self) -> str:
        return FMHA_BWD_KERNEL_HEADER + \
            FMHA_BWD_DQ_DK_DV_KERNEL_BODY.format(
                F_idx           = self.F_idx,
                F_hdim          = self.F_hdim,
                F_dtype         = DTYPE_MAP[self.F_dtype],
                F_bm0           = self.F_tile.F_bm0,
                F_bn0           = self.F_tile.F_bn0,
                F_bk0           = self.F_tile.F_bk0,
                F_bk1           = self.F_tile.F_bk1,
                F_bk2           = self.F_tile.F_bk2,
                F_bk3           = self.F_tile.F_bk3,
                F_bk4           = self.F_tile.F_bk4,
                F_bhdq          = self.F_tile.F_bhdq,
                F_bhdv          = self.F_tile.F_bhdv,
                F_rm0           = self.F_tile.F_rm0,
                F_rn0           = self.F_tile.F_rn0,
                F_rk0           = self.F_tile.F_rk0,
                F_rm1           = self.F_tile.F_rm1,
                F_rn1           = self.F_tile.F_rn1,
                F_rk1           = self.F_tile.F_rk1,
                F_rm2           = self.F_tile.F_rm2,
                F_rn2           = self.F_tile.F_rn2,
                F_rk2           = self.F_tile.F_rk2,
                F_wm0           = self.F_tile.F_wm0,
                F_wn0           = self.F_tile.F_wn0,
                F_wk0           = self.F_tile.F_wk0,
                F_wm1           = self.F_tile.F_wm1,
                F_wn1           = self.F_tile.F_wn1,
                F_wk1           = self.F_tile.F_wk1,
                F_spad          = BOOL_MAP[self.F_spad],
                F_skpad         = BOOL_MAP[self.F_skpad],
                F_dpad          = BOOL_MAP[self.F_dpad],
                F_dvpad         = BOOL_MAP[self.F_dvpad],
                F_bias          = BIAS_MAP[self.F_bias],
                F_dbias         = BOOL_MAP[self.F_dbias],
                F_dropout       = DROPOUT_MAP[self.F_dropout],
                F_occupancy     = self.F_tile.F_occupancy,
                F_mask          = get_mask_map(self.mask_impl)[self.F_mask],
                F_mode          = MODE_MAP[self.F_mode],
                F_deterministic = BOOL_MAP[self.F_deterministic],
                F_pipeline_enum = BWD_DQDKDV_PIPELINE_ENUM_MAP[self.F_pipeline],
                F_pipeline      = BWD_DQDKDV_PIPELINE_MAP[self.F_pipeline])

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ''
            if self.F_spad == 't': n += 's'
            if self.F_skpad == 't' : n += 'sk'
            if self.F_dpad == 't' : n += 'd'
            if self.F_dvpad == 't' : n += 'dv'
            if n != '' : n = 'p' + n
            return n
        pn = pad_name()
        n = f"fmha_bwd_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_" + self.F_tile.name
        if pn != '' : n += f'_{pn}'
        if self.F_bias != 'no' : n += f'_{self.F_bias}'
        if self.F_dbias == 't' : n += '_dbias'
        if self.F_mask[0:2] == 's_':
            if self.F_mask == 's_mask': n += f'_mask'
        else:
            if self.F_mask != 'no' : n += f'_m{self.F_mask[0]}'
        if self.F_dropout != 'no' : n += f'_{self.F_dropout}'
        if self.F_deterministic == 't' : n += '_deterministic'
        return n

    @property
    def filename(self) -> str:
        return self.name + ".cpp"

    def api_trait(self) -> FmhaBwdDQDKDVApiTrait:
        return FmhaBwdDQDKDVApiTrait(pipeline=self.F_pipeline,
                hdim=str(self.F_hdim),
                dtype=self.F_dtype,
                mode=self.F_mode,
                bm0=self.F_tile.F_bm0,
                bn0=self.F_tile.F_bn0,
                bhdq=self.F_tile.F_bhdq,
                bhdv=self.F_tile.F_bhdv,
                mask=self.F_mask,
                bias=self.F_bias,
                dbias=self.F_dbias,
                dropout=self.F_dropout,
                spad=self.F_spad,
                skpad=self.F_skpad,
                dpad=self.F_dpad,
                dvpad=self.F_dvpad,
                deterministic=self.F_deterministic
                )

# TODO: design a more practical way to do it
# this is current supported tile size & pipeline.
def get_fmha_bwd_dq_dk_dv_tile_ppl_dict_from_dtype(dtype : str) -> Optional[dict]:
    if dtype == 'fp16' or dtype == 'bf16':
        return {
            # '32'  : [FmhaBwdDQDKDVTileSize( 64,  64,  32, 64,  32, 64, 64,  32,  32, 1, 2, 1, 2, 1, 1, 2, 1, 1, 32, 32, 16, 32, 32, 16, 1),
            #             "kr_ktr_vr"],
            #'64'  : [FmhaBwdDQDKDVTileSize( 64, 128,  64, 64,  64, 64, 64,  64,  64, 1, 4, 1, 4, 1, 1, 2, 2, 1, 32, 32, 16, 32, 32, 16, 1),
            #            "kr_ktr_vr"],
            '64'  : [FmhaBwdDQDKDVTileSize( 64, 128,  64, 64,  64, 64, 64,  64,  64, 1, 8, 1, 8, 1, 1, 2, 4, 1, 16, 16, 32, 16, 16, 32, 1),
                        "kr_ktr_vr"],
            # '128' : [FmhaBwdDQDKDVTileSize( 32, 128, 128, 32, 128, 32, 32, 128, 128, 1, 4, 1, 4, 1, 1, 1, 4, 1, 32, 32, 16, 32, 32, 16, 1),
            #             "kr_ktr_vr"],
            # '256' : [FmhaBwdDQDKDVTileSize( 16,  64, 256, 16, 256, 16, 32, 256, 256, 1, 4, 1, 4, 1, 1, 1, 4, 1, 16, 16, 32, 16, 16, 16, 1),
            #             "kr_ktr_vr"]
        }
    else:
        return None

def get_bwd_dq_dk_dv_blobs(kernel_filter : Optional[str], receipt, mask_impl) -> Tuple[FmhaBwdApiPool, List[FmhaBwdDQDKDVKernel]]:
    # TODO: we don't support tuning yet, so pick up one value for pad
    #       support this in future
    gen = list()
    api_pool = FmhaBwdApiPool(mask_impl)

    for dtype in DTYPE_MAP.keys():
        d = get_fmha_bwd_dq_dk_dv_tile_ppl_dict_from_dtype(dtype)
        if d == None:
            continue
        for hdim_str, mode, mask, bias, dbias, dropout, spad, skpad, dpad, dvpad, deterministic in itertools.product(d.keys(), MODE_MAP.keys(), get_mask_map(mask_impl).keys(), BIAS_MAP.keys(), ["t", "f"], DROPOUT_MAP.keys(), ["t", "f"], ["t", "f"], ["t", "f"], ["t", "f"], ["t", "f"]):
            tile = d[hdim_str][0]
            ppl = d[hdim_str][1]
            hdim = int(hdim_str)
            if (mode == "group") and (spad == "f" or skpad == "f"):
                continue
            if ((bias == "no" or bias == "alibi") and dbias == "t"):
                continue
            if ((hdim <= 128 and ("wg16" in dropout)) or (hdim == 256 and ("wg32" in dropout))):
                continue
            k = FmhaBwdDQDKDVKernel(F_idx=0, F_hdim=hdim, F_dtype=dtype, F_tile=tile,
                                F_spad=spad, F_skpad=skpad, F_dpad=dpad, F_dvpad=dvpad,
                                F_bias=bias, F_dbias=dbias, F_dropout=dropout, F_mask=mask, F_mode=mode,
                                F_pipeline=ppl, mask_impl=mask_impl, F_deterministic=deterministic)
            if kernel_filter != None:
                if not fnmatch.fnmatch(k.name, kernel_filter):
                    continue
            if receipt == 2:
                    cond = dtype in ['fp16', 'bf16']
                    cond &= bias in ['no', 'alibi']
                    if not cond:
                        continue
            api_pool.register_dq_dk_dv_traits(k.api_trait())
            gen.append(k)

    return (api_pool, gen)

FMHA_BWD_DOT_DO_O_KERNEL_BODY="""
using fmha_dtype_{F_idx} = {F_dtype};

using fmha_bwd_dot_do_o_trait_{F_idx} =
    ck_tile::TileFmhaBwdOGradDotOTraits<{F_spad}, {F_dvpad}, {F_occupancy}>;

using fmha_bwd_dot_do_o_pipeline_problem_{F_idx} = ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::ODataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::OGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::DDataType,
    /* BlockSize = */ 64,
    {F_hdim},
    {F_mode},
    fmha_bwd_dot_do_o_trait_{F_idx}>;

using fmha_bwd_dot_do_o_{F_idx} =
    typename ck_tile::BlockFmhaBwdOGradDotO<fmha_bwd_dot_do_o_pipeline_problem_{F_idx}>;

using fmha_bwd_dot_do_o_kernel_{F_idx} =
    ck_tile::FmhaBwdOGradDotOKernel<ck_tile::FmhaBwdQTilePartitioner</* BlockSize = */ 64>,
                                    fmha_bwd_dot_do_o_{F_idx}>;

using dot_do_o_trait_{F_idx} =
    fmha_bwd_dot_do_o_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_spad}, {F_dvpad}>;

#include <iostream>

template <>
float fmha_bwd_dot_do_o_<dot_do_o_trait_{F_idx}>(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    using k_ = fmha_bwd_dot_do_o_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;
    auto [kargs, grids]                    = fmha_bwd_dot_do_o_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks                  = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
}}

template <>
void fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_{F_idx}>(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    using k_                               = fmha_bwd_dot_do_o_kernel_{F_idx};
    auto [kargs, grids]                    = fmha_bwd_dot_do_o_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks                  = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{{s.stream_id_}});
}}

template <>
std::string fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_{F_idx}>()
{{
    using k_ = fmha_bwd_dot_do_o_kernel_{F_idx};
    return k_::GetName();
}}
"""

@dataclass
class FmhaBwdOGradDotOKernel:
    F_idx       : int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim      : int  # hdim
    F_dtype     : str  # data type
    F_spad      : str  # true/false
    F_dvpad     : str  #
    F_mode      : str  # value from MODE_MAP
    F_occupancy : int

    @property
    def template(self) -> str:
        return FMHA_BWD_KERNEL_HEADER + \
            FMHA_BWD_DOT_DO_O_KERNEL_BODY.format(
                F_idx       = self.F_idx,
                F_hdim      = self.F_hdim,
                F_dtype     = DTYPE_MAP[self.F_dtype],
                F_spad      = BOOL_MAP[self.F_spad],
                F_dvpad     = BOOL_MAP[self.F_dvpad],
                F_mode      = MODE_MAP[self.F_mode],
                F_occupancy = self.F_occupancy)

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ''
            if self.F_spad == 't': n += 's'
            if self.F_dvpad == 't' : n += 'dv'
            if n != '' : n = 'p' + n
            return n
        pn = pad_name()
        n = f"fmha_bwd_dot_do_o_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_o{self.F_occupancy}"
        if pn != '' : n += f'_{pn}'
        return n

    @property
    def filename(self) -> str:
        return self.name + ".cpp"

def get_bwd_dot_do_o_blobs() -> List[FmhaBwdOGradDotOKernel]:
    # TODO: we don't support tuning yet, so pick up one value for pad/occupancy
    #       support this in future
    def get_occupancy(dtype, hdim):
        return 2

    gen = list()

    for dtype in DTYPE_MAP.keys():
        d = get_fmha_bwd_dq_dk_dv_tile_ppl_dict_from_dtype(dtype)
        if d == None:
            continue
        for hdim_str, mode, spad, dvpad in itertools.product(d.keys(), MODE_MAP.keys(), ["t", "f"], ["t", "f"]):
            hdim = int(hdim_str)
            if (mode == "group" and spad == "f"):
                continue
            k = FmhaBwdOGradDotOKernel(F_idx=0, F_hdim=hdim, F_dtype=dtype,
                                F_spad=spad, F_dvpad=dvpad, F_mode=mode,
                                F_occupancy=get_occupancy(dtype, hdim))
            gen.append(k)

    return gen

FMHA_BWD_CONVERT_DQ_KERNEL_BODY="""
using fmha_dtype_{F_idx} = {F_dtype};

using fmha_block_tile_{F_idx}  = ck_tile::sequence<{F_bm0}, {F_bn0}, {F_hdim}>;
using fmha_block_warps_{F_idx} = ck_tile::sequence<{F_rm}, {F_rn}, {F_rk}>;
using fmha_warp_tile_{F_idx}   = ck_tile::sequence<{F_wm}, {F_wn}, {F_wk}>;

using fmha_bwd_convert_dq_shape_{F_idx} =
    ck_tile::TileFmhaBwdConvertQGradShape<fmha_block_tile_{F_idx},
                                          fmha_block_warps_{F_idx},
                                          fmha_warp_tile_{F_idx}>;

using fmha_bwd_convert_dq_trait_{F_idx} =
    ck_tile::TileFmhaBwdConvertQGradTraits<{F_spad}, {F_dpad}, {F_occupancy}>;

using fmha_bwd_convert_dq_pipeline_problem_{F_idx} =
    ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<
        typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::AccDataType,
        typename FmhaBwdTypeConfig<fmha_dtype_{F_idx}>::QGradDataType,
        fmha_bwd_convert_dq_shape_{F_idx},
        fmha_bwd_convert_dq_trait_{F_idx},
        {F_mode},
        {F_deterministic}>;

using fmha_bwd_convert_dq_{F_idx} =
    typename ck_tile::BlockFmhaBwdConvertQGrad<fmha_bwd_convert_dq_pipeline_problem_{F_idx}>;

using fmha_bwd_convert_dq_kernel_{F_idx} =
    ck_tile::FmhaBwdConvertQGradKernel<ck_tile::FmhaBwdQGradTilePartitioner<{F_bm0}>,
                                       fmha_bwd_convert_dq_{F_idx}>;

using convert_dq_trait_{F_idx} = fmha_bwd_convert_dq_traits_<{F_hdim},
                                                             {F_dtype},
                                                             {F_mode},
                                                             {F_spad},
                                                             {F_dpad},
                                                             {F_deterministic}>;

#include <iostream>

template <>
float fmha_bwd_convert_dq_<convert_dq_trait_{F_idx}>(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    using k_ = fmha_bwd_convert_dq_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;
    auto [kargs, grids]                    = fmha_bwd_convert_dq_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks                  = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
}}

template <>
void fmha_bwd_convert_dq_oneshot_<convert_dq_trait_{F_idx}>(const ck_tile::stream_config& s,
                                                            fmha_bwd_args a)
{{
    using k_                               = fmha_bwd_convert_dq_kernel_{F_idx};
    auto [kargs, grids]                    = fmha_bwd_convert_dq_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks                  = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{{s.stream_id_}});
}}

template <>
std::string fmha_bwd_convert_dq_get_name_<convert_dq_trait_{F_idx}>()
{{
    using k_ = fmha_bwd_convert_dq_kernel_{F_idx};
    return k_::GetName();
}}
"""

@dataclass
class FmhaBwdConvertQGradKernel:
    F_idx           : int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim          : int  # hdim
    F_dtype         : str  # data type
    F_bm0           : int  # tile size along q seqlen (block size)
    F_bn0           : int  # tile size along k seqlen
    F_rm            : int  # number of warps along k seqlen (block warps) in gemm4
    F_rn            : int  # number of warps along q seqlen (block warps) in gemm4
    F_rk            : int  # number of warps along gemm-k (not used) in gemm4
    F_wm            : int  # warp size along m in gemm4
    F_wn            : int  # warp size along n in gemm4
    F_wk            : int  # warp size along k in gemm4
    F_spad          : str  # true/false
    F_dpad          : str  #
    F_mode          : str  # value from MODE_MAP
    F_occupancy     : int  # 
    F_deterministic : str  #

    @property
    def template(self) -> str:
        return FMHA_BWD_KERNEL_HEADER + \
            FMHA_BWD_CONVERT_DQ_KERNEL_BODY.format(
                F_idx           = self.F_idx,
                F_hdim          = self.F_hdim,
                F_dtype         = DTYPE_MAP[self.F_dtype],
                F_bm0           = self.F_bm0,
                F_bn0           = self.F_bn0,
                F_rm            = self.F_rm,
                F_rn            = self.F_rn,
                F_rk            = self.F_rk,
                F_wm            = self.F_wm,
                F_wn            = self.F_wn,
                F_wk            = self.F_wk,
                F_spad          = BOOL_MAP[self.F_spad],
                F_dpad          = BOOL_MAP[self.F_dpad],
                F_mode          = MODE_MAP[self.F_mode],
                F_occupancy     = self.F_occupancy,
                F_deterministic = BOOL_MAP[self.F_deterministic])

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ''
            if self.F_spad == 't': n += 's'
            if self.F_dpad == 't' : n += 'd'
            if n != '' : n = 'p' + n
            return n
        pn = pad_name()
        n = f"fmha_bwd_convert_dq_d{self.F_hdim}_{self.F_dtype}_b{self.F_bm0}x{self.F_bn0}_r{self.F_rm}x{self.F_rn}x{self.F_rk}" +\
        f"_w{self.F_wm}x{self.F_wn}x{self.F_wk}_{self.F_mode}_o{self.F_occupancy}"
        if pn != '' : n += f'_{pn}'
        if self.F_deterministic == 't' : n += f'_deterministic'
        return n

    @property
    def filename(self) -> str:
        return self.name + ".cpp"

def get_bwd_convert_dq_blobs() -> List[FmhaBwdConvertQGradKernel]:
    # TODO: we don't support tuning yet, so pick up one value for pad/occupancy
    #       support this in future
    def get_occupancy(dtype, hdim):
        return 2

    gen = list()

    for dtype in DTYPE_MAP.keys():
        d = get_fmha_bwd_dq_dk_dv_tile_ppl_dict_from_dtype(dtype)
        if d == None:
            continue
        for hdim_str, mode, spad, dpad, deterministic in itertools.product(d.keys(), MODE_MAP.keys(), ["t", "f"], ["t", "f"], ["t", "f"]):
            hdim = int(hdim_str)
            tile = d[hdim_str][0]
            if (mode == "group" and spad == "f"):
                continue
            k = FmhaBwdConvertQGradKernel(F_idx=0, F_hdim=hdim, F_dtype=dtype, F_bm0=64, F_bn0=tile.F_bn0,
                                F_rm=tile.F_rm2, F_rn=tile.F_rn2, F_rk=tile.F_rk2, F_wm=tile.F_wm0, F_wn=tile.F_wn0, F_wk=tile.F_wk0,
                                F_spad=spad, F_dpad=dpad, F_mode=mode, F_occupancy=get_occupancy(dtype, hdim), F_deterministic=deterministic)
            gen.append(k)

    return gen

def write_single_bwd_dq_dk_dv_kernel(kernel: FmhaBwdDQDKDVKernel, autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)

def write_single_bwd_dot_do_o_kernel(kernel: FmhaBwdOGradDotOKernel, autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)

def write_single_bwd_convert_dq_kernel(kernel: FmhaBwdConvertQGradKernel, autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)

def write_bwd_api(api_pool : FmhaBwdApiPool, autogen_dir: Path) -> None:
    (autogen_dir / FMHA_BWD_API_FILENAME).write_text(api_pool.api)

def write_blobs(output_dir : Path, kernel_filter : Optional[str], receipt, mask_impl) -> None:
    kernels = get_bwd_dot_do_o_blobs()
    for kernel in kernels:
        write_single_bwd_dot_do_o_kernel(kernel, output_dir)
    kernels = get_bwd_convert_dq_blobs()
    for kernel in kernels:
        write_single_bwd_convert_dq_kernel(kernel, output_dir)
    api_pool, kernels = get_bwd_dq_dk_dv_blobs(kernel_filter, receipt, mask_impl)
    for kernel in kernels:
        write_single_bwd_dq_dk_dv_kernel(kernel, output_dir)
    write_bwd_api(api_pool, output_dir)

def list_blobs(file_path : Path, kernel_filter : Optional[str], receipt, mask_impl) -> None:
    with file_path.open('a') as f:
        kernels = get_bwd_dot_do_o_blobs()
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        kernels = get_bwd_convert_dq_blobs()
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        _, kernels = get_bwd_dq_dk_dv_blobs(kernel_filter, receipt, mask_impl)
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        f.write(str(file_path.parent / GEN_DIR / FMHA_BWD_API_FILENAME) + "\n")
