// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include <string>
#include <type_traits>

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// dV[seqlen_k, hdim_v] = P^T[seqlen_k, seqlen_q] @ dO^T[hdim_v, seqlen_q]
// dP[seqlen_q, seqlen_k] = dO[seqlen_q, hdim_v] @ V[seqlen_k, hdim_v]
// D[seqlen_q] = rowsum(dO[seqlen_q, hdim_v] * O[seqlen_q, hdim_v])
// dS''[seqlen_q, seqlen_k] = P[seqlen_q, seqlen_k] * (dP[seqlen_q, seqlen_k] - D[seqlen_q])
// dBias[seqlen_q, seqlen_k] = dS'[seqlen_q, seqlen_k] = dS''[seqlen_q, seqlen_k]
// dK[seqlen_k, hdim_q] = dS'^T[seqlen_k, seqlen_q] @ Q^T[hdim_q, seqlen_q] * Scale[1]
// dQ[seqlen_q, hdim_q] = dS'[seqlen_q, seqlen_k] @ K^T[hdim_q, seqlen_k] * Scale[1]

#define REMOVE_ATOMICADD 0
#define REMOVE_GEMM4_LDS_READ 0
#define REMOVE_Q_DO_GLOBAL_LOAD 0
#define REMOVE_Q_DO_LDS_WRITE 0
#define REMOVE_GEMM0_LDS_READ 0
#define REMOVE_SOFTMAX 0
#define REMOVE_GEMM1_LDS_READ 0
#define REMOVE_GEMM2_LDS_READ 0
#define REMOVE_DS 0
#define REMOVE_GEMM3_LDS_READ 0
#define REMOVE_DS_LDS_WRITE 0

namespace ck_tile {

#define GCN_MFMA_INSTR_32 __builtin_amdgcn_mfma_f32_32x32x8bf16_1k
#define GCN_MFMA_INSTR_16 __builtin_amdgcn_mfma_f32_16x16x16bf16_1k

template <ck_tile::index_t InstM,
          ck_tile::index_t InstN,
          ck_tile::index_t InstK>
struct GcnMfmaInstr;

template <>
struct GcnMfmaInstr<32, 32, 8>
{
    using CVecType = float __attribute__((ext_vector_type(16)));// __attribute__((vector_size(16 * sizeof(float)))) float;
    using ABVecType = __attribute__((vector_size(4 * sizeof(bf16_t)))) bf16_t;
    __device__ static void mfma_run(const ABVecType& a, const ABVecType& b, CVecType& c)
    {
        c = GCN_MFMA_INSTR_32(a, b, c, 0, 0, 0);
    }
};

template <>
struct GcnMfmaInstr<16, 16, 16>
{
    using CVecType = float __attribute__((ext_vector_type(4)));// __attribute__((vector_size(16 * sizeof(float)))) float;
    using ABVecType = __attribute__((vector_size(4 * sizeof(bf16_t)))) bf16_t;
    __device__ static void mfma_run(const ABVecType& a, const ABVecType& b, CVecType& c)
    {
        c = GCN_MFMA_INSTR_16(a, b, c, 0, 0, 0);
    }
};

template <typename T, ck_tile::index_t N>
struct QGemm3DOGemm1LdsLoadVecType;

template <>
struct QGemm3DOGemm1LdsLoadVecType<bf16_t, 2>
{
    using Type = float;
};

template <>
struct QGemm3DOGemm1LdsLoadVecType<bf16_t, 4>
{
    using Type = float2;
};

template <typename TilePartitioner_,
          typename FmhaPipeline_,
          typename KGradEpiloguePipeline_,
          typename VGradEpiloguePipeline_>
struct FmhaBwdDQDKDVKernel
{
    using TilePartitioner                         = ck_tile::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using KGradEpiloguePipeline                   = ck_tile::remove_cvref_t<KGradEpiloguePipeline_>;
    using VGradEpiloguePipeline                   = ck_tile::remove_cvref_t<VGradEpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using GemmDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::GemmDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using AccDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::AccDataType>;
    using DDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::DDataType>;
    using RandValOutputDataType =
        ck_tile::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using OGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::OGradDataType>;
    using QGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QGradDataType>;
    using KGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KGradDataType>;
    using VGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VGradDataType>;
    using BiasGradDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasGradDataType>;

    static constexpr bool kIsGroupMode = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
    static constexpr auto BiasEnum     = FmhaPipeline::BiasEnum;
    static constexpr bool kHasBiasGrad = FmhaPipeline::kHasBiasGrad;
    using FmhaMask                     = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    using FmhaDropout                 = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaDropout>;
    static constexpr bool kHasMask    = FmhaMask::IsMasking;
    static constexpr bool kHasDropout = FmhaDropout::IsDropout;
    static constexpr bool kIsStoreRandval  = FmhaDropout::IsStoreRandval;
    static constexpr bool kIsDeterministic = FmhaPipeline::kIsDeterministic;

    // tile/warp
    static constexpr ck_tile::index_t kM0        = FmhaPipeline::kM0;
    static constexpr ck_tile::index_t kN0        = FmhaPipeline::kN0;
    static constexpr ck_tile::index_t kK0        = FmhaPipeline::kK0;
    static constexpr ck_tile::index_t kK1        = FmhaPipeline::kK1;
    static constexpr ck_tile::index_t kK2        = FmhaPipeline::kK2;
    static constexpr ck_tile::index_t kK3        = FmhaPipeline::kK3;
    static constexpr ck_tile::index_t kK4        = FmhaPipeline::kK4;
    static constexpr ck_tile::index_t kQKHeaddim = FmhaPipeline::kQKHeaddim;
    static constexpr ck_tile::index_t kVHeaddim  = FmhaPipeline::kVHeaddim;

    using BlockShape = typename FmhaPipeline::BlockFmhaShape;
    using Gemm0Gemm2BlockWarps = typename BlockShape::Gemm0BlockWarps;
    using Gemm1Gemm3BlockWarps = typename BlockShape::Gemm1BlockWarps;
    using Gemm4BlockWarps = typename BlockShape::Gemm4BlockWarps;
    using Gemm0Gemm2Gemm4WarpTile = typename BlockShape::Gemm0WarpTile;
    using Gemm1Gemm3WarpTile = typename BlockShape::Gemm1WarpTile;

    static constexpr ck_tile::index_t kGemm0Gemm2rm = Gemm0Gemm2BlockWarps::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t kGemm0Gemm2rn = Gemm0Gemm2BlockWarps::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t kGemm1Gemm3rm = Gemm1Gemm3BlockWarps::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t kGemm1Gemm3rn = Gemm1Gemm3BlockWarps::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t kGemm4rm = Gemm4BlockWarps::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t kGemm4rn = Gemm4BlockWarps::at(ck_tile::number<1>{});

    static constexpr ck_tile::index_t kGemm0Gemm2Gemm4WarpM = Gemm0Gemm2Gemm4WarpTile::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t kGemm0Gemm2Gemm4WarpN = Gemm0Gemm2Gemm4WarpTile::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t kGemm0Gemm2Gemm4WarpK = Gemm0Gemm2Gemm4WarpTile::at(ck_tile::number<2>{});
    static constexpr ck_tile::index_t kGemm4WarpK = kGemm0Gemm2Gemm4WarpK / 2;
    static constexpr ck_tile::index_t kGemm0Gemm2Gemm4WarpKInst = kGemm0Gemm2Gemm4WarpK / 2;
    static constexpr ck_tile::index_t kGemm1Gemm3WarpM = Gemm1Gemm3WarpTile::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t kGemm1Gemm3WarpN = Gemm1Gemm3WarpTile::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t kGemm1Gemm3WarpK = Gemm1Gemm3WarpTile::at(ck_tile::number<2>{});
    static constexpr ck_tile::index_t kGemm1Gemm3WarpKInst = kGemm1Gemm3WarpK / 2;
    
    using Gemm0Gemm2Gemm4MfmaInstr = GcnMfmaInstr<kGemm0Gemm2Gemm4WarpM, kGemm0Gemm2Gemm4WarpN, kGemm0Gemm2Gemm4WarpKInst>;
    using Gemm1Gemm3MfmaInstr = GcnMfmaInstr<kGemm1Gemm3WarpM, kGemm1Gemm3WarpN, kGemm1Gemm3WarpKInst>;

    static constexpr ck_tile::index_t kGemm0Gemm2Gemm4AccNum = kGemm0Gemm2Gemm4WarpM * kGemm0Gemm2Gemm4WarpN / 64;
    static constexpr ck_tile::index_t kGemm1Gemm3AccNum = kGemm1Gemm3WarpM * kGemm1Gemm3WarpN / 64;
    static constexpr ck_tile::index_t kGemm4GroupM = kGemm0Gemm2Gemm4WarpM / kGemm0Gemm2Gemm4AccNum * 4;

    static constexpr ck_tile::index_t kGemm0Gemm2KLoops = kQKHeaddim / kGemm0Gemm2Gemm4WarpK;
    static constexpr ck_tile::index_t kGemm1Gemm3KLoops = kN0 / kGemm1Gemm3WarpKInst;
    static constexpr ck_tile::index_t kGemm4KLoops = kN0 / kGemm4WarpK;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        using bfs = typename FmhaPipeline::BlockFmhaShape;
        using gbr = typename bfs::Gemm0BlockWarps;
        using gwt = typename bfs::Gemm0WarpTile;
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadSeqLenK) n += "sk";
            if (kPadHeadDimQ) n += "d";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_bwd_d") + _TS_(bfs::kQKHeaddim) + "_" + _SS_(t2s<QDataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_" +
            "b" + _TS_(bfs::kM0) + "x" + _TS_(bfs::kN0) + "x" + _TS_(bfs::kK0) + "x" +
                    _TS_(bfs::kQKHeaddim) + "x" + _TS_(bfs::kVHeaddim) + "_" +
            "r" + _TS_(gbr::at(ck_tile::number<0>{})) + "x" + _TS_(gbr::at(ck_tile::number<1>{})) + "x" + _TS_(gbr::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(gwt::at(ck_tile::number<0>{})) + "x" + _TS_(gwt::at(ck_tile::number<1>{})) + "x" + _TS_(gwt::at(ck_tile::number<2>{})) + "_" +
            ("o" + _TS_(kBlockPerCu) + "_") + _SS_(FmhaPipeline::name) + (pn.empty() ? "" : "_" + pn) +
            (BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? _SS_("") : (_SS_("_") + BlockAttentionBiasEnumToStr<BiasEnum>::name)) +
            (kHasBiasGrad ? "_dbias" : "") + (kHasMask ? "_" + _SS_(FmhaMask::name) : "") + (kHasDropout ? "_dropout" : "" ) +
            (kIsStoreRandval ? "_storerandval" : "" ) + (kIsDeterministic ? "_deterministic" : "" );
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct FmhaBwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* lse_ptr;
        const void* do_ptr;
        const void* d_ptr;
        void* dq_acc_ptr;
        void* dk_ptr;
        void* dv_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t num_head_q;
        ck_tile::index_t nhead_ratio_qk;
        float raw_scale;
        float scale;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_do;
        ck_tile::index_t stride_dk;
        ck_tile::index_t stride_dv;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_do;
        ck_tile::index_t nhead_stride_lsed;

        ck_tile::index_t batch_stride_lsed;
    };

    struct FmhaBwdCommonBiasKargs
    {
        const void* bias_ptr               = nullptr;
        ck_tile::index_t stride_bias       = 0;
        ck_tile::index_t nhead_stride_bias = 0;
    };

    struct FmhaBwdBatchModeBiasKargs : FmhaBwdCommonBiasKargs
    {
        ck_tile::index_t batch_stride_bias = 0;
    };

    struct FmhaBwdAlibiKargs
    {
        // alibi is batch*nhead*1, no matter in batch/group mode, they are the same
        const void* alibi_slope_ptr;
        ck_tile::index_t alibi_slope_stride; // stride in batch, or 0 for all batch share same slope
    };

    struct FmhaBwdCommonBiasGradKargs
    {
        void* dbias_ptr                     = nullptr;
        ck_tile::index_t stride_dbias       = 0;
        ck_tile::index_t nhead_stride_dbias = 0;
    };

    struct FmhaBwdBatchModeBiasGradKargs : FmhaBwdCommonBiasGradKargs
    {
        ck_tile::index_t batch_stride_dbias = 0;
    };

    struct FmhaBwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmhaBwdCommonDropoutKargs
    {
        void init_dropout(const float p_drop,
                          const std::tuple<uint64_t, uint64_t>& drop_seed_offset,
                          const float raw_scale)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop       = 1.0 / p_undrop;
            scale_rp_undrop = rp_undrop * raw_scale;

            drop_seed   = std::get<0>(drop_seed_offset);
            drop_offset = std::get<1>(drop_seed_offset);
        }
        float rp_undrop             = 1;
        float scale_rp_undrop       = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
        uint64_t drop_seed          = 1;
        uint64_t drop_offset        = 0;
        void* rand_val_ptr          = nullptr;

        ck_tile::index_t stride_randval       = 0;
        ck_tile::index_t nhead_stride_randval = 0;
    };
    struct FmhaBwdBatchModeDropoutKargs : FmhaBwdCommonDropoutKargs
    {
        ck_tile::index_t batch_stride_randval = 0;
    };
    struct FmhaBwdDeterministicKargs
    {
        ck_tile::index_t split_stride_dq_acc = 0;
    };

    struct FmhaBwdBatchModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaBwdBatchModeBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaBwdAlibiKargs,
                                                FmhaBwdEmptyKargs<0>>>,
          std::conditional_t<kHasBiasGrad, FmhaBwdBatchModeBiasGradKargs, FmhaBwdEmptyKargs<1>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<2>>,
          std::conditional_t<kHasDropout, FmhaBwdBatchModeDropoutKargs, FmhaBwdEmptyKargs<3>>,
          std::conditional_t<kIsDeterministic, FmhaBwdDeterministicKargs, FmhaBwdEmptyKargs<4>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_do;
        ck_tile::index_t batch_stride_dk;
        ck_tile::index_t batch_stride_dv;
    };

    struct FmhaBwdGroupModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaBwdCommonBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaBwdAlibiKargs,
                                                FmhaBwdEmptyKargs<0>>>,
          std::conditional_t<kHasBiasGrad, FmhaBwdCommonBiasGradKargs, FmhaBwdEmptyKargs<1>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<2>>,
          std::conditional_t<kHasDropout, FmhaBwdCommonDropoutKargs, FmhaBwdEmptyKargs<3>>,
          std::conditional_t<kIsDeterministic, FmhaBwdDeterministicKargs, FmhaBwdEmptyKargs<4>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaBwdGroupModeKargs, FmhaBwdBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              const void* lse_ptr,
              const void* do_ptr,
              const void* d_ptr,
              void* rand_val_ptr,
              void* dk_ptr,
              void* dv_ptr,
              void* dbias_ptr,
              void* dq_acc_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_randval,
              ck_tile::index_t stride_do,
              ck_tile::index_t stride_dk,
              ck_tile::index_t stride_dv,
              ck_tile::index_t stride_dbias,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_randval,
              ck_tile::index_t nhead_stride_do,
              ck_tile::index_t nhead_stride_lsed,
              ck_tile::index_t nhead_stride_dbias,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_bias,
              ck_tile::index_t batch_stride_randval,
              ck_tile::index_t batch_stride_do,
              ck_tile::index_t batch_stride_lsed,
              ck_tile::index_t batch_stride_dk,
              ck_tile::index_t batch_stride_dv,
              ck_tile::index_t batch_stride_dbias,
              ck_tile::index_t split_stride_dq_acc,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              float p_drop,
              const std::tuple<uint64_t, uint64_t>& drop_seed_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_ptr,
                     do_ptr,
                     d_ptr,
                     dq_acc_ptr,
                     dk_ptr,
                     dv_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     scale,
                     static_cast<float>(scale * ck_tile::log2e_v<>),
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_do,
                     stride_dk,
                     stride_dv,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_do,
                     nhead_stride_lsed,
                     batch_stride_lsed}, // args for common karg
                    {},                  // placeholder for bias
                    {},                  // placeholder for dbias
                    {},                  // placeholder for mask
                    {},                  // placeholder for dropout
                    {},                  // placeholder for deterministic
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_do,
                    batch_stride_dk,
                    batch_stride_dv};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }

        if constexpr(kHasBiasGrad)
        {
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
            kargs.batch_stride_dbias = batch_stride_dbias;
        }

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }

        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, drop_seed_offset, scale);
            if constexpr(kIsStoreRandval)
            {
                kargs.rand_val_ptr         = rand_val_ptr;
                kargs.stride_randval       = stride_randval;
                kargs.nhead_stride_randval = nhead_stride_randval;
                kargs.batch_stride_randval = batch_stride_randval;
            }
        }

        if constexpr(kIsDeterministic)
        {
            kargs.split_stride_dq_acc = split_stride_dq_acc;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              const void* lse_ptr,
              const void* do_ptr,
              const void* d_ptr,
              void* rand_val_ptr,
              void* dk_ptr,
              void* dv_ptr,
              void* dbias_ptr,
              void* dq_acc_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_randval,
              ck_tile::index_t stride_do,
              ck_tile::index_t stride_dk,
              ck_tile::index_t stride_dv,
              ck_tile::index_t stride_dbias,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_randval,
              ck_tile::index_t nhead_stride_do,
              ck_tile::index_t nhead_stride_lsed,
              ck_tile::index_t nhead_stride_dbias,
              ck_tile::index_t batch_stride_lsed,
              ck_tile::index_t split_stride_dq_acc,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              float p_drop,
              const std::tuple<uint64_t, uint64_t>& drop_seed_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_ptr,
                     do_ptr,
                     d_ptr,
                     dq_acc_ptr,
                     dk_ptr,
                     dv_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     scale,
                     static_cast<float>(scale * ck_tile::log2e_v<>),
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_do,
                     stride_dk,
                     stride_dv,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_do,
                     nhead_stride_lsed,
                     batch_stride_lsed}, // args for common karg
                    {},                  // placeholder for bias
                    {},                  // placeholder for dbias
                    {},                  // placeholder for mask
                    {},                  // placeholder for dropout
                    {},                  // placeholder for deterministic
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr)};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasBiasGrad)
        {
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kHasDropout)
        {
            kargs.init_dropout(p_drop, drop_seed_offset, scale);
            if constexpr(kIsStoreRandval)
            {
                kargs.rand_val_ptr         = rand_val_ptr;
                kargs.stride_randval       = stride_randval;
                kargs.nhead_stride_randval = nhead_stride_randval;
            }
        }
        if constexpr(kIsDeterministic)
        {
            kargs.split_stride_dq_acc = split_stride_dq_acc;
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_k_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_k_);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(),
                            KGradEpiloguePipeline::GetSmemSize(),
                            VGradEpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_n, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_k);

        const index_t i_n0 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN0);

        long_index_t batch_offset_q       = 0;
        long_index_t batch_offset_k       = 0;
        long_index_t batch_offset_v       = 0;
        long_index_t batch_offset_bias    = 0;
        long_index_t batch_offset_randval = 0;
        long_index_t batch_offset_do      = 0;
        long_index_t batch_offset_lsed    = 0;
        long_index_t batch_offset_dk      = 0;
        long_index_t batch_offset_dv      = 0;
        long_index_t batch_offset_dbias   = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q    = query_start * kargs.stride_q;
            batch_offset_k    = key_start * kargs.stride_k;
            batch_offset_v    = key_start * kargs.stride_v;
            batch_offset_do   = query_start * kargs.stride_do;
            batch_offset_lsed = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lsed;
            batch_offset_dk   = key_start * kargs.stride_dk;
            batch_offset_dv   = key_start * kargs.stride_dv;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = query_start * kargs.stride_bias;
            }
            if constexpr(kHasBiasGrad)
            {
                batch_offset_dbias = query_start * kargs.stride_dbias;
            }
            else
            {
                batch_offset_dbias = key_start;
            }
            if constexpr(kIsStoreRandval)
            {
                batch_offset_randval = query_start * kargs.stride_randval;
            }

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_k <= i_n0)
            {
                return;
            }
        }
        else
        {
            batch_offset_q    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_do   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_lsed = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lsed;
            batch_offset_dk   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dk;
            batch_offset_dv   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dv;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
            }
            if constexpr(kHasBiasGrad)
            {
                batch_offset_dbias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dbias;
            }
            if constexpr(kIsStoreRandval)
            {
                batch_offset_randval =
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_randval;
            }
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const LSEDataType* lse_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                                     static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                     batch_offset_lsed;
        const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                 batch_offset_lsed;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_k +
                                batch_offset_dk;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_v +
                                batch_offset_dv;
        
        AccDataType* dq_acc_ptr_tmp =
                    reinterpret_cast<AccDataType*>(kargs.dq_acc_ptr) +
                    static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q + batch_offset_q;

        // vector type
        using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
        using bfloat16x4 = __attribute__((__vector_size__(4 * sizeof(bf16_t)))) bf16_t;
        typedef struct __BF16x8_t
        {
            bfloat16x4 xy[2];
        } _BF16x8_t;
        using CVecType = ext_vector_t<float, kGemm0Gemm2Gemm4AccNum>;

        // imm number
        constexpr int32_t m0 = 0x05040100;
        constexpr int32_t m1 = 0x07060302;

        constexpr int32_t kQKHeaddimBytes = kQKHeaddim * sizeof(KDataType);

        // ptr and buffer res
        const int q_range_in_bytes = (kargs.seqlen_q - 1) * kargs.stride_q * sizeof(QDataType) + kQKHeaddimBytes;
        const int dq_acc_range_in_bytes = (kargs.seqlen_q - 1) * kargs.stride_q * sizeof(AccDataType) + kargs.stride_q * sizeof(AccDataType);
        const int k_dk_range_in_bytes = (kargs.seqlen_k - 1) * kargs.stride_k * sizeof(KDataType) + kQKHeaddimBytes;
        const int v_dv_range_in_bytes = (kargs.seqlen_k - 1) * kargs.stride_v * sizeof(VDataType) + kQKHeaddimBytes;
        const int do_range_in_bytes = (kargs.seqlen_q - 1) * kargs.stride_q * sizeof(QDataType) + kQKHeaddimBytes;
        const int lse_range_in_bytes = kargs.seqlen_q * sizeof(LSEDataType);
        const int d_range_in_bytes = kargs.seqlen_q * sizeof(DDataType);
        int32x4_t q_resource = make_wave_buffer_resource(q_ptr, q_range_in_bytes);
        int32x4_t dq_acc_resource = make_wave_buffer_resource(dq_acc_ptr_tmp, dq_acc_range_in_bytes);
        int32x4_t k_resource = make_wave_buffer_resource(k_ptr, k_dk_range_in_bytes);
        int32x4_t dk_resource = make_wave_buffer_resource(dk_ptr, k_dk_range_in_bytes);
        int32x4_t v_resource = make_wave_buffer_resource(v_ptr, v_dv_range_in_bytes);
        int32x4_t dv_resource = make_wave_buffer_resource(dv_ptr, v_dv_range_in_bytes);
        int32x4_t do_resource = make_wave_buffer_resource(do_ptr, do_range_in_bytes);
        int32x4_t lse_resource = make_wave_buffer_resource(lse_ptr, lse_range_in_bytes);
        int32x4_t d_resource = make_wave_buffer_resource(d_ptr, d_range_in_bytes);
        bool lse_d_exec_mask = threadIdx.x < kM0;

#if 0
        if(threadIdx.x == 0)
        {
            printf("block=[%d,%d,%d], dq_acc_range_in_bytes=%d, k_dk_range_in_bytes=%d, v_dv_range_in_bytes=%d, do_range_in_bytes=%d\n",
                type_convert<int>(blockIdx.x),
                type_convert<int>(blockIdx.y),
                type_convert<int>(blockIdx.z),
                dq_acc_range_in_bytes,
                k_dk_range_in_bytes,
                v_dv_range_in_bytes,
                do_range_in_bytes);
        }
#endif

        // prepare k and v_t in smem
        constexpr int k_reg_num = kN0 * kQKHeaddimBytes / (kBlockSize * sizeof(float4));
        constexpr int kv_vec_global = sizeof(float4) / sizeof(KDataType);
        constexpr int k_reg_row = kBlockSize * sizeof(float4) / kQKHeaddimBytes;
        constexpr int num_threads_per_hd_global_load = kQKHeaddimBytes / sizeof(float4);
        constexpr int num_threads_per_hd_global_load_minus_1 = num_threads_per_hd_global_load - 1;
        float4 k_reg[k_reg_num];
        int kv_load_offset = ((threadIdx.x & num_threads_per_hd_global_load_minus_1) * kv_vec_global + (threadIdx.x / num_threads_per_hd_global_load) * kargs.stride_k)  * sizeof(KDataType);
        int k_wave_offset = i_n0 * kargs.stride_k * sizeof(KDataType);
        const int kv_reg_offset = kargs.stride_k * k_reg_row  * sizeof(KDataType);

        constexpr int q_do_padding = sizeof(float4);

        // k do smem offset
        constexpr int kt_reg_gemm0_vt_reg_gemm2_num = kGemm0Gemm2rm * kQKHeaddim * kN0 * sizeof(KDataType) / (kBlockSize * sizeof(_BF16x8_t));
        _BF16x8_t kt_reg_to_gemm0[kt_reg_gemm0_vt_reg_gemm2_num];
        int wave_id = threadIdx.x / 64;
        int wave_lane_id = threadIdx.x % 64;
        int k0_id = wave_lane_id / kGemm0Gemm2Gemm4WarpM;
        int n_id = wave_lane_id % kGemm0Gemm2Gemm4WarpM;
        int n_wave_repeat_id = wave_id % kGemm0Gemm2rn;
        int k_smem_gemm0_offset = n_wave_repeat_id * kGemm0Gemm2Gemm4WarpN * (kQKHeaddim * sizeof(KDataType) + q_do_padding) + n_id * (kQKHeaddim * sizeof(KDataType) + q_do_padding) + k0_id * 16;
        constexpr int k_smem_read_reg_offset = kGemm0Gemm2Gemm4WarpK * sizeof(KDataType);

        // loading offset
        constexpr int q_do_reg_rows = kBlockSize * sizeof(float4) / (kQKHeaddim * sizeof(KDataType));
        int q_do_load_offset = (threadIdx.x & num_threads_per_hd_global_load_minus_1) * kv_vec_global + (threadIdx.x / num_threads_per_hd_global_load) * kargs.stride_q;
        q_do_load_offset *= sizeof(QDataType);
        const int q_do_load_reg_offset = kargs.stride_q * q_do_reg_rows * sizeof(QDataType);
        int q_do_load_wave_offset = 0;

        constexpr int st_acc_gemmk_offset = 4;
        
#pragma unroll
        for(int i_k_reg = 0; i_k_reg < k_reg_num; i_k_reg++)
        {
            k_reg[i_k_reg] = bit_cast<float4>(llvm_amdgcn_raw_buffer_load_fp32x4(k_resource, kv_load_offset, k_wave_offset, 0));
            k_wave_offset += kv_reg_offset;
        }

        constexpr int v_reg_num = kN0 * kVHeaddim * sizeof(VDataType) / (kBlockSize * sizeof(float4));
        int v_wave_offset = i_n0 * kargs.stride_v * sizeof(VDataType);
        float4 v_reg[v_reg_num];

#pragma unroll
        for(int i_v_reg = 0; i_v_reg < v_reg_num; i_v_reg++)
        {
            v_reg[i_v_reg] = bit_cast<float4>(llvm_amdgcn_raw_buffer_load_fp32x4(v_resource, kv_load_offset, v_wave_offset, 0));
            v_wave_offset += kv_reg_offset;
        }

        // prefetch
        constexpr int q_do_global_num = kQKHeaddim * kM0 * sizeof(KDataType) / (kBlockSize * sizeof(float4));
        float4 q_reg[q_do_global_num];
        float4 do_reg[q_do_global_num];

        constexpr int gemm4_groups = 64 / kGemm0Gemm2Gemm4WarpM;
        constexpr int lse_d_reg_offset = 4 * gemm4_groups * sizeof(LSEDataType);
        int lse_d_hbm_offset = threadIdx.x * sizeof(LSEDataType);
        int lse_d_lds_write_offset = threadIdx.x * sizeof(LSEDataType);
        int lse_d_lds_read_offset = k0_id * 4 * sizeof(LSEDataType);
        int lse_d_wave_offset = 0;

#pragma unroll
        for(int i = 0; i < q_do_global_num; i++)
        {
            q_reg[i] = bit_cast<float4>(llvm_amdgcn_raw_buffer_load_fp32x4(q_resource, q_do_load_offset, q_do_load_wave_offset, 0));
            do_reg[i] = bit_cast<float4>(llvm_amdgcn_raw_buffer_load_fp32x4(do_resource, q_do_load_offset, q_do_load_wave_offset, 0));
            q_do_load_wave_offset += q_do_load_reg_offset;
        }

        float lse_reg = 0, d_reg = 0;
        lse_reg = lse_d_exec_mask ? llvm_amdgcn_raw_buffer_load_fp32(lse_resource, lse_d_hbm_offset, lse_d_wave_offset, 0): 0;
        d_reg = lse_d_exec_mask ? llvm_amdgcn_raw_buffer_load_fp32(d_resource, lse_d_hbm_offset, lse_d_wave_offset, 0): 0;
        lse_d_wave_offset += kM0 * sizeof(LSEDataType);

        char* k_smem = smem_ptr;
        int kvqdo_smem_offset = threadIdx.x * sizeof(float4);
        int kvqdo_smem_offset_padding = kvqdo_smem_offset / kN0 * q_do_padding;
        kvqdo_smem_offset += kvqdo_smem_offset_padding;
        constexpr int kv_smem_reg_offset = k_reg_row * (kQKHeaddim * sizeof(KDataType) + q_do_padding);

#pragma unroll
        for(int i_k_reg = 0; i_k_reg < k_reg_num; i_k_reg++)
        {
            *reinterpret_cast<float4*>(k_smem + kvqdo_smem_offset + kv_smem_reg_offset * i_k_reg) = k_reg[i_k_reg];
        }
       
        __syncthreads();

#pragma unroll
        for(int i_kt_reg_gemm0 = 0; i_kt_reg_gemm0 < kt_reg_gemm0_vt_reg_gemm2_num; i_kt_reg_gemm0++)
        {
            kt_reg_to_gemm0[i_kt_reg_gemm0] = *reinterpret_cast<_BF16x8_t*>(k_smem + k_smem_gemm0_offset + k_smem_read_reg_offset * i_kt_reg_gemm0);
        }

        constexpr int k_reg_gemm4_num = kGemm4rm * kN0 * kQKHeaddim * sizeof(KDataType) / (kBlockSize * sizeof(bfloat16x4));
        bfloat16x4 k_reg_to_gemm4[k_reg_gemm4_num];
        int gemm4_n_wave_id = wave_id / kGemm4rm;
        int k_smem_gemm4_offset = n_id * sizeof(KDataType) + k0_id * (kQKHeaddim * sizeof(KDataType) + q_do_padding) * 4 + gemm4_n_wave_id * kGemm0Gemm2Gemm4WarpN * sizeof(KDataType);
        constexpr int k_smem_gemm4_offset_k1 = kQKHeaddim * sizeof(KDataType)  + q_do_padding;
        constexpr int k_smem_gemm4_offset_reg = (kQKHeaddim * sizeof(KDataType) + q_do_padding) * kGemm4WarpK;

#pragma unroll
        for (int i = 0; i < k_reg_gemm4_num; i++)
        {
            k_reg_to_gemm4[i][0] = *reinterpret_cast<unsigned short*>(k_smem + k_smem_gemm4_offset);
            k_reg_to_gemm4[i][1] = *reinterpret_cast<unsigned short*>(k_smem + k_smem_gemm4_offset + k_smem_gemm4_offset_k1);
            k_reg_to_gemm4[i][2] = *reinterpret_cast<unsigned short*>(k_smem + k_smem_gemm4_offset + k_smem_gemm4_offset_k1 * 2);
            k_reg_to_gemm4[i][3] = *reinterpret_cast<unsigned short*>(k_smem + k_smem_gemm4_offset + k_smem_gemm4_offset_k1 * 3);
            k_smem_gemm4_offset += k_smem_gemm4_offset_reg;
        }

#if 0
        printf("thread=%d, k_smem_gemm4_offset=%d, k_smem_gemm4_offset_k1=%d, k_smem_gemm4_offset_reg=%d\n",
            type_convert<int>(threadIdx.x),
            k_smem_gemm4_offset - k_smem_gemm4_offset_reg * k_reg_gemm4_num,
            k_smem_gemm4_offset_k1,
            k_smem_gemm4_offset_reg);
#endif

        __syncthreads();

        char* v_smem = smem_ptr;
#pragma unroll
        for(int i = 0; i < v_reg_num; i++)
        {
            *reinterpret_cast<float4*>(v_smem + kvqdo_smem_offset + kv_smem_reg_offset * i) = v_reg[i];
        }

        __syncthreads();

        _BF16x8_t vt_reg_gemm2[kt_reg_gemm0_vt_reg_gemm2_num];
#pragma unroll
        for(int i = 0; i < kt_reg_gemm0_vt_reg_gemm2_num; i++)
        {
            vt_reg_gemm2[i] = *reinterpret_cast<_BF16x8_t*>(v_smem + k_smem_gemm0_offset + k_smem_read_reg_offset * i);
        }
        
        __syncthreads();

        // prepare core loop
        auto seqlen_q_start = 0;
        auto seqlen_q_end = kargs.seqlen_q;
        index_t i_total_loops = 0;
        // index_t seqlen_q_step = seqlen_q_start;
        const auto num_total_loop = integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0);

#if 0
        if(threadIdx.x == 0)
        {
            printf("block=[%d,%d,%d], num_total_loop=%d, dq_acc_range_in_bytes=%d\n",
                type_convert<int>(blockIdx.x),
                type_convert<int>(blockIdx.y),
                type_convert<int>(blockIdx.z),
                num_total_loop,
                dq_acc_range_in_bytes);
        }
#endif

        // hbm store offset
#if 1 //DQ_ATOMICADD
        int dq_acc_offset = n_id + k0_id * 4 * kargs.stride_q;
        int dq_acc_wave_offset = __builtin_amdgcn_readfirstlane((wave_id / kGemm4rm) * kGemm0Gemm2Gemm4WarpN + (wave_id % kGemm4rm) * kGemm0Gemm2Gemm4WarpM * kargs.stride_q);
        const int stride_dq_acc_in_bytes = kargs.stride_q * sizeof(AccDataType);
        dq_acc_offset *= sizeof(AccDataType);
        dq_acc_wave_offset *= sizeof(AccDataType);
        dq_acc_offset += dq_acc_wave_offset;
        dq_acc_wave_offset = 0;
#endif

        // lds write offset
        // gemm4 ds offset
        constexpr int ds_padding_bytes = 8;
#if !REMOVE_DS_LDS_WRITE
        const int ds_lds_write_offset = n_id * sizeof(KDataType) + k0_id * (kN0 * sizeof(KDataType) + ds_padding_bytes) * 4 + n_wave_repeat_id * kGemm0Gemm2Gemm4WarpM * sizeof(KDataType);
        constexpr int ds_lds_write_reg_offset = kN0 * sizeof(KDataType) + ds_padding_bytes;
        constexpr int ds_lds_gemm_m_group_offset = (kN0 * sizeof(KDataType) + ds_padding_bytes) * kGemm4WarpK;
        constexpr int ds_lds_gemm_m_acc_reg_offset = (kN0 * sizeof(KDataType) + ds_padding_bytes) * kGemm0Gemm2Gemm4WarpM;
#endif

        // lds read offset
        int q_gemm0_do_gemm2_offset = n_id * (kQKHeaddimBytes + q_do_padding) + k0_id * 4 * 2 * sizeof(KDataType);
        constexpr int q_gemm0_do_gemm2_reg_rows = kGemm0Gemm2Gemm4WarpM * kGemm0Gemm2rm;
        constexpr int q_gemm0_do_gemm2_reg_offset = q_gemm0_do_gemm2_reg_rows * (kQKHeaddimBytes + q_do_padding);
        constexpr int q_gemm0_do_gemm2_gemmk_offset = kGemm0Gemm2Gemm4WarpK * sizeof(KDataType);
        constexpr int q_gemm3_do_gemm1_elements = kQKHeaddim / (kGemm1Gemm3WarpN * kGemm1Gemm3rn);
        using q_gemm3_do_gemm1_vec_type = typename QGemm3DOGemm1LdsLoadVecType<QDataType, q_gemm3_do_gemm1_elements>::Type;
        constexpr int q_gemm3_do_gemm1_elements_in_byte = q_gemm3_do_gemm1_elements * sizeof(KDataType);
        int q_gemm3_do_gemm1_offset = n_id * q_gemm3_do_gemm1_elements_in_byte  + k0_id * (kQKHeaddimBytes + q_do_padding) * 4;
        constexpr int q_gemm3_do_gemm1_reg_offset = kQKHeaddimBytes + q_do_padding;
        constexpr int q_gemm3_do_gemm1_gemmk_offset = (kQKHeaddimBytes + q_do_padding) * kGemm1Gemm3WarpKInst;

        // gemm4 ds offset
        int ds_gemm4_offset = n_id * (kN0 * sizeof(KDataType) + ds_padding_bytes) + k0_id * 4 * sizeof(KDataType);
        const int ds_gemm4_m_wave_offset = (wave_id % kGemm4rm) * (kN0 * sizeof(KDataType) + ds_padding_bytes) * kGemm0Gemm2Gemm4WarpM;
        ds_gemm4_offset += ds_gemm4_m_wave_offset;
        constexpr int ds_gemm4_kiter_offset = kGemm4WarpK * sizeof(KDataType);

#if !REMOVE_GEMM4_LDS_READ
        constexpr int dq_acc_reg_offset = (kN0 * sizeof(KDataType) + ds_padding_bytes) * kGemm0Gemm2Gemm4WarpM * kGemm4rm;
#endif

        // lse and d hbm offset and lds write read offset
        auto scale = kargs.scale;

        // acc clear
        constexpr int dv_acc_num = kVHeaddim * kN0 / (kGemm1Gemm3rm * kGemm1Gemm3rn * kGemm1Gemm3WarpM * kGemm1Gemm3WarpN);
        CVecType dv_acc[dv_acc_num];
#pragma unroll
        for(int i = 0; i < dv_acc_num; i++)
        {
            dv_acc[i] = {0};
        }
        
        constexpr int dk_acc_num = kQKHeaddim * kN0 / (kGemm1Gemm3rm * kGemm1Gemm3rn * kGemm1Gemm3WarpM * kGemm1Gemm3WarpN);
        CVecType dk_acc[dk_acc_num];
#pragma unroll
        for(int i = 0; i < dk_acc_num; i++)
        {
            dk_acc[i] = {0};
        }

        // reg definitions
        constexpr int st_acc_num = kM0 * kN0 * sizeof(AccDataType) / (kBlockSize * sizeof(CVecType));
        constexpr int q_gemm0_reg_num = kM0 * kQKHeaddim * sizeof(KDataType) * kGemm0Gemm2rn / (st_acc_num * kBlockSize * sizeof(_BF16x8_t));
        _BF16x8_t q_reg_gemm0[q_gemm0_reg_num];
        CVecType st_acc[st_acc_num]; 
        floatx4 lse_d;
        
        constexpr int do_gemm1_q_gemm3_reg_num = 2;
        constexpr int gemm1_gemm3_k_inner_loop = kM0 / (st_acc_num * kGemm1Gemm3WarpKInst);
        bfloat16x4 pt_reg_gemm1;
        q_gemm3_do_gemm1_vec_type do_reg_gemm1_tmp[4];
        uint2 do_reg_transpose_gemm1[do_gemm1_q_gemm3_reg_num];
        bfloat16x4 do_reg_gemm1[do_gemm1_q_gemm3_reg_num];

        CVecType dpt_acc[st_acc_num];
        
        constexpr int dp_gemm4_reg_num = 2;
        constexpr int dq_acc_num = kQKHeaddim * kM0 / (kGemm4rm * kGemm4rn * kGemm0Gemm2Gemm4WarpM * kGemm0Gemm2Gemm4WarpN);
        bfloat16x4 dp_reg_gemm4[dp_gemm4_reg_num];

        char* lse_smem = smem_ptr;
        char* d_smem = lse_smem + kM0 * sizeof(LSEDataType);
        char* q_smem = d_smem + kM0 * sizeof(DDataType);
        char* do_smem = q_smem + (kM0 * sizeof(KDataType) + q_do_padding) * kQKHeaddim;
        char* ds_smem = do_smem;
#if !REMOVE_Q_DO_LDS_WRITE
#pragma unroll
        for(int i = 0; i < q_do_global_num; i++)
        {
            *reinterpret_cast<float4*>(q_smem + kvqdo_smem_offset + kv_smem_reg_offset * i) = q_reg[i];
            *reinterpret_cast<float4*>(do_smem + kvqdo_smem_offset + kv_smem_reg_offset * i) = do_reg[i];
        }
#endif
        if (lse_d_exec_mask)
        {
            *reinterpret_cast<float*>(lse_smem + lse_d_lds_write_offset) = log2e_v<LSEDataType> * lse_reg;
            *reinterpret_cast<float*>(d_smem + lse_d_lds_write_offset) = d_reg;
        }

        // core loop
        do
        {
            if(i_total_loops < (num_total_loop - 1))
            {
#if !REMOVE_Q_DO_GLOBAL_LOAD
                // q and do: HBM->reg->lds
#pragma unroll
                for(int i = 0; i < q_do_global_num; i++)
                {
                    q_reg[i] = bit_cast<float4>(llvm_amdgcn_raw_buffer_load_fp32x4(q_resource, q_do_load_offset, q_do_load_wave_offset, 0));
                    do_reg[i] = bit_cast<float4>(llvm_amdgcn_raw_buffer_load_fp32x4(do_resource, q_do_load_offset, q_do_load_wave_offset, 0));
                    q_do_load_wave_offset += q_do_load_reg_offset;
                }
#else
#pragma unroll
                for(int i = 0; i < q_do_global_num; i++)
                {
                    q_reg[i] = k_reg[i];
                    q_ptr += q_do_load_reg_offset + q_do_load_offset;
                    do_reg[i] = v_reg[i];
                    do_ptr += q_do_load_reg_offset;
                }
#endif
                // lse and d: HBM->reg->lds
                lse_reg = lse_d_exec_mask ? llvm_amdgcn_raw_buffer_load_fp32(lse_resource, lse_d_hbm_offset, lse_d_wave_offset, 0): 0;
                d_reg = lse_d_exec_mask ? llvm_amdgcn_raw_buffer_load_fp32(d_resource, lse_d_hbm_offset, lse_d_wave_offset, 0): 0;
                lse_d_wave_offset += kM0 * sizeof(LSEDataType);
            
            }
           
            __syncthreads();

            // gemm 0
#pragma unroll
            for(int i = 0; i < st_acc_num; i++)
            {
                st_acc[i] = {0};
            }

#pragma unroll
            for(int i_st_acc = 0; i_st_acc < st_acc_num; i_st_acc++)
            {
#if !REMOVE_GEMM0_LDS_READ
#pragma unroll
                for(int i = 0; i < q_gemm0_reg_num; i++)
                {
                    q_reg_gemm0[i] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_gemmk_offset * i + q_gemm0_do_gemm2_reg_offset * i_st_acc);
                }
#else
#pragma unroll
                for(int i = 0; i < q_gemm0_reg_num; i++)
                {
                    q_reg_gemm0[i] = {static_cast<bf16_t>(i + i_st_acc * q_gemm0_reg_num + 1)};
                }

#endif

#if 1
#pragma unroll
                for(int i = 0; i < kGemm0Gemm2KLoops; i++)
                {
                    Gemm0Gemm2Gemm4MfmaInstr::mfma_run(q_reg_gemm0[i].xy[0], kt_reg_to_gemm0[i].xy[0], st_acc[i_st_acc]);
                    Gemm0Gemm2Gemm4MfmaInstr::mfma_run(q_reg_gemm0[i].xy[1], kt_reg_to_gemm0[i].xy[1], st_acc[i_st_acc]);
                }
#endif
            }

#if 0
            printf("thread=%d, q_gemm0_do_gemm2_offset=%d, q_gemm0_do_gemm2_gemmk_offset=%d, q_reg_gemm0[1].xy[1]=[%f,%f,%f,%f], kt_reg_to_gemm0[1].xy[1]=[%f, %f, %f, %f]\n",
                type_convert<int>(threadIdx.x),
                q_gemm0_do_gemm2_offset,
                q_gemm0_do_gemm2_gemmk_offset,
                type_convert<float>(q_reg_gemm0[1].xy[1][0]),
                type_convert<float>(q_reg_gemm0[1].xy[1][1]),
                type_convert<float>(q_reg_gemm0[1].xy[1][2]),
                type_convert<float>(q_reg_gemm0[1].xy[1][3]),
                type_convert<float>(kt_reg_to_gemm0[1].xy[1][0]),
                type_convert<float>(kt_reg_to_gemm0[1].xy[1][1]),
                type_convert<float>(kt_reg_to_gemm0[1].xy[1][2]),
                type_convert<float>(kt_reg_to_gemm0[1].xy[1][3]));
#endif
#if 0
            printf("thread=%d, st_acc[3]=[%f,%f,%f,%f]\n",
                type_convert<int>(threadIdx.x),
                st_acc[3][0],
                st_acc[3][1],
                st_acc[3][2],
                st_acc[3][3]);
#endif

            // softmax
#pragma unroll
            for(int i_pt = 0; i_pt < st_acc_num; i_pt++)
            {
#pragma unroll
                for(int i_pt_vec = 0; i_pt_vec < kGemm0Gemm2Gemm4AccNum; i_pt_vec += 4)
                {
                    lse_d = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset);
                    lse_smem += lse_d_reg_offset;

                    st_acc[i_pt][0 + i_pt_vec] = exp2(scale * st_acc[i_pt][0 + i_pt_vec] - lse_d[0]);
#if !REMOVE_SOFTMAX
                    st_acc[i_pt][1 + i_pt_vec] = exp2(scale * st_acc[i_pt][1 + i_pt_vec] - lse_d[1]);
                    st_acc[i_pt][2 + i_pt_vec] = exp2(scale * st_acc[i_pt][2 + i_pt_vec] - lse_d[2]);
                    st_acc[i_pt][3 + i_pt_vec] = exp2(scale * st_acc[i_pt][3 + i_pt_vec] - lse_d[3]);
#endif
                }    
            }

#if 0
            if(threadIdx.x == 0)
            {
                printf("thread=%d, st_acc[0]=[%f,%f,%f,%f]\n",
                    type_convert<int>(threadIdx.x),
                    st_acc[0][0],
                    st_acc[0][1],
                    st_acc[0][2],
                    st_acc[0][3]);
                
            }
#endif

            // gemm1
#pragma unroll
            for(int i_st_acc_reg_k = 0; i_st_acc_reg_k < st_acc_num; i_st_acc_reg_k++)
            {
#pragma unroll
                for(int i_st_acc = 0; i_st_acc < gemm1_gemm3_k_inner_loop; i_st_acc++)
                {
#if !REMOVE_GEMM1_LDS_READ
                    do_reg_gemm1_tmp[0] = *reinterpret_cast<q_gemm3_do_gemm1_vec_type*>(do_smem + q_gemm3_do_gemm1_offset);
                    do_reg_gemm1_tmp[1] = *reinterpret_cast<q_gemm3_do_gemm1_vec_type*>(do_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset);
                    do_reg_gemm1_tmp[2] = *reinterpret_cast<q_gemm3_do_gemm1_vec_type*>(do_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset * 2);
                    do_reg_gemm1_tmp[3] = *reinterpret_cast<q_gemm3_do_gemm1_vec_type*>(do_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset * 3);
#else
                    do_reg_gemm1_tmp[0] = {(i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1), (i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1)};
                    do_reg_gemm1_tmp[1] = {(i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 2), (i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1)};
                    do_reg_gemm1_tmp[2] = {(i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 3), (i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1)};
                    do_reg_gemm1_tmp[3] = {(i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 4), (i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1)};
#endif

                    pt_reg_gemm1[0] = type_convert<bf16_t, float>(st_acc[i_st_acc_reg_k][0 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[1] = type_convert<bf16_t, float>(st_acc[i_st_acc_reg_k][1 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[2] = type_convert<bf16_t, float>(st_acc[i_st_acc_reg_k][2 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[3] = type_convert<bf16_t, float>(st_acc[i_st_acc_reg_k][3 + st_acc_gemmk_offset * i_st_acc]);

                    if constexpr(dv_acc_num == 2)
                    {
                        do_reg_transpose_gemm1[0].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1]), bit_cast<uint32_t>(do_reg_gemm1_tmp[0]), m0);
                        do_reg_transpose_gemm1[0].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3]), bit_cast<uint32_t>(do_reg_gemm1_tmp[2]), m0);
                        do_reg_transpose_gemm1[1].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1]), bit_cast<uint32_t>(do_reg_gemm1_tmp[0]), m1);
                        do_reg_transpose_gemm1[1].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3]), bit_cast<uint32_t>(do_reg_gemm1_tmp[2]), m1);

                        do_reg_gemm1[0] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[0]);
                        do_reg_gemm1[1] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[1]);
           
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[0], dv_acc[0]);
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[1], dv_acc[1]);
                    }
                    else if constexpr(dv_acc_num == 4)
                    {
                        do_reg_transpose_gemm1[0].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1].x), bit_cast<uint32_t>(do_reg_gemm1_tmp[0].x), m0);
                        do_reg_transpose_gemm1[0].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3].x), bit_cast<uint32_t>(do_reg_gemm1_tmp[2].x), m0);
                        do_reg_transpose_gemm1[1].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1].x), bit_cast<uint32_t>(do_reg_gemm1_tmp[0].x), m1);
                        do_reg_transpose_gemm1[1].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3].x), bit_cast<uint32_t>(do_reg_gemm1_tmp[2].x), m1);

                        do_reg_gemm1[0] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[0]);
                        do_reg_gemm1[1] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[1]);
           
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[0], dv_acc[0]);
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[1], dv_acc[1]);
                        
                        do_reg_transpose_gemm1[0].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1].y), bit_cast<uint32_t>(do_reg_gemm1_tmp[0].y), m0);
                        do_reg_transpose_gemm1[0].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3].y), bit_cast<uint32_t>(do_reg_gemm1_tmp[2].y), m0);
                        do_reg_transpose_gemm1[1].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1].y), bit_cast<uint32_t>(do_reg_gemm1_tmp[0].y), m1);
                        do_reg_transpose_gemm1[1].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3].y), bit_cast<uint32_t>(do_reg_gemm1_tmp[2].y), m1);

                        do_reg_gemm1[0] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[0]);
                        do_reg_gemm1[1] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[1]);
           
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[0], dv_acc[2]);
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[1], dv_acc[3]);
                    }

                    do_smem += q_gemm3_do_gemm1_gemmk_offset;
                }
            }

#if 0
            printf("thread=%d, q_gemm3_do_gemm1_offset=%d\n",
                type_convert<int>(threadIdx.x),
                q_gemm3_do_gemm1_offset);

            printf("thread=%d, dv_acc[0]=[%f,%f,%f,%f]\n",
                type_convert<int>(threadIdx.x),
                dv_acc[0][0],
                dv_acc[0][1],
                dv_acc[0][2],
                dv_acc[0][3]);
#endif

            do_smem -= q_gemm3_do_gemm1_gemmk_offset * (st_acc_num * gemm1_gemm3_k_inner_loop);

            // gemm 2
#pragma unroll
            for(int i = 0; i < st_acc_num; i++)
            {
                dpt_acc[i] = {0};
            }

#pragma unroll
            for(int i_dpt_acc = 0; i_dpt_acc < st_acc_num; i_dpt_acc++)
            {
#if !REMOVE_GEMM2_LDS_READ
#pragma unroll
                for(int i = 0; i < q_gemm0_reg_num; i++)
                {
                    q_reg_gemm0[i] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_gemmk_offset * i + q_gemm0_do_gemm2_reg_offset * i_dpt_acc);
                }
#else
#pragma unroll
                for(int i = 0; i < q_gemm0_reg_num; i++)
                {
                    q_reg_gemm0[i] = {static_cast<bf16_t>(i + i_dpt_acc * q_gemm0_reg_num + 1)};
                }
#endif

#if 1
#pragma unroll
                for(int i = 0; i < kGemm0Gemm2KLoops; i++)
                {
                    Gemm0Gemm2Gemm4MfmaInstr::mfma_run(q_reg_gemm0[i].xy[0], vt_reg_gemm2[i].xy[0], dpt_acc[i_dpt_acc]);
                    Gemm0Gemm2Gemm4MfmaInstr::mfma_run(q_reg_gemm0[i].xy[1], vt_reg_gemm2[i].xy[1], dpt_acc[i_dpt_acc]);
                }
#endif
            }

#if 1
            // ds
#pragma unroll
            for(int i_dpt = 0; i_dpt < st_acc_num; i_dpt++)
            {
#pragma unroll
                for(int i_dpt_vec = 0; i_dpt_vec < kGemm0Gemm2Gemm4AccNum; i_dpt_vec += 4)
                {
                    lse_d = *reinterpret_cast<floatx4*>(d_smem + lse_d_lds_read_offset);
                    d_smem += lse_d_reg_offset;

                    dpt_acc[i_dpt][0 + i_dpt_vec] = st_acc[i_dpt][0 + i_dpt_vec] * (dpt_acc[i_dpt][0 + i_dpt_vec] - lse_d[0]);
#if !REMOVE_DS
                    dpt_acc[i_dpt][1 + i_dpt_vec] = st_acc[i_dpt][1 + i_dpt_vec] * (dpt_acc[i_dpt][1 + i_dpt_vec] - lse_d[1]);
                    dpt_acc[i_dpt][2 + i_dpt_vec] = st_acc[i_dpt][2 + i_dpt_vec] * (dpt_acc[i_dpt][2 + i_dpt_vec] - lse_d[2]);
                    dpt_acc[i_dpt][3 + i_dpt_vec] = st_acc[i_dpt][3 + i_dpt_vec] * (dpt_acc[i_dpt][3 + i_dpt_vec] - lse_d[3]);
#endif
                }    
            }
#endif

            // gemm 3
#pragma unroll
            for(int i_st_acc_reg_k = 0; i_st_acc_reg_k < st_acc_num; i_st_acc_reg_k++)
            {
#pragma unroll
                for(int i_st_acc = 0; i_st_acc < gemm1_gemm3_k_inner_loop; i_st_acc++)
                {
#if !REMOVE_GEMM3_LDS_READ 
                    do_reg_gemm1_tmp[0] = *reinterpret_cast<q_gemm3_do_gemm1_vec_type*>(q_smem + q_gemm3_do_gemm1_offset);
                    do_reg_gemm1_tmp[1] = *reinterpret_cast<q_gemm3_do_gemm1_vec_type*>(q_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset);
                    do_reg_gemm1_tmp[2] = *reinterpret_cast<q_gemm3_do_gemm1_vec_type*>(q_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset * 2);
                    do_reg_gemm1_tmp[3] = *reinterpret_cast<q_gemm3_do_gemm1_vec_type*>(q_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset * 3);
#else
                    do_reg_gemm1_tmp[0] = {(i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1), (i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1)};
                    do_reg_gemm1_tmp[1] = {(i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 2), (i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1)};
                    do_reg_gemm1_tmp[2] = {(i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 3), (i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1)};
                    do_reg_gemm1_tmp[3] = {(i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 4), (i_st_acc_reg_k * gemm1_gemm3_k_inner_loop + i_st_acc + 1)};
#endif

                    pt_reg_gemm1[0] = type_convert<bf16_t, float>(dpt_acc[i_st_acc_reg_k][0 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[1] = type_convert<bf16_t, float>(dpt_acc[i_st_acc_reg_k][1 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[2] = type_convert<bf16_t, float>(dpt_acc[i_st_acc_reg_k][2 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[3] = type_convert<bf16_t, float>(dpt_acc[i_st_acc_reg_k][3 + st_acc_gemmk_offset * i_st_acc]);
                    
                    if constexpr(dk_acc_num == 2)
                    {
                        do_reg_transpose_gemm1[0].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1]), bit_cast<uint32_t>(do_reg_gemm1_tmp[0]), m0);
                        do_reg_transpose_gemm1[0].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3]), bit_cast<uint32_t>(do_reg_gemm1_tmp[2]), m0);
                        do_reg_transpose_gemm1[1].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1]), bit_cast<uint32_t>(do_reg_gemm1_tmp[0]), m1);
                        do_reg_transpose_gemm1[1].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3]), bit_cast<uint32_t>(do_reg_gemm1_tmp[2]), m1);

                        do_reg_gemm1[0] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[0]);
                        do_reg_gemm1[1] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[1]);

                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[0], dk_acc[0]);
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[1], dk_acc[1]);
                    }
                    else if constexpr(dk_acc_num == 4)
                    {
                        do_reg_transpose_gemm1[0].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1].x), bit_cast<uint32_t>(do_reg_gemm1_tmp[0].x), m0);
                        do_reg_transpose_gemm1[0].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3].x), bit_cast<uint32_t>(do_reg_gemm1_tmp[2].x), m0);
                        do_reg_transpose_gemm1[1].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1].x), bit_cast<uint32_t>(do_reg_gemm1_tmp[0].x), m1);
                        do_reg_transpose_gemm1[1].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3].x), bit_cast<uint32_t>(do_reg_gemm1_tmp[2].x), m1);

                        do_reg_gemm1[0] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[0]);
                        do_reg_gemm1[1] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[1]);

                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[0], dk_acc[0]);
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[1], dk_acc[1]);

                        do_reg_transpose_gemm1[0].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1].y), bit_cast<uint32_t>(do_reg_gemm1_tmp[0].y), m0);
                        do_reg_transpose_gemm1[0].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3].y), bit_cast<uint32_t>(do_reg_gemm1_tmp[2].y), m0);
                        do_reg_transpose_gemm1[1].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1].y), bit_cast<uint32_t>(do_reg_gemm1_tmp[0].y), m1);
                        do_reg_transpose_gemm1[1].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3].y), bit_cast<uint32_t>(do_reg_gemm1_tmp[2].y), m1);

                        do_reg_gemm1[0] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[0]);
                        do_reg_gemm1[1] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[1]);

                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[0], dk_acc[2]);
                        Gemm1Gemm3MfmaInstr::mfma_run(pt_reg_gemm1, do_reg_gemm1[1], dk_acc[3]);
                    }

#if !REMOVE_DS_LDS_WRITE
                    *reinterpret_cast<bf16_t*>(ds_smem + ds_lds_write_offset + ds_lds_gemm_m_group_offset * i_st_acc + ds_lds_gemm_m_acc_reg_offset * i_st_acc_reg_k) = pt_reg_gemm1[0];
                    *reinterpret_cast<bf16_t*>(ds_smem + ds_lds_write_offset + ds_lds_gemm_m_group_offset * i_st_acc + ds_lds_gemm_m_acc_reg_offset * i_st_acc_reg_k + ds_lds_write_reg_offset) = pt_reg_gemm1[1];
                    *reinterpret_cast<bf16_t*>(ds_smem + ds_lds_write_offset + ds_lds_gemm_m_group_offset * i_st_acc + ds_lds_gemm_m_acc_reg_offset * i_st_acc_reg_k + ds_lds_write_reg_offset * 2) = pt_reg_gemm1[2];
                    *reinterpret_cast<bf16_t*>(ds_smem + ds_lds_write_offset + ds_lds_gemm_m_group_offset * i_st_acc + ds_lds_gemm_m_acc_reg_offset * i_st_acc_reg_k + ds_lds_write_reg_offset * 3) = pt_reg_gemm1[3];
#endif

                    q_smem += q_gemm3_do_gemm1_gemmk_offset;
                }
            }
#if 0
            printf("thread=%d, st_acc_num=%d, gemm1_gemm3_k_inner_loop=%d, ds_lds_write_offset=%d, ds_lds_gemm_m_group_offset=%d, ds_lds_gemm_m_acc_reg_offset=%d, ds_lds_write_reg_offset=%d\n",
                type_convert<int>(threadIdx.x),
                st_acc_num,
                gemm1_gemm3_k_inner_loop,
                ds_lds_write_offset,
                ds_lds_gemm_m_group_offset,
                ds_lds_gemm_m_acc_reg_offset,
                ds_lds_write_reg_offset);
#endif
            // gemm 4
#pragma unroll
            for(int i = 0; i < dq_acc_num; i++)
            {
                st_acc[st_acc_num - dq_acc_num + i] = {0};
            }

            __syncthreads();

#pragma unroll
            for(int i_dq_acc = 0; i_dq_acc < dq_acc_num; i_dq_acc++)
            //for(int i_dq_acc = 0; i_dq_acc < 1; i_dq_acc++)
            {
#pragma unroll
                for(int i_gemm4_k = 0; i_gemm4_k < kN0; i_gemm4_k += kGemm0Gemm2Gemm4WarpK)
                {
#if !REMOVE_GEMM4_LDS_READ
                    dp_reg_gemm4[0] = *reinterpret_cast<bfloat16x4*>(ds_smem + i_dq_acc * dq_acc_reg_offset + ds_gemm4_kiter_offset * (i_gemm4_k / kGemm0Gemm2Gemm4WarpKInst) + ds_gemm4_offset);
                    // ds_smem += ds_gemm4_kiter_offset;
                    dp_reg_gemm4[1] = *reinterpret_cast<bfloat16x4*>(ds_smem + i_dq_acc * dq_acc_reg_offset + ds_gemm4_kiter_offset * (i_gemm4_k / kGemm0Gemm2Gemm4WarpKInst + 1) + ds_gemm4_offset);
                    // ds_smem += ds_gemm4_kiter_offset;
#else
                    dp_reg_gemm4[0] = {static_cast<bf16_t>(i_dq_acc * kN0 + 1 + i_gemm4_k)};
                    ds_smem += ds_gemm4_kiter_offset + ds_gemm4_offset;
                    dp_reg_gemm4[1] = {static_cast<bf16_t>(i_dq_acc * kN0 + 1 + i_gemm4_k)};
                    ds_smem += ds_gemm4_kiter_offset;

#endif
                    Gemm0Gemm2Gemm4MfmaInstr::mfma_run(dp_reg_gemm4[0], bit_cast<bfloat16x4>(k_reg_to_gemm4[i_gemm4_k / kGemm0Gemm2Gemm4WarpKInst]), st_acc[st_acc_num - dq_acc_num + i_dq_acc]);
                    Gemm0Gemm2Gemm4MfmaInstr::mfma_run(dp_reg_gemm4[1], bit_cast<bfloat16x4>(k_reg_to_gemm4[i_gemm4_k / kGemm0Gemm2Gemm4WarpKInst + 1]), st_acc[st_acc_num - dq_acc_num + i_dq_acc]);
                }
            }

#if 0
            printf("thread=%d, dq_acc[2,3]=[%f,%f,%f,%f,%f,%f,%f,%f], dp_reg_gemm4=[%f,%f,%f,%f], dq_acc_reg_offset=%d\n",
                type_convert<int>(threadIdx.x),
                st_acc[2][0],
                st_acc[2][1],
                st_acc[2][2],
                st_acc[2][3],
                st_acc[3][0],
                st_acc[3][1],
                st_acc[3][2],
                st_acc[3][3],
                type_convert<float>(dp_reg_gemm4[1][0]),
                type_convert<float>(dp_reg_gemm4[1][1]),
                type_convert<float>(dp_reg_gemm4[1][2]),
                type_convert<float>(dp_reg_gemm4[1][3]),
                dq_acc_reg_offset
                );
#endif

#pragma unroll
            for(int i_dq_acc = 0; i_dq_acc < dq_acc_num; i_dq_acc++)
            {
#pragma unroll
                for(int i_dq_vec = 0; i_dq_vec < (kGemm0Gemm2Gemm4AccNum / 4); i_dq_vec++)
                {
#if REMOVE_ATOMICADD
                    *(dq_acc_ptr_tmp + dq_acc_offset) = st_acc[st_acc_num - dq_acc_num + i_dq_acc][i_dq_vec * 4 + 0] * kargs.raw_scale;
                    dq_acc_ptr_tmp += kargs.stride_q;
                    *(dq_acc_ptr_tmp + dq_acc_offset) = st_acc[st_acc_num - dq_acc_num + i_dq_acc][i_dq_vec * 4 + 1] * kargs.raw_scale;
                    dq_acc_ptr_tmp += kargs.stride_q;
                    *(dq_acc_ptr_tmp + dq_acc_offset) = st_acc[st_acc_num - dq_acc_num + i_dq_acc][i_dq_vec * 4 + 2] * kargs.raw_scale;
                    dq_acc_ptr_tmp += kargs.stride_q;
                    *(dq_acc_ptr_tmp + dq_acc_offset) = st_acc[st_acc_num - dq_acc_num + i_dq_acc][i_dq_vec * 4 + 3] * kargs.raw_scale;
                    dq_acc_ptr_tmp += (kGemm4GroupM - 3) * kargs.stride_q;
#else
                    llvm_amdgcn_raw_buffer_atomic_add_fp32(st_acc[st_acc_num - dq_acc_num + i_dq_acc][i_dq_vec * 4 + 0] * kargs.raw_scale,
                                                           dq_acc_resource,
                                                           dq_acc_offset,
                                                           dq_acc_wave_offset,
                                                           0);
                    dq_acc_wave_offset += stride_dq_acc_in_bytes;
                    llvm_amdgcn_raw_buffer_atomic_add_fp32(st_acc[st_acc_num - dq_acc_num + i_dq_acc][i_dq_vec * 4 + 1] * kargs.raw_scale,
                                                           dq_acc_resource,
                                                           dq_acc_offset,
                                                           dq_acc_wave_offset,
                                                           0);
                    dq_acc_wave_offset += stride_dq_acc_in_bytes;
                    llvm_amdgcn_raw_buffer_atomic_add_fp32(st_acc[st_acc_num - dq_acc_num + i_dq_acc][i_dq_vec * 4 + 2] * kargs.raw_scale,
                                                           dq_acc_resource,
                                                           dq_acc_offset,
                                                           dq_acc_wave_offset,
                                                           0);
                    dq_acc_wave_offset += stride_dq_acc_in_bytes;
                    llvm_amdgcn_raw_buffer_atomic_add_fp32(st_acc[st_acc_num - dq_acc_num + i_dq_acc][i_dq_vec * 4 + 3] * kargs.raw_scale,
                                                           dq_acc_resource,
                                                           dq_acc_offset,
                                                           dq_acc_wave_offset,
                                                           0);
                    dq_acc_wave_offset += (kGemm4GroupM - 3) * stride_dq_acc_in_bytes;
#endif
                }
                dq_acc_wave_offset += kGemm0Gemm2Gemm4WarpM * (kGemm4rm - 1) * stride_dq_acc_in_bytes;
            }

            __syncthreads();

            if(i_total_loops < (num_total_loop - 1))
            {
                lse_smem = smem_ptr;
                d_smem = lse_smem + kM0 * sizeof(LSEDataType);
                q_smem = d_smem + kM0 * sizeof(DDataType);
                do_smem = q_smem + (kM0 * sizeof(KDataType) + q_do_padding) * kQKHeaddim;
                ds_smem = do_smem;
#if !REMOVE_Q_DO_LDS_WRITE
#pragma unroll
                for(int i = 0; i < q_do_global_num; i++)
                {
                    *reinterpret_cast<float4*>(q_smem + kvqdo_smem_offset + kv_smem_reg_offset * i) = q_reg[i];
                    *reinterpret_cast<float4*>(do_smem + kvqdo_smem_offset + kv_smem_reg_offset * i) = do_reg[i];
                }
#endif
                if (lse_d_exec_mask)
                {
                    *reinterpret_cast<float*>(lse_smem + lse_d_lds_write_offset) = log2e_v<LSEDataType> * lse_reg;
                    *reinterpret_cast<float*>(d_smem + lse_d_lds_write_offset) = d_reg;
                }
            }

            i_total_loops += 1;
            // seqlen_q_step += kM0;
            
        } while(i_total_loops < (num_total_loop - 0));

        // write out dv
        constexpr int dv_dk_acc_vec_size = kVHeaddim / (kGemm1Gemm3rn * kGemm1Gemm3WarpN);
        const int& stride_v_seq = kargs.stride_v;
        int dv_hbm_offset = (n_id * dv_dk_acc_vec_size + k0_id * stride_v_seq * 4) * sizeof(VDataType);
        const int dv_cta_offset = i_n0 * stride_v_seq * sizeof(VDataType);
        const int dk_cta_offset = i_n0 * kargs.stride_k * sizeof(KDataType);

        // const int dv_hbm_reg_offset = stride_v_seq;
        // const int dv_hbm_a_group_offset = stride_v_seq * 8;
        int wave_offset_gemm1_gemm3 = (wave_id * kGemm1Gemm3WarpM * stride_v_seq * sizeof(VDataType));
        dv_hbm_offset += wave_offset_gemm1_gemm3;
        // dv_hbm_offset += wave_offset_gemm1_gemm3;
        const int reg_offset_gemm1_gemm3 = stride_v_seq * sizeof(VDataType);
        const int group_offset_gemm1_gemm3 = stride_v_seq * (kGemm1Gemm3WarpM / kGemm1Gemm3AccNum) * 4 * sizeof(VDataType);

#pragma unroll
        for (int i_dv = 0; i_dv < (kGemm1Gemm3AccNum / 4); i_dv++)
        {
            if constexpr(dv_acc_num == 2)
            {
                uint32_t dv_pack;
                dv_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4]), m1);
                llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(dv_pack),
                                                  dv_resource,
                                                  dv_hbm_offset,
                                                  dv_cta_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 0,
                                                  0);
                dv_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 1]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 1]), m1);
                llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(dv_pack),
                                                  dv_resource,
                                                  dv_hbm_offset,
                                                  dv_cta_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 1,
                                                  0);
                dv_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 2]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 2]), m1);
                llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(dv_pack),
                                                  dv_resource,
                                                  dv_hbm_offset,
                                                  dv_cta_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 2,
                                                  0);
                dv_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 3]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 3]), m1);
                llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(dv_pack),
                                                  dv_resource,
                                                  dv_hbm_offset,
                                                  dv_cta_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 3,
                                                  0);
            }
            else if constexpr(dv_acc_num == 4)
            {
                uint2 dv_pack;
                dv_pack.x = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4]), m1);
                dv_pack.y = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[3][i_dv * 4]), bit_cast<uint32_t>(dv_acc[2][i_dv * 4]), m1);
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dv_pack),
                                                    dv_resource,
                                                    dv_hbm_offset,
                                                    dv_cta_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 0,
                                                    0);
                dv_pack.x = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 1]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 1]), m1);
                dv_pack.y = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[3][i_dv * 4 + 1]), bit_cast<uint32_t>(dv_acc[2][i_dv * 4 + 1]), m1);
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dv_pack),
                                                    dv_resource,
                                                    dv_hbm_offset,
                                                    dv_cta_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 1,
                                                    0);
                dv_pack.x = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 2]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 2]), m1);
                dv_pack.y = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[3][i_dv * 4 + 2]), bit_cast<uint32_t>(dv_acc[2][i_dv * 4 + 2]), m1);
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dv_pack),
                                                    dv_resource,
                                                    dv_hbm_offset,
                                                    dv_cta_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 2,
                                                    0);
                dv_pack.x = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 3]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 3]), m1);
                dv_pack.y = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[3][i_dv * 4 + 3]), bit_cast<uint32_t>(dv_acc[2][i_dv * 4 + 3]), m1);
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dv_pack),
                                                    dv_resource,
                                                    dv_hbm_offset,
                                                    dv_cta_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 3,
                                                    0);

            }
        }

        // dk = dk * scale
#pragma unroll
        for(int i = 0; i < dk_acc_num; i++)
        {
            dk_acc[i] *= kargs.raw_scale;
        }
        
#pragma unroll
        for (int i_dk = 0; i_dk < (kGemm1Gemm3AccNum / 4); i_dk++)
        {
            if constexpr(dk_acc_num == 2)
            {
                uint32_t dk_pack;
                dk_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4]), m1);
                llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(dk_pack),
                                                  dk_resource,
                                                  dv_hbm_offset,
                                                  dk_cta_offset + group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 0,
                                                  0);
                dk_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 1]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 1]), m1);
                llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(dk_pack),
                                                  dk_resource,
                                                  dv_hbm_offset,
                                                  dk_cta_offset + group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 1,
                                                  0);
                dk_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 2]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 2]), m1);
                llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(dk_pack),
                                                  dk_resource,
                                                  dv_hbm_offset,
                                                  dk_cta_offset + group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 2,
                                                  0);
                dk_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 3]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 3]), m1);
                llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(dk_pack),
                                                  dk_resource,
                                                  dv_hbm_offset,
                                                  dk_cta_offset + group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 3,
                                                  0);
            }
            else if constexpr(dk_acc_num == 4)
            {
                uint2 dk_pack;
                dk_pack.x = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4]), m1);
                dk_pack.y = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[3][i_dk * 4]), bit_cast<uint32_t>(dk_acc[2][i_dk * 4]), m1);
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dk_pack),
                                                    dk_resource,
                                                    dv_hbm_offset,
                                                    dk_cta_offset + group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 0,
                                                    0);
                dk_pack.x = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 1]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 1]), m1);
                dk_pack.y = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[3][i_dk * 4 + 1]), bit_cast<uint32_t>(dk_acc[2][i_dk * 4 + 1]), m1);
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dk_pack),
                                                    dk_resource,
                                                    dv_hbm_offset,
                                                    dk_cta_offset +group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 1,
                                                    0);
                dk_pack.x = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 2]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 2]), m1);
                dk_pack.y = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[3][i_dk * 4 + 2]), bit_cast<uint32_t>(dk_acc[2][i_dk * 4 + 2]), m1);
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dk_pack),
                                                    dk_resource,
                                                    dv_hbm_offset,
                                                    dk_cta_offset +group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 2,
                                                    0);
                dk_pack.x = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 3]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 3]), m1);
                dk_pack.y = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[3][i_dk * 4 + 3]), bit_cast<uint32_t>(dk_acc[2][i_dk * 4 + 3]), m1);
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dk_pack),
                                                    dk_resource,
                                                    dv_hbm_offset,
                                                    dk_cta_offset +group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 3,
                                                    0);
            }
        }
        

        return;

        // Q/K/V/LSE/D/dO/dQ/dK/dV DRAM and DRAM window
        const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            q_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_q, 1),
            number<FmhaPipeline::kAlignmentQ>{},
            number<1>{});
        const auto q_dram = pad_tensor_view(
            q_dram_naive,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            sequence<kPadSeqLenQ, kPadHeadDimQ>{});

        const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            k_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_q),
            make_tuple(kargs.stride_k, 1),
            number<FmhaPipeline::kAlignmentK>{},
            number<1>{});
        const auto k_dram = pad_tensor_view(
            k_dram_naive,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            sequence<kPadSeqLenK, kPadHeadDimQ>{});

        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<FmhaPipeline::kAlignmentV>{},
                number<1>{});
            return pad_tensor_view(
                v_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
                sequence<kPadSeqLenK, kPadHeadDimV>{});
        }();

        const auto lse_dram = [&]() {
            const auto lse_dram_naive = make_naive_tensor_view_packed<address_space_enum::global>(
                lse_ptr, make_tuple(kargs.seqlen_q), number<1>{});
            return pad_tensor_view(
                lse_dram_naive, make_tuple(number<FmhaPipeline::kM0>{}), sequence<kPadSeqLenQ>{});
        }();

        const auto d_dram = [&]() {
            const auto d_dram_naive = make_naive_tensor_view_packed<address_space_enum::global>(
                d_ptr, make_tuple(kargs.seqlen_q), number<1>{});
            return pad_tensor_view(
                d_dram_naive, make_tuple(number<FmhaPipeline::kM0>{}), sequence<kPadSeqLenQ>{});
        }();

        const auto do_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            do_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_do, 1),
            number<FmhaPipeline::kAlignmentOGrad>{},
            number<1>{});
        const auto do_dram = pad_tensor_view(
            do_dram_naive,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kVHeaddim>{}),
            sequence<kPadSeqLenQ, kPadHeadDimV>{});

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {i_n0, 0});

        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {i_n0, 0});

        auto do_dram_window = make_tile_window(
            do_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {0, 0});

        auto dq_dram_window = [&, i_tile_n_ = i_tile_n, i_nhead_ = i_nhead]() {
            if constexpr(kIsDeterministic)
            {
                AccDataType* dq_acc_ptr =
                    reinterpret_cast<AccDataType*>(kargs.dq_acc_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_q +
                    static_cast<long_index_t>(i_tile_n_) * kargs.split_stride_dq_acc +
                    batch_offset_q;

                auto dq_acc_dram = [&]() {
                    const auto dq_acc_dram_naive =
                        make_naive_tensor_view<address_space_enum::global>(
                            dq_acc_ptr,
                            make_tuple(kargs.seqlen_q, kargs.hdim_q),
                            make_tuple(kargs.stride_q, 1),
                            number<FmhaPipeline::kAlignmentQGrad>{},
                            number<1>{});

                    return pad_tensor_view(
                        dq_acc_dram_naive,
                        make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                        sequence<kPadSeqLenQ, kPadHeadDimQ>{});
                }();

                return make_tile_window(
                    dq_acc_dram,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                    {0, 0});
            }
            else
            {
                AccDataType* dq_acc_ptr =
                    reinterpret_cast<AccDataType*>(kargs.dq_acc_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_q + batch_offset_q;

                auto dq_acc_dram = [&]() {
                    const auto dq_acc_dram_naive =
                        make_naive_tensor_view<address_space_enum::global,
                                               memory_operation_enum::atomic_add>(
                            dq_acc_ptr,
                            make_tuple(kargs.seqlen_q, kargs.hdim_q),
                            make_tuple(kargs.stride_q, 1),
                            number<FmhaPipeline::kAlignmentQGrad>{},
                            number<1>{});

                    return pad_tensor_view(
                        dq_acc_dram_naive,
                        make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                        sequence<kPadSeqLenQ, kPadHeadDimQ>{});
                }();

                return make_tile_window(
                    dq_acc_dram,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                    {0, 0});
            }
        }();

        auto lse_dram_window =
            make_tile_window(lse_dram, make_tuple(number<FmhaPipeline::kM0>{}), {0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(number<FmhaPipeline::kM0>{}), {0});

        /// FIXME: Before C++20, capturing structured binding variables are not supported. Remove
        /// following copy capture of the 'i_nhead' if in C++20
        constexpr auto bias_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                    batch_offset_bias;

                const auto bias_dram = [&]() {
                    const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        bias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_bias, 1),
                        number<FmhaPipeline::kAlignmentBias>{},
                        number<1>{});

                    return pad_tensor_view(bias_dram_naive,
                                           bias_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        auto dbias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(kHasBiasGrad)
            {
                BiasGradDataType* dbias_ptr =
                    reinterpret_cast<BiasGradDataType*>(kargs.dbias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dbias +
                    batch_offset_dbias;

                auto dbias_dram = [&]() {
                    const auto dbias_dram_naive =
                        make_naive_tensor_view<address_space_enum::global>(
                            dbias_ptr,
                            make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                            make_tuple(kargs.stride_dbias, 1),
                            number<FmhaPipeline::kAlignmentBias>{},
                            number<1>{});

                    return pad_tensor_view(dbias_dram_naive,
                                           bias_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(dbias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        // WA i_batch capture structure binding before c++20
        auto position_encoding = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                // data loading, shared by entire wg
                // TODO: how to use s_read?
                AccDataType slope = *(reinterpret_cast<const AccDataType*>(kargs.alibi_slope_ptr) +
                                      i_batch_ * kargs.alibi_slope_stride + i_nhead_);
                slope *= ck_tile::log2e_v<>;
                if constexpr(kHasMask)
                {
                    return make_alibi_from_lr_mask<AccDataType, false>(slope,
                                                                       kargs.window_size_left,
                                                                       kargs.window_size_right,
                                                                       kargs.seqlen_q,
                                                                       kargs.seqlen_k,
                                                                       kargs.mask_type);
                }
                else
                {
                    return Alibi<AccDataType, false>{
                        slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
                }
            }
            else
            {
                return EmptyPositionEncoding<AccDataType>{};
            }
        }();

        // dropout
        float rp_undrop             = 1;
        float scale_rp_undrop       = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
        uint64_t drop_seed          = 0;
        uint64_t drop_offset        = 0;

        if constexpr(kHasDropout)
        {
            rp_undrop           = kargs.rp_undrop;
            scale_rp_undrop     = kargs.scale_rp_undrop;
            p_undrop_in_uint8_t = kargs.p_undrop_in_uint8_t;
            drop_seed           = kargs.drop_seed;
            drop_offset         = kargs.drop_offset;
        }
        FmhaDropout dropout(i_batch,
                            i_nhead,
                            kargs.num_head_q,
                            drop_seed,
                            drop_offset,
                            rp_undrop,
                            p_undrop_in_uint8_t);

        auto randval_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto randval_dram_window_lengths =
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
            if constexpr(kIsStoreRandval)
            {
                RandValOutputDataType* rand_val_ptr =
                    reinterpret_cast<RandValOutputDataType*>(kargs.rand_val_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_randval +
                    batch_offset_randval;

                const auto randval_dram = [&]() {
                    const auto randval_dram_naive =
                        make_naive_tensor_view<address_space_enum::global>(
                            rand_val_ptr,
                            make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                            make_tuple(kargs.stride_randval, 1),
                            number<1>{},
                            number<1>{});

                    return pad_tensor_view(randval_dram_naive,
                                           randval_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(randval_dram, randval_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(randval_dram_window_lengths);
            }
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        auto [dk_acc_tile, dv_acc_tile] = FmhaPipeline{}(q_dram_window,
                                                         k_dram_window,
                                                         v_dram_window,
                                                         bias_dram_window,
                                                         randval_dram_window,
                                                         do_dram_window,
                                                         lse_dram_window,
                                                         d_dram_window,
                                                         dq_dram_window,
                                                         dbias_dram_window,
                                                         mask,
                                                         position_encoding,
                                                         kargs.raw_scale,
                                                         kargs.scale,
                                                         rp_undrop,
                                                         scale_rp_undrop,
                                                         smem_ptr,
                                                         dropout);

        auto dk_dram = [&]() {
            const auto dk_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                dk_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_dk, 1),
                number<FmhaPipeline::kAlignmentKGrad>{},
                number<1>{});

            return pad_tensor_view(
                dk_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();

        auto dv_dram = [&]() {
            const auto dv_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                dv_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_dv, 1),
                number<FmhaPipeline::kAlignmentVGrad>{},
                number<1>{});

            return pad_tensor_view(
                dv_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
                sequence<kPadSeqLenK, kPadHeadDimV>{});
        }();

        auto dk_dram_window = make_tile_window(
            dk_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {i_n0, 0});

        auto dv_dram_window = make_tile_window(
            dv_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {i_n0, 0});

        KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile);
        VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile);
    }
};

template <typename TilePartitioner_, typename FmhaBwdOGradDotO_>
struct FmhaBwdOGradDotOKernel
{
    using TilePartitioner                         = ck_tile::remove_cvref_t<TilePartitioner_>;
    using FmhaBwdOGradDotO                        = ck_tile::remove_cvref_t<FmhaBwdOGradDotO_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaBwdOGradDotO::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaBwdOGradDotO::kBlockPerCu;
    static constexpr ck_tile::index_t kM0         = kBlockSize;
    static constexpr ck_tile::index_t kVHeaddim   = FmhaBwdOGradDotO::kVHeaddim;

    using DDataType     = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::DDataType>;
    using ODataType     = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::ODataType>;
    using OGradDataType = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::OGradDataType>;

    static constexpr bool kIsGroupMode = FmhaBwdOGradDotO::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaBwdOGradDotO::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV = FmhaBwdOGradDotO::kPadHeadDimV;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_bwd_dot_do_o_d") + _TS_(kVHeaddim) + "_" + _SS_(t2s<ODataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_" +
            ("o" + _TS_(kBlockPerCu)) + (pn.empty() ? "" : "_" + pn);
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdOGradDotOCommonKargs
    {
        const void* o_ptr;
        const void* do_ptr;
        void* d_ptr;

        float p_undrop;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t stride_do;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_do;
        ck_tile::index_t nhead_stride_o;
        ck_tile::index_t nhead_stride_d;
        ck_tile::index_t batch_stride_d;
    };

    struct FmhaBwdOGradDotOBatchModeKargs : FmhaBwdOGradDotOCommonKargs
    {
        ck_tile::index_t batch_stride_do;
        ck_tile::index_t batch_stride_o;
    };

    struct FmhaBwdOGradDotOGroupModeKargs : FmhaBwdOGradDotOCommonKargs
    {
        const int32_t* seqstart_q_ptr;
    };

    using Kargs = std::
        conditional_t<kIsGroupMode, FmhaBwdOGradDotOGroupModeKargs, FmhaBwdOGradDotOBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* o_ptr,
              const void* do_ptr,
              void* d_ptr,
              float p_undrop,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t stride_do,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_do,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_d,
              ck_tile::index_t batch_stride_do,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t batch_stride_d)
    {
        Kargs kargs{{o_ptr,
                     do_ptr,
                     d_ptr,
                     p_undrop,
                     seqlen_q,
                     hdim_v,
                     stride_do,
                     stride_o,
                     nhead_stride_do,
                     nhead_stride_o,
                     nhead_stride_d,
                     batch_stride_d},
                    batch_stride_do,
                    batch_stride_o};

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* o_ptr,
              const void* do_ptr,
              void* d_ptr,
              float p_undrop,
              const void* seqstart_q_ptr,
              ck_tile::index_t hdim_v,
              ck_tile::index_t stride_do,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_do,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_d,
              ck_tile::index_t batch_stride_d)
    {
        Kargs kargs{{o_ptr,
                     do_ptr,
                     d_ptr,
                     p_undrop,
                     -1, // seqlen will be updated by another pointer
                     hdim_v,
                     stride_do,
                     stride_o,
                     nhead_stride_do,
                     nhead_stride_o,
                     nhead_stride_d,
                     batch_stride_d},
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr)};

        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_q_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_q);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kM0);

        long_index_t batch_offset_o  = 0;
        long_index_t batch_offset_do = 0;
        long_index_t batch_offset_d  = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_o  = query_start * kargs.stride_o;
            batch_offset_do = query_start * kargs.stride_do;
            batch_offset_d  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_d;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }
        }
        else
        {
            batch_offset_o  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
            batch_offset_do = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_d  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_d;
        }

        // for simplicity, batch stride we just modify the pointer
        const ODataType* o_ptr = reinterpret_cast<const ODataType*>(kargs.o_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                                 batch_offset_o;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        DDataType* d_ptr = reinterpret_cast<DDataType*>(kargs.d_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_d +
                           batch_offset_d;

        // O/dO/D DRAM and DRAM window
        const auto o_dram = [&]() {
            auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<FmhaBwdOGradDotO::kAlignmentO>{},
                number<1>{});
            return pad_tensor_view(o_dram_naive,
                                   make_tuple(number<kM0>{}, number<kVHeaddim>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();
        const auto do_dram = [&]() {
            auto do_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                do_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_do, 1),
                number<FmhaBwdOGradDotO::kAlignmentOGrad>{},
                number<1>{});
            return pad_tensor_view(do_dram_naive,
                                   make_tuple(number<kM0>{}, number<kVHeaddim>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();
        auto d_dram = [&]() {
            const auto d_dram_naive = make_naive_tensor_view_packed<address_space_enum::global>(
                d_ptr, make_tuple(kargs.seqlen_q), number<1>{});
            return pad_tensor_view(
                d_dram_naive, make_tuple(number<kM0>{}), sequence<kPadSeqLenQ>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {i_m0, 0});

        auto do_dram_window =
            make_tile_window(do_dram, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {i_m0, 0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(number<kM0>{}), {i_m0});

        FmhaBwdOGradDotO{}(o_dram_window, do_dram_window, d_dram_window, kargs.p_undrop);
    }
};

template <typename TilePartitioner_, typename FmhaBwdConvertQGrad_>
struct FmhaBwdConvertQGradKernel
{
    using TilePartitioner                         = ck_tile::remove_cvref_t<TilePartitioner_>;
    using FmhaBwdConvertQGrad                     = ck_tile::remove_cvref_t<FmhaBwdConvertQGrad_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaBwdConvertQGrad::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaBwdConvertQGrad::kBlockPerCu;
    static constexpr ck_tile::index_t kM0         = FmhaBwdConvertQGrad::kM0;
    static constexpr ck_tile::index_t kN0         = FmhaBwdConvertQGrad::kN0;
    static constexpr ck_tile::index_t kQKHeaddim  = FmhaBwdConvertQGrad::kQKHeaddim;

    using AccDataType   = ck_tile::remove_cvref_t<typename FmhaBwdConvertQGrad::AccDataType>;
    using QGradDataType = ck_tile::remove_cvref_t<typename FmhaBwdConvertQGrad::QGradDataType>;

    static constexpr bool kIsGroupMode     = FmhaBwdConvertQGrad::kIsGroupMode;
    static constexpr bool kPadSeqLenQ      = FmhaBwdConvertQGrad::kPadSeqLenQ;
    static constexpr bool kPadHeadDimQ     = FmhaBwdConvertQGrad::kPadHeadDimQ;
    static constexpr bool kIsDeterministic = FmhaBwdConvertQGrad::kIsDeterministic;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadHeadDimQ) n += "d";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_bwd_convert_dq_d") + _TS_(kQKHeaddim) + "_" + _SS_(t2s<QGradDataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + (kIsDeterministic ? "_deterministic" : "") + "_" +
            ("o" + _TS_(kBlockPerCu)) + (pn.empty() ? "" : "_" + pn);
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    // to avoid duplicated base class prblem, introduce an template arg
    template <ck_tile::index_t I>
    struct FmhaBwdConvertQGradEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdConvertQGradCommonKargs
    {
        const void* dq_acc_ptr;
        void* dq_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;

        ck_tile::index_t stride_dq;
        ck_tile::index_t nhead_stride_dq;
    };

    struct FmhaBwdConvertQGradDeterministicKargs
    {
        ck_tile::index_t split_stride_dq_acc = 0;
    };

    struct FmhaBwdConvertQGradBatchModeKargs
        : FmhaBwdConvertQGradCommonKargs,
          std::conditional_t<kIsDeterministic,
                             FmhaBwdConvertQGradDeterministicKargs,
                             FmhaBwdConvertQGradEmptyKargs<0>>
    {
        ck_tile::index_t batch_stride_dq;
    };

    struct FmhaBwdConvertQGradGroupModeKargs
        : FmhaBwdConvertQGradCommonKargs,
          std::conditional_t<kIsDeterministic,
                             FmhaBwdConvertQGradDeterministicKargs,
                             FmhaBwdConvertQGradEmptyKargs<0>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode,
                                     FmhaBwdConvertQGradGroupModeKargs,
                                     FmhaBwdConvertQGradBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* dq_acc_ptr,
              void* dq_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t stride_dq,
              ck_tile::index_t nhead_stride_dq,
              ck_tile::index_t batch_stride_dq,
              ck_tile::index_t split_stride_dq_acc)
    {
        Kargs kargs{{dq_acc_ptr, dq_ptr, seqlen_q, seqlen_k, hdim_q, stride_dq, nhead_stride_dq},
                    {},
                    batch_stride_dq};

        if constexpr(kIsDeterministic)
        {
            kargs.split_stride_dq_acc = split_stride_dq_acc;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* dq_acc_ptr,
              void* dq_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t stride_dq,
              ck_tile::index_t nhead_stride_dq,
              ck_tile::index_t split_stride_dq_acc)
    {
        Kargs kargs{{dq_acc_ptr,
                     dq_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     stride_dq,
                     nhead_stride_dq},
                    {},
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr)};

        if constexpr(kIsDeterministic)
        {
            kargs.split_stride_dq_acc = split_stride_dq_acc;
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_q_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_q);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kM0);
        // const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * 1);

        long_index_t batch_offset_dq = 0;
        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            batch_offset_dq                = query_start * kargs.stride_dq;

            const int num_head_q = kargs.stride_dq / kargs.hdim_q;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
            const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
            kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            //if(kargs.seqlen_q <= i_m0)
            if(kargs.seqlen_q * num_head_q <= i_m0)
            {
                return;
            }
        }
        else
        {
            batch_offset_dq = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq;
        }

        // for simplicity, batch stride we just modify the pointer
        QGradDataType* dq_ptr = reinterpret_cast<QGradDataType*>(kargs.dq_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dq +
                                batch_offset_dq;

        // dQAcc/dQ DRAM and DRAM window
        const auto dq_acc_dram = [&, i_nhead_ = i_nhead]() {
            if constexpr(kIsDeterministic)
            {
                const AccDataType* dq_acc_ptr =
                    reinterpret_cast<const AccDataType*>(kargs.dq_acc_ptr) +
                    static_cast<long_index_t>(i_nhead_) * (kargs.nhead_stride_dq) + batch_offset_dq;

                const index_t nsplits = ck_tile::integer_divide_ceil(kargs.seqlen_k, kN0);

                auto dq_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    dq_acc_ptr,
                    make_tuple(nsplits, kargs.seqlen_q, kargs.hdim_q),
                    make_tuple(kargs.split_stride_dq_acc, kargs.stride_dq, 1),
                    number<FmhaBwdConvertQGrad::kAlignmentQGradAcc>{},
                    number<1>{});
                return pad_tensor_view(dq_acc_dram_naive,
                                       make_tuple(number<1>{}, number<kM0>{}, number<kQKHeaddim>{}),
                                       sequence<false, kPadSeqLenQ, kPadHeadDimQ>{});
            }
            else
            {
                const AccDataType* dq_acc_ptr =
                    reinterpret_cast<const AccDataType*>(kargs.dq_acc_ptr) +
                    static_cast<long_index_t>(i_nhead_) * (kargs.nhead_stride_dq) + batch_offset_dq;

                auto dq_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    dq_acc_ptr,
                    make_tuple(kargs.seqlen_q, kargs.hdim_q),
                    make_tuple(kargs.stride_dq, 1),
                    number<FmhaBwdConvertQGrad::kAlignmentQGradAcc>{},
                    number<1>{});
                return pad_tensor_view(dq_acc_dram_naive,
                                       make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                       sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();

        auto dq_dram = [&]() {
            auto dq_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                dq_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_dq, 1),
                number<FmhaBwdConvertQGrad::kAlignmentQGrad>{},
                number<1>{});
            return pad_tensor_view(dq_dram_naive,
                                   make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();

        auto dq_acc_dram_window = [&]() {
            if constexpr(kIsDeterministic)
            {
                return make_tile_window(
                    dq_acc_dram,
                    make_tuple(number<1>{}, number<kM0>{}, number<kQKHeaddim>{}),
                    {0, i_m0, 0});
            }
            else
            {
                return make_tile_window(
                    dq_acc_dram, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {i_m0, 0});
            }
        }();

        auto dq_dram_window =
            make_tile_window(dq_dram, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {i_m0, 0});

        if constexpr(kIsDeterministic)
        {
            const index_t nsplits = ck_tile::integer_divide_ceil(kargs.seqlen_k, kN0);
            FmhaBwdConvertQGrad{}(dq_acc_dram_window, dq_dram_window, nsplits);
        }
        else
        {

#if 1
            const AccDataType* dq_acc_ptr =
                reinterpret_cast<const AccDataType*>(kargs.dq_acc_ptr) +
                static_cast<long_index_t>(i_nhead) * (kargs.nhead_stride_dq) + batch_offset_dq;
            
            const int dq_acc_range_in_bytes = kargs.seqlen_q * kargs.stride_dq * sizeof(AccDataType);
            const int dq_range_in_bytes = kargs.seqlen_q * kargs.stride_dq * sizeof(QGradDataType);
            int32x4_t dq_acc_resource = make_wave_buffer_resource(dq_acc_ptr, dq_acc_range_in_bytes);
            int32x4_t dq_resource = make_wave_buffer_resource(dq_ptr, dq_range_in_bytes);

            constexpr int32_t m1 = 0x07060302;
            int dq_acc_thread_offset = threadIdx.x * sizeof(float4);
            int dq_thread_offset = threadIdx.x * sizeof(float2);
            int dq_acc_wave_offset = kQKHeaddim * i_m0 * sizeof(AccDataType);
            int dq_wave_offset = kQKHeaddim * i_m0 * sizeof(QGradDataType);

            constexpr int dq_reg_num = kM0 * kQKHeaddim * sizeof(AccDataType) / (kBlockSize * sizeof(float4));
            constexpr int dq_acc_reg_offset = (kBlockSize * sizeof(float4));
            constexpr int dq_reg_offset = (kBlockSize * sizeof(float2));
            float4 dq_acc_reg[dq_reg_num];
            float2 dq_reg[dq_reg_num];

#pragma unroll
            for(int i = 0; i < dq_reg_num; i++)
            {
                dq_acc_reg[i] = bit_cast<float4>(llvm_amdgcn_raw_buffer_load_fp32x4(dq_acc_resource, dq_acc_thread_offset, dq_acc_wave_offset, 0));
                dq_acc_wave_offset += dq_acc_reg_offset;
            }

#pragma unroll
            for(int i = 0; i < dq_reg_num; i++)
            {
                dq_reg[i].x = bit_cast<float>(__builtin_amdgcn_perm(bit_cast<uint32_t>(dq_acc_reg[i].y), bit_cast<uint32_t>(dq_acc_reg[i].x), m1));
                dq_reg[i].y = bit_cast<float>(__builtin_amdgcn_perm(bit_cast<uint32_t>(dq_acc_reg[i].w), bit_cast<uint32_t>(dq_acc_reg[i].z), m1));
            }

#pragma unroll
            for(int i = 0; i < dq_reg_num; i++)
            {
                llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(dq_reg[i]), dq_resource, dq_thread_offset, dq_wave_offset, 0);
                dq_wave_offset += dq_reg_offset;
            }
#if 0
            if(threadIdx.x == 0)
            {
                printf("hhhh\n");
            }
#endif

#else
            FmhaBwdConvertQGrad{}(dq_acc_dram_window, dq_dram_window);
#endif
        }
    }
};

} // namespace ck_tile
