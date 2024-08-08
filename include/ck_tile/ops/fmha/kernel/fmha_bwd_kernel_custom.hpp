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

namespace ck_tile {

#define GCN_MFMA_INSTR_32 __builtin_amdgcn_mfma_f32_32x32x8bf16_1k
#define GCN_MFMA_INSTR_16 __builtin_amdgcn_mfma_f32_16x16x16bf16_1k

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
    static constexpr ck_tile::index_t kGemm1Gemm3WarpM = Gemm1Gemm3WarpTile::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t kGemm1Gemm3WarpN = Gemm1Gemm3WarpTile::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t kGemm1Gemm3WarpK = Gemm1Gemm3WarpTile::at(ck_tile::number<2>{});
    
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
        using CVecType = ext_vector_t<float, 16>;

        // imm number
        constexpr int32_t m0 = 0x05040100;
        constexpr int32_t m1 = 0x07060302;

        // prepare k and v_t in smem
        constexpr int k_reg_num = kN0 * kQKHeaddim * sizeof(KDataType) / (kBlockSize * 16);
        constexpr int k_reg_row = kBlockSize * 16 / (kQKHeaddim * sizeof(KDataType));
        float4 k_reg[k_reg_num];
        int kv_load_offset = (threadIdx.x & 7) * 8 + (threadIdx.x >> 3) * kargs.stride_k;
        int kv_reg_offset = kargs.stride_k * k_reg_row;

#pragma unroll
        for(int i_k_reg = 0; i_k_reg < k_reg_num; i_k_reg++)
        {
            k_reg[i_k_reg] = *reinterpret_cast<const float4 *>(k_ptr + kv_load_offset + i_n0 * kargs.stride_k);
            k_ptr += kv_reg_offset;
        }

        constexpr int v_reg_num = kN0 * kVHeaddim * sizeof(VDataType) / (kBlockSize * 16);
        // constexpr int v_reg_row = kBlockSize * 16 / (kVHeaddim * sizeof(VDataType));
        float4 v_reg[v_reg_num];
        
#pragma unroll
        for(int i_v_reg = 0; i_v_reg < v_reg_num; i_v_reg++)
        {
            v_reg[i_v_reg] = *reinterpret_cast<const float4 *>(v_ptr + kv_load_offset + i_n0 * kargs.stride_k);
            v_ptr += kv_reg_offset;
        }

        char* k_smem = smem_ptr;
        int kvqdo_smem_offset = threadIdx.x * sizeof(float4);
        constexpr int q_do_padding = sizeof(float4);
        int kvqdo_smem_offset_padding = kvqdo_smem_offset / kN0 * q_do_padding;
        kvqdo_smem_offset += kvqdo_smem_offset_padding;
        constexpr int kv_smem_reg_offset = k_reg_row * (kQKHeaddim * sizeof(KDataType) + q_do_padding);

#pragma unroll
        for(int i_k_reg = 0; i_k_reg < k_reg_num; i_k_reg++)
        {
            *reinterpret_cast<float4*>(k_smem + kvqdo_smem_offset + kv_smem_reg_offset * i_k_reg) = k_reg[i_k_reg];
        }
       
        __syncthreads();

        constexpr int kt_reg_gemm0_vt_reg_gemm2_num =  
        _BF16x8_t kt_reg_to_gemm0[4];
        int wave_id = threadIdx.x / 64;
        int wave_lane_id = threadIdx.x % 64;
        int k0_id = wave_lane_id / 32;
        int n_id = wave_lane_id % 32;
        int n_wave_repeat_id = wave_id % 4;
        int k_smem_gemm0_offset = n_wave_repeat_id * 32 * (64 * 2 + q_do_padding) + n_id * (64 * 2 + q_do_padding) + k0_id * 16;
        constexpr int k_smem_read_reg_offset = 32;
        kt_reg_to_gemm0[0] = *reinterpret_cast<_BF16x8_t*>(k_smem + k_smem_gemm0_offset);
        kt_reg_to_gemm0[1] = *reinterpret_cast<_BF16x8_t*>(k_smem + k_smem_gemm0_offset + k_smem_read_reg_offset);
        kt_reg_to_gemm0[2] = *reinterpret_cast<_BF16x8_t*>(k_smem + k_smem_gemm0_offset + k_smem_read_reg_offset * 2);
        kt_reg_to_gemm0[3] = *reinterpret_cast<_BF16x8_t*>(k_smem + k_smem_gemm0_offset + k_smem_read_reg_offset * 3);

        float2 k_reg_to_gemm4[16];
        unsigned short k_gemm4[4];
        int gemm4_n_wave_id = wave_id / 2;
        int k_smem_gemm4_offset = n_id * 2 + k0_id * (64 * 2 + q_do_padding) * 4 + gemm4_n_wave_id * 32 * 2;
        constexpr int k_smem_gemm4_offset_k1 = 64 * 2 + q_do_padding;
        constexpr int k_smem_gemm4_offset_reg = (64 * 2 + q_do_padding) * 8;

#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            k_gemm4[0] = *reinterpret_cast<unsigned short*>(k_smem + k_smem_gemm4_offset);
            k_gemm4[1] = *reinterpret_cast<unsigned short*>(k_smem + k_smem_gemm4_offset + k_smem_gemm4_offset_k1);
            k_gemm4[2] = *reinterpret_cast<unsigned short*>(k_smem + k_smem_gemm4_offset + k_smem_gemm4_offset_k1 * 2);
            k_gemm4[3] = *reinterpret_cast<unsigned short*>(k_smem + k_smem_gemm4_offset + k_smem_gemm4_offset_k1 * 3);
            k_smem_gemm4_offset += k_smem_gemm4_offset_reg;
            asm volatile("v_pack_b32_f16 %0, %1, %2"
                         : "=v"(k_reg_to_gemm4[i].x)
                         : "v"(k_gemm4[0]), "v"(k_gemm4[1]));
            asm volatile("v_pack_b32_f16 %0, %1, %2"
                         : "=v"(k_reg_to_gemm4[i].y)
                         : "v"(k_gemm4[2]), "v"(k_gemm4[3]));
        }

        __syncthreads();

        char* v_smem = smem_ptr;
        *reinterpret_cast<float4*>(v_smem + kvqdo_smem_offset) = v_reg[0];
        *reinterpret_cast<float4*>(v_smem + kvqdo_smem_offset + kv_smem_reg_offset) = v_reg[1];
        *reinterpret_cast<float4*>(v_smem + kvqdo_smem_offset + kv_smem_reg_offset * 2) = v_reg[2];
        *reinterpret_cast<float4*>(v_smem + kvqdo_smem_offset + kv_smem_reg_offset * 3) = v_reg[3];

        __syncthreads();

        _BF16x8_t vt_reg_gemm2[4];
        vt_reg_gemm2[0] = *reinterpret_cast<_BF16x8_t*>(v_smem + k_smem_gemm0_offset);
        vt_reg_gemm2[1] = *reinterpret_cast<_BF16x8_t*>(v_smem + k_smem_gemm0_offset + k_smem_read_reg_offset);
        vt_reg_gemm2[2] = *reinterpret_cast<_BF16x8_t*>(v_smem + k_smem_gemm0_offset + k_smem_read_reg_offset * 2);
        vt_reg_gemm2[3] = *reinterpret_cast<_BF16x8_t*>(v_smem + k_smem_gemm0_offset + k_smem_read_reg_offset * 3);
        
        __syncthreads();

        // prepare core loop
        auto seqlen_q_start = 0;
        auto seqlen_q_end = kargs.seqlen_q;
        index_t i_total_loops = 0;
        // index_t seqlen_q_step = seqlen_q_start;
        const auto num_total_loop = integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0);

        // loading offset
        int q_do_load_offset = (threadIdx.x & 7) * 8 + (threadIdx.x >> 3) * kargs.stride_q;
        int q_do_load_reg_offset = kargs.stride_q * 32;

        constexpr int st_acc_gemmk_offset = 4;
        
        // hbm store offset
        int dq_acc_offset = n_id + k0_id * 4 * kargs.stride_q;
        const int dq_acc_wave_offset = (wave_id / 2) * 32 + (wave_id % 2) * 32 * kargs.stride_q;
        dq_acc_offset += dq_acc_wave_offset;

        // lds write offset
        // gemm4 ds offset
        constexpr int ds_padding_bytes = 8;
#if 1
        const int ds_lds_write_offset = n_id * 2 + k0_id * (128 * 2 + ds_padding_bytes) * 4 + n_wave_repeat_id * 32 * 2;
        constexpr int ds_lds_write_reg_offset = 128 * 2 + ds_padding_bytes;
        constexpr int ds_lds_gemm_m_group_offset = (128 * 2 + ds_padding_bytes) * 8;
        constexpr int ds_lds_gemm_m_acc_reg_offset = (128 * 2 + ds_padding_bytes) * 32;
#endif

        // lds read offset
        int q_gemm0_do_gemm2_offset = n_id * (64 * 2 + q_do_padding) + k0_id * 16;
        constexpr int q_gemm0_do_gemm2_reg_offset = 32 * (64 * 2 + q_do_padding);
        constexpr int q_gemm0_do_gemm2_gemmk_offset = 16 * 2;
        int q_gemm3_do_gemm1_offset = n_id * 4 + k0_id * (64 * 2 + q_do_padding) * 4;
        constexpr int q_gemm3_do_gemm1_reg_offset = 64 * 2 + q_do_padding;
        constexpr int q_gemm3_do_gemm1_gemmk_offset = (64 * 2 + q_do_padding) * 8;

        // gemm4 ds offset
        int ds_gemm4_offset = n_id * (128 * 2 + ds_padding_bytes) + k0_id * 4 * 2;
        const int ds_gemm4_m_wave_offset = (wave_id % 2) * (128 * 2 + ds_padding_bytes) * 32;
        ds_gemm4_offset += ds_gemm4_m_wave_offset;
        constexpr int ds_gemm4_kiter_offset = 8 * 2;

        // lse and d hbm offset and lds write read offset
        // constexpr int lse_d_step_offset = 64 * sizeof(float);
        constexpr int lse_d_reg_offset = 8 * sizeof(float);
        int lse_d_hbm_offset = threadIdx.x;
        int lse_d_lds_write_offset = threadIdx.x * sizeof(float);
        int lse_d_lds_read_offset = k0_id * 4 * sizeof(float);
        const float* lse_raw = lse_ptr + seqlen_q_start;
        const float* d_raw = d_ptr + seqlen_q_start;

        auto scale = kargs.scale;

        // acc clear
        CVecType dv_acc[2];
        dv_acc[0] = {0};
        dv_acc[1] = {0};
        
        CVecType dk_acc[2];
        dk_acc[0] = {0};
        dk_acc[1] = {0};
       
        float4 q_reg[2];
        float4 do_reg[2];
        float4 q_reg_tmp[2];
        float4 do_reg_tmp[2];

        q_reg[0] = *reinterpret_cast<const float4*>(q_ptr + q_do_load_offset);
        q_ptr += q_do_load_reg_offset;
        q_reg[1] = *reinterpret_cast<const float4*>(q_ptr + q_do_load_offset);
        q_ptr += q_do_load_reg_offset;
        
        do_reg[0] = *reinterpret_cast<const float4*>(do_ptr + q_do_load_offset);
        do_ptr += q_do_load_reg_offset;
        do_reg[1] = *reinterpret_cast<const float4*>(do_ptr + q_do_load_offset);
        do_ptr += q_do_load_reg_offset;
 
        // core loop
        do
        {
            // lse and d: HBM->reg->lds
            float lse_reg, d_reg;
            lse_reg = threadIdx.x < 64 ? lse_raw[lse_d_hbm_offset] : 0;
            lse_raw += 64;
            d_reg = threadIdx.x < 64 ? d_raw[lse_d_hbm_offset] : 0;
            d_raw += 64;
            char* lse_smem = smem_ptr;
            char* d_smem = lse_smem + 256;
            if (threadIdx.x < 64)
            {
                *reinterpret_cast<float*>(lse_smem + lse_d_lds_write_offset) = log2e_v<LSEDataType> * lse_reg;
                *reinterpret_cast<float*>(d_smem + lse_d_lds_write_offset) = d_reg;
            }

            //printf("thread:[%d], lse_d_hbm_offset=%d\n", type_convert<int>(threadIdx.x), lse_d_hbm_offset);
            //printf("thread:[%d], lse_d_lds_write_offset=%d\n", type_convert<int>(threadIdx.x), lse_d_lds_write_offset);
            //printf("thread:[%d], lse_reg=%f\n", type_convert<int>(threadIdx.x), lse_reg);

            // q and do: HBM->reg->lds
            // float4 q_reg[2];
            if(i_total_loops < (num_total_loop - 1))
            {
#if 1
            q_reg_tmp[0] = *reinterpret_cast<const float4*>(q_ptr + q_do_load_offset);
            q_ptr += q_do_load_reg_offset;
            q_reg_tmp[1] = *reinterpret_cast<const float4*>(q_ptr + q_do_load_offset);
            q_ptr += q_do_load_reg_offset;
#else
            q_reg_tmp[0] = k_reg[0];
            q_ptr += q_do_load_reg_offset + q_do_load_offset;
            q_reg_tmp[1] = k_reg[1];
            q_ptr += q_do_load_reg_offset;
#endif
            // float4 do_reg[2];
#if 1
            do_reg_tmp[0] = *reinterpret_cast<const float4*>(do_ptr + q_do_load_offset);
            do_ptr += q_do_load_reg_offset;
            do_reg_tmp[1] = *reinterpret_cast<const float4*>(do_ptr + q_do_load_offset);
            do_ptr += q_do_load_reg_offset;
#else
            do_reg_tmp[0] = v_reg[0];
            do_ptr += q_do_load_reg_offset;
            do_reg_tmp[1] = v_reg[1];
            do_ptr += q_do_load_reg_offset;
#endif
            }
           
            char* q_smem = d_smem + 64 * 4;
            char* do_smem = q_smem + (128 + q_do_padding) * 64;
            char* ds_smem = do_smem;
#if 1
            *reinterpret_cast<float4*>(q_smem + kvqdo_smem_offset) = q_reg[0];
            *reinterpret_cast<float4*>(q_smem + kvqdo_smem_offset + kv_smem_reg_offset) = q_reg[1];
            *reinterpret_cast<float4*>(do_smem + kvqdo_smem_offset) = do_reg[0];
            *reinterpret_cast<float4*>(do_smem + kvqdo_smem_offset + kv_smem_reg_offset) = do_reg[1];
#endif
            __syncthreads();

            // gemm 0
            _BF16x8_t q_reg_gemm0[4];
            CVecType st_acc[2];
            st_acc[0] = {0};
            st_acc[1] = {0};

#if 1
            q_reg_gemm0[0] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset);
            q_reg_gemm0[1] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_reg_offset);
            q_reg_gemm0[2] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_gemmk_offset);
            q_reg_gemm0[3] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_reg_offset + q_gemm0_do_gemm2_gemmk_offset);
#endif

#if 1
            st_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[0].xy[0], kt_reg_to_gemm0[0].xy[0], st_acc[0], 0, 0, 0);
            st_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[0].xy[1], kt_reg_to_gemm0[0].xy[1], st_acc[0], 0, 0, 0);
            st_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[1].xy[0], kt_reg_to_gemm0[0].xy[0], st_acc[1], 0, 0, 0);
            st_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[1].xy[1], kt_reg_to_gemm0[0].xy[1], st_acc[1], 0, 0, 0);
            st_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[2].xy[0], kt_reg_to_gemm0[1].xy[0], st_acc[0], 0, 0, 0);
            st_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[2].xy[1], kt_reg_to_gemm0[1].xy[1], st_acc[0], 0, 0, 0);
            st_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[3].xy[0], kt_reg_to_gemm0[1].xy[0], st_acc[1], 0, 0, 0);
            st_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[3].xy[1], kt_reg_to_gemm0[1].xy[1], st_acc[1], 0, 0, 0);
#endif

#if 1
            q_reg_gemm0[0] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_gemmk_offset * 2);
            q_reg_gemm0[1] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_reg_offset + q_gemm0_do_gemm2_gemmk_offset * 2);
            q_reg_gemm0[2] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_gemmk_offset * 3);
            q_reg_gemm0[3] = *reinterpret_cast<_BF16x8_t*>(q_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_reg_offset + q_gemm0_do_gemm2_gemmk_offset * 3);
#endif

#if 1
            st_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[0].xy[0], kt_reg_to_gemm0[2].xy[0], st_acc[0], 0, 0, 0);
            st_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[0].xy[1], kt_reg_to_gemm0[2].xy[1], st_acc[0], 0, 0, 0);
            st_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[1].xy[0], kt_reg_to_gemm0[2].xy[0], st_acc[1], 0, 0, 0);
            st_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[1].xy[1], kt_reg_to_gemm0[2].xy[1], st_acc[1], 0, 0, 0);
            st_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[2].xy[0], kt_reg_to_gemm0[3].xy[0], st_acc[0], 0, 0, 0);
            st_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[2].xy[1], kt_reg_to_gemm0[3].xy[1], st_acc[0], 0, 0, 0);
            st_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[3].xy[0], kt_reg_to_gemm0[3].xy[0], st_acc[1], 0, 0, 0);
            st_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[3].xy[1], kt_reg_to_gemm0[3].xy[1], st_acc[1], 0, 0, 0);
#endif

            // softmax
            floatx4 lse_d[2];
            lse_d[0] = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset);
            lse_d[1] = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset + lse_d_reg_offset);

            st_acc[0][0] = exp2(scale * st_acc[0][0] - lse_d[0][0]);
#if 1
            st_acc[0][1] = exp2(scale * st_acc[0][1] - lse_d[0][1]);
            st_acc[0][2] = exp2(scale * st_acc[0][2] - lse_d[0][2]);
            st_acc[0][3] = exp2(scale * st_acc[0][3] - lse_d[0][3]);
            st_acc[0][4] = exp2(scale * st_acc[0][4] - lse_d[1][0]);
            st_acc[0][5] = exp2(scale * st_acc[0][5] - lse_d[1][1]);
            st_acc[0][6] = exp2(scale * st_acc[0][6] - lse_d[1][2]);
            st_acc[0][7] = exp2(scale * st_acc[0][7] - lse_d[1][3]);

            lse_d[0] = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset + lse_d_reg_offset * 2);
            lse_d[1] = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset + lse_d_reg_offset * 3);

            st_acc[0][8]  = exp2(scale * st_acc[0][8]  - lse_d[0][0]);
            st_acc[0][9]  = exp2(scale * st_acc[0][9]  - lse_d[0][1]);
            st_acc[0][10] = exp2(scale * st_acc[0][10] - lse_d[0][2]);
            st_acc[0][11] = exp2(scale * st_acc[0][11] - lse_d[0][3]);
            st_acc[0][12] = exp2(scale * st_acc[0][12] - lse_d[1][0]);
            st_acc[0][13] = exp2(scale * st_acc[0][13] - lse_d[1][1]);
            st_acc[0][14] = exp2(scale * st_acc[0][14] - lse_d[1][2]);
            st_acc[0][15] = exp2(scale * st_acc[0][15] - lse_d[1][3]);

            lse_d[0] = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset + lse_d_reg_offset * 4);
            lse_d[1] = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset + lse_d_reg_offset * 5);

            st_acc[1][0] = exp2(scale * st_acc[1][0] - lse_d[0][0]);
            st_acc[1][1] = exp2(scale * st_acc[1][1] - lse_d[0][1]);
            st_acc[1][2] = exp2(scale * st_acc[1][2] - lse_d[0][2]);
            st_acc[1][3] = exp2(scale * st_acc[1][3] - lse_d[0][3]);
            st_acc[1][4] = exp2(scale * st_acc[1][4] - lse_d[1][0]);
            st_acc[1][5] = exp2(scale * st_acc[1][5] - lse_d[1][1]);
            st_acc[1][6] = exp2(scale * st_acc[1][6] - lse_d[1][2]);
            st_acc[1][7] = exp2(scale * st_acc[1][7] - lse_d[1][3]);

            lse_d[0] = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset + lse_d_reg_offset * 6);
            lse_d[1] = *reinterpret_cast<floatx4*>(lse_smem + lse_d_lds_read_offset + lse_d_reg_offset * 7);

            st_acc[1][8]  = exp2(scale * st_acc[1][8]  - lse_d[0][0]);
            st_acc[1][9]  = exp2(scale * st_acc[1][9]  - lse_d[0][1]);
            st_acc[1][10] = exp2(scale * st_acc[1][10] - lse_d[0][2]);
            st_acc[1][11] = exp2(scale * st_acc[1][11] - lse_d[0][3]);
            st_acc[1][12] = exp2(scale * st_acc[1][12] - lse_d[1][0]);
            st_acc[1][13] = exp2(scale * st_acc[1][13] - lse_d[1][1]);
            st_acc[1][14] = exp2(scale * st_acc[1][14] - lse_d[1][2]);
#endif
            st_acc[1][15] = exp2(scale * st_acc[1][15] - lse_d[1][3]);

            // gemm1
            bfloat16x4 pt_reg_gemm1;
            floatx4 do_reg_gemm1_tmp;
            uint2 do_reg_transpose_gemm1[2];
            bfloat16x4 do_reg_gemm1[2];

#pragma unroll
            for(int i_st_acc_reg_k = 0; i_st_acc_reg_k < 2; i_st_acc_reg_k++)
            {
#pragma unroll
                for(int i_st_acc = 0; i_st_acc < 4; i_st_acc++)
                {
#if 1
                    do_reg_gemm1_tmp[0] = *reinterpret_cast<float*>(do_smem + q_gemm3_do_gemm1_offset);
                    do_reg_gemm1_tmp[1] = *reinterpret_cast<float*>(do_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset);
                    do_reg_gemm1_tmp[2] = *reinterpret_cast<float*>(do_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset * 2);
                    do_reg_gemm1_tmp[3] = *reinterpret_cast<float*>(do_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset * 3);
#endif

                    pt_reg_gemm1[0] = type_convert<bf16_t, float>(st_acc[i_st_acc_reg_k][0 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[1] = type_convert<bf16_t, float>(st_acc[i_st_acc_reg_k][1 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[2] = type_convert<bf16_t, float>(st_acc[i_st_acc_reg_k][2 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[3] = type_convert<bf16_t, float>(st_acc[i_st_acc_reg_k][3 + st_acc_gemmk_offset * i_st_acc]);

                    do_reg_transpose_gemm1[0].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1]), bit_cast<uint32_t>(do_reg_gemm1_tmp[0]), m0);
                    do_reg_transpose_gemm1[0].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3]), bit_cast<uint32_t>(do_reg_gemm1_tmp[2]), m0);
                    do_reg_transpose_gemm1[1].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1]), bit_cast<uint32_t>(do_reg_gemm1_tmp[0]), m1);
                    do_reg_transpose_gemm1[1].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3]), bit_cast<uint32_t>(do_reg_gemm1_tmp[2]), m1);

                    do_reg_gemm1[0] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[0]);
                    do_reg_gemm1[1] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[1]);
            
                    dv_acc[0] = GCN_MFMA_INSTR_32(pt_reg_gemm1, do_reg_gemm1[0], dv_acc[0], 0, 0, 0);
                    dv_acc[1] = GCN_MFMA_INSTR_32(pt_reg_gemm1, do_reg_gemm1[1], dv_acc[1], 0, 0, 0);

                    do_smem += q_gemm3_do_gemm1_gemmk_offset;
                }
            }

            do_smem -= q_gemm3_do_gemm1_gemmk_offset * 8;

            // gemm 2
            CVecType dpt_acc[2];
            dpt_acc[0] = {0};
            dpt_acc[1] = {0};
#if 1
            q_reg_gemm0[0] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset);
            q_reg_gemm0[1] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_reg_offset);
            q_reg_gemm0[2] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_gemmk_offset);
            q_reg_gemm0[3] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_reg_offset + q_gemm0_do_gemm2_gemmk_offset);
#endif

#if 1
            dpt_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[0].xy[0], vt_reg_gemm2[0].xy[0], dpt_acc[0], 0, 0, 0);
            dpt_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[0].xy[1], vt_reg_gemm2[0].xy[1], dpt_acc[0], 0, 0, 0);
            dpt_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[1].xy[0], vt_reg_gemm2[0].xy[0], dpt_acc[1], 0, 0, 0);
            dpt_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[1].xy[1], vt_reg_gemm2[0].xy[1], dpt_acc[1], 0, 0, 0);
            dpt_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[2].xy[0], vt_reg_gemm2[1].xy[0], dpt_acc[0], 0, 0, 0);
            dpt_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[2].xy[1], vt_reg_gemm2[1].xy[1], dpt_acc[0], 0, 0, 0);
            dpt_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[3].xy[0], vt_reg_gemm2[1].xy[0], dpt_acc[1], 0, 0, 0);
            dpt_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[3].xy[1], vt_reg_gemm2[1].xy[1], dpt_acc[1], 0, 0, 0);
#endif

#if 1
            q_reg_gemm0[0] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_gemmk_offset * 2);
            q_reg_gemm0[1] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_reg_offset + q_gemm0_do_gemm2_gemmk_offset * 2);
            q_reg_gemm0[2] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_gemmk_offset * 3);
            q_reg_gemm0[3] = *reinterpret_cast<_BF16x8_t*>(do_smem + q_gemm0_do_gemm2_offset + q_gemm0_do_gemm2_reg_offset + q_gemm0_do_gemm2_gemmk_offset * 3);
#endif

#if 1
            dpt_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[0].xy[0], vt_reg_gemm2[2].xy[0], dpt_acc[0], 0, 0, 0);
            dpt_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[0].xy[1], vt_reg_gemm2[2].xy[1], dpt_acc[0], 0, 0, 0);
            dpt_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[1].xy[0], vt_reg_gemm2[2].xy[0], dpt_acc[1], 0, 0, 0);
            dpt_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[1].xy[1], vt_reg_gemm2[2].xy[1], dpt_acc[1], 0, 0, 0);
            dpt_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[2].xy[0], vt_reg_gemm2[3].xy[0], dpt_acc[0], 0, 0, 0);
            dpt_acc[0] = GCN_MFMA_INSTR_32(q_reg_gemm0[2].xy[1], vt_reg_gemm2[3].xy[1], dpt_acc[0], 0, 0, 0);
            dpt_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[3].xy[0], vt_reg_gemm2[3].xy[0], dpt_acc[1], 0, 0, 0);
            dpt_acc[1] = GCN_MFMA_INSTR_32(q_reg_gemm0[3].xy[1], vt_reg_gemm2[3].xy[1], dpt_acc[1], 0, 0, 0);
#endif

#if 1
            // ds
#pragma unroll
            for(int i_dpt = 0; i_dpt < 2; i_dpt++)
            {
#pragma unroll
                for(int i_dpt_vec = 0; i_dpt_vec < 16; i_dpt_vec += 8)
                {
                    lse_d[0] = *reinterpret_cast<floatx4*>(d_smem + lse_d_lds_read_offset);
                    d_smem += lse_d_reg_offset;
                    lse_d[1] = *reinterpret_cast<floatx4*>(d_smem + lse_d_lds_read_offset);
                    d_smem += lse_d_reg_offset;

                    dpt_acc[i_dpt][0 + i_dpt_vec] = st_acc[i_dpt][0 + i_dpt_vec] * (dpt_acc[i_dpt][0 + i_dpt_vec] - lse_d[0][0]);
                    dpt_acc[i_dpt][1 + i_dpt_vec] = st_acc[i_dpt][1 + i_dpt_vec] * (dpt_acc[i_dpt][1 + i_dpt_vec] - lse_d[0][1]);
                    dpt_acc[i_dpt][2 + i_dpt_vec] = st_acc[i_dpt][2 + i_dpt_vec] * (dpt_acc[i_dpt][2 + i_dpt_vec] - lse_d[0][2]);
                    dpt_acc[i_dpt][3 + i_dpt_vec] = st_acc[i_dpt][3 + i_dpt_vec] * (dpt_acc[i_dpt][3 + i_dpt_vec] - lse_d[0][3]);
                    dpt_acc[i_dpt][4 + i_dpt_vec] = st_acc[i_dpt][4 + i_dpt_vec] * (dpt_acc[i_dpt][4 + i_dpt_vec] - lse_d[1][0]);
                    dpt_acc[i_dpt][5 + i_dpt_vec] = st_acc[i_dpt][5 + i_dpt_vec] * (dpt_acc[i_dpt][5 + i_dpt_vec] - lse_d[1][1]);
                    dpt_acc[i_dpt][6 + i_dpt_vec] = st_acc[i_dpt][6 + i_dpt_vec] * (dpt_acc[i_dpt][6 + i_dpt_vec] - lse_d[1][2]);
                    dpt_acc[i_dpt][7 + i_dpt_vec] = st_acc[i_dpt][7 + i_dpt_vec] * (dpt_acc[i_dpt][7 + i_dpt_vec] - lse_d[1][3]);
                }    
            }
#endif

            // gemm 3
#pragma unroll
            for(int i_st_acc_reg_k = 0; i_st_acc_reg_k < 2; i_st_acc_reg_k++)
            {
#pragma unroll
                for(int i_st_acc = 0; i_st_acc < 4; i_st_acc++)
                {
#if 1 
                    do_reg_gemm1_tmp[0] = *reinterpret_cast<float*>(q_smem + q_gemm3_do_gemm1_offset);
                    do_reg_gemm1_tmp[1] = *reinterpret_cast<float*>(q_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset);
                    do_reg_gemm1_tmp[2] = *reinterpret_cast<float*>(q_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset * 2);
                    do_reg_gemm1_tmp[3] = *reinterpret_cast<float*>(q_smem + q_gemm3_do_gemm1_offset + q_gemm3_do_gemm1_reg_offset * 3);
#endif

                    pt_reg_gemm1[0] = type_convert<bf16_t, float>(dpt_acc[i_st_acc_reg_k][0 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[1] = type_convert<bf16_t, float>(dpt_acc[i_st_acc_reg_k][1 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[2] = type_convert<bf16_t, float>(dpt_acc[i_st_acc_reg_k][2 + st_acc_gemmk_offset * i_st_acc]);
                    pt_reg_gemm1[3] = type_convert<bf16_t, float>(dpt_acc[i_st_acc_reg_k][3 + st_acc_gemmk_offset * i_st_acc]);

                    do_reg_transpose_gemm1[0].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1]), bit_cast<uint32_t>(do_reg_gemm1_tmp[0]), m0);
                    do_reg_transpose_gemm1[0].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3]), bit_cast<uint32_t>(do_reg_gemm1_tmp[2]), m0);
                    do_reg_transpose_gemm1[1].x = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[1]), bit_cast<uint32_t>(do_reg_gemm1_tmp[0]), m1);
                    do_reg_transpose_gemm1[1].y = __builtin_amdgcn_perm(bit_cast<uint32_t>(do_reg_gemm1_tmp[3]), bit_cast<uint32_t>(do_reg_gemm1_tmp[2]), m1);

                    do_reg_gemm1[0] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[0]);
                    do_reg_gemm1[1] = bit_cast<bfloat16x4>(do_reg_transpose_gemm1[1]);
            
                    dk_acc[0] = GCN_MFMA_INSTR_32(pt_reg_gemm1, do_reg_gemm1[0], dk_acc[0], 0, 0, 0);
                    dk_acc[1] = GCN_MFMA_INSTR_32(pt_reg_gemm1, do_reg_gemm1[1], dk_acc[1], 0, 0, 0);

#if 1
                    *reinterpret_cast<bf16_t*>(ds_smem + ds_lds_write_offset + ds_lds_gemm_m_group_offset * i_st_acc + ds_lds_gemm_m_acc_reg_offset * i_st_acc_reg_k) = pt_reg_gemm1[0];
                    *reinterpret_cast<bf16_t*>(ds_smem + ds_lds_write_offset + ds_lds_gemm_m_group_offset * i_st_acc + ds_lds_gemm_m_acc_reg_offset * i_st_acc_reg_k + ds_lds_write_reg_offset) = pt_reg_gemm1[1];
                    *reinterpret_cast<bf16_t*>(ds_smem + ds_lds_write_offset + ds_lds_gemm_m_group_offset * i_st_acc + ds_lds_gemm_m_acc_reg_offset * i_st_acc_reg_k + ds_lds_write_reg_offset * 2) = pt_reg_gemm1[2];
                    *reinterpret_cast<bf16_t*>(ds_smem + ds_lds_write_offset + ds_lds_gemm_m_group_offset * i_st_acc + ds_lds_gemm_m_acc_reg_offset * i_st_acc_reg_k + ds_lds_write_reg_offset * 3) = pt_reg_gemm1[3];
#endif

                    q_smem += q_gemm3_do_gemm1_gemmk_offset;
                }
            }

            // gemm 4
            bfloat16x4 dp_reg_gemm4[2];
            st_acc[0] = {0};

            __syncthreads();
            int i_k_gemmk_gemm4 = 0;

#pragma unroll
            for(int i_gemm4_k = 0; i_gemm4_k < 128; i_gemm4_k += 16)
            {
#if 1
                dp_reg_gemm4[0] = *reinterpret_cast<bfloat16x4*>(ds_smem + ds_gemm4_offset);
                ds_smem += ds_gemm4_kiter_offset;
                dp_reg_gemm4[1] = *reinterpret_cast<bfloat16x4*>(ds_smem + ds_gemm4_offset);
                ds_smem += ds_gemm4_kiter_offset;
#else
                dp_reg_gemm4[0] = {1};
                ds_smem += ds_gemm4_kiter_offset + ds_gemm4_offset;
                dp_reg_gemm4[1] = {1};
                ds_smem += ds_gemm4_kiter_offset;

#endif
                
                st_acc[0] = GCN_MFMA_INSTR_32(dp_reg_gemm4[0], bit_cast<bfloat16x4>(k_reg_to_gemm4[i_k_gemmk_gemm4]), st_acc[0], 0, 0, 0);
                i_k_gemmk_gemm4++;
                st_acc[0] = GCN_MFMA_INSTR_32(dp_reg_gemm4[1], bit_cast<bfloat16x4>(k_reg_to_gemm4[i_k_gemmk_gemm4]), st_acc[0], 0, 0, 0);
                i_k_gemmk_gemm4++;
            }

#pragma unroll
            for(int i_dq_acc = 0; i_dq_acc < 4; i_dq_acc++)
            {
#if 0
                *(dq_acc_ptr_tmp + dq_acc_offset) = st_acc[0][i_dq_acc * 4 + 0] * kargs.raw_scale;
                dq_acc_ptr_tmp += kargs.stride_q;
                *(dq_acc_ptr_tmp + dq_acc_offset) = st_acc[0][i_dq_acc * 4 + 1] * kargs.raw_scale;
                dq_acc_ptr_tmp += kargs.stride_q;
                *(dq_acc_ptr_tmp + dq_acc_offset) = st_acc[0][i_dq_acc * 4 + 2] * kargs.raw_scale;
                dq_acc_ptr_tmp += kargs.stride_q;
                *(dq_acc_ptr_tmp + dq_acc_offset) = st_acc[0][i_dq_acc * 4 + 3] * kargs.raw_scale;
                dq_acc_ptr_tmp += kargs.stride_q;
#else
                unsafeAtomicAdd(dq_acc_ptr_tmp + dq_acc_offset, st_acc[0][i_dq_acc * 4 + 0] * kargs.raw_scale);
                dq_acc_ptr_tmp += kargs.stride_q;
                unsafeAtomicAdd(dq_acc_ptr_tmp + dq_acc_offset, st_acc[0][i_dq_acc * 4 + 1] * kargs.raw_scale);
                dq_acc_ptr_tmp += kargs.stride_q;
                unsafeAtomicAdd(dq_acc_ptr_tmp + dq_acc_offset, st_acc[0][i_dq_acc * 4 + 2] * kargs.raw_scale);
                dq_acc_ptr_tmp += kargs.stride_q;
                unsafeAtomicAdd(dq_acc_ptr_tmp + dq_acc_offset, st_acc[0][i_dq_acc * 4 + 3] * kargs.raw_scale);
                dq_acc_ptr_tmp += 5 * kargs.stride_q;
#endif
            }
            dq_acc_ptr_tmp += 32 * kargs.stride_q;
            
            __syncthreads();

            float4 q_do_swap_tmp;
#pragma unroll
            for(int i_qdo_reg = 0; i_qdo_reg < 2; i_qdo_reg++)
            {
                // q reg
                q_do_swap_tmp = q_reg_tmp[i_qdo_reg];
                q_reg_tmp[i_qdo_reg] = q_reg[i_qdo_reg];
                q_reg[i_qdo_reg] = q_do_swap_tmp;
                // do reg
                q_do_swap_tmp = do_reg_tmp[i_qdo_reg];
                do_reg_tmp[i_qdo_reg] = do_reg[i_qdo_reg];
                do_reg[i_qdo_reg] = q_do_swap_tmp;
            }

            i_total_loops += 1;
            // seqlen_q_step += kM0;
            
        } while(i_total_loops < (num_total_loop - 0));

        // write out dv
        const int& stride_v_seq = kargs.stride_v;
        int dv_hbm_offset = n_id * 2 + k0_id * stride_v_seq * 4;

        // const int dv_hbm_reg_offset = stride_v_seq;
        // const int dv_hbm_a_group_offset = stride_v_seq * 8;
        const int wave_offset_gemm1_gemm3 = wave_id * 32 * stride_v_seq;
        dv_hbm_offset += wave_offset_gemm1_gemm3;
        const int reg_offset_gemm1_gemm3 = stride_v_seq;
        const int group_offset_gemm1_gemm3 = stride_v_seq * 8;
        
        dv_ptr += i_n0 * kargs.stride_k;
#pragma unroll
        for (int i_dv = 0; i_dv < 4; i_dv++)
        {
            uint32_t dv_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4]), m1);
            char* dv_ptr_tmp = reinterpret_cast<char*>(dv_ptr + dv_hbm_offset + group_offset_gemm1_gemm3 * i_dv);
            *reinterpret_cast<float*>(dv_ptr_tmp) = bit_cast<float>(dv_pack);
            dv_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 1]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 1]), m1);
            dv_ptr_tmp = reinterpret_cast<char*>(dv_ptr + dv_hbm_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3);
            *reinterpret_cast<float*>(dv_ptr_tmp) = bit_cast<float>(dv_pack);
            dv_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 2]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 2]), m1);
            dv_ptr_tmp = reinterpret_cast<char*>(dv_ptr + dv_hbm_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 2);
            *reinterpret_cast<float*>(dv_ptr_tmp) = bit_cast<float>(dv_pack);
            dv_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dv_acc[1][i_dv * 4 + 3]), bit_cast<uint32_t>(dv_acc[0][i_dv * 4 + 3]), m1);
            dv_ptr_tmp = reinterpret_cast<char*>(dv_ptr + dv_hbm_offset + group_offset_gemm1_gemm3 * i_dv + reg_offset_gemm1_gemm3 * 3);
            *reinterpret_cast<float*>(dv_ptr_tmp) = bit_cast<float>(dv_pack);
        }

        // dk = dk * scale
        dk_acc[0] *= kargs.raw_scale;
        dk_acc[1] *= kargs.raw_scale;
        
        dk_ptr += i_n0 * kargs.stride_k;
#pragma unroll
        for (int i_dk = 0; i_dk < 4; i_dk++)
        {
            uint32_t dk_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4]), m1);
            char* dk_ptr_tmp = reinterpret_cast<char*>(dk_ptr + dv_hbm_offset + group_offset_gemm1_gemm3 * i_dk);
            *reinterpret_cast<float*>(dk_ptr_tmp) = bit_cast<float>(dk_pack);
            dk_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 1]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 1]), m1);
            dk_ptr_tmp = reinterpret_cast<char*>(dk_ptr + dv_hbm_offset + group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3);
            *reinterpret_cast<float*>(dk_ptr_tmp) = bit_cast<float>(dk_pack);
            dk_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 2]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 2]), m1);
            dk_ptr_tmp = reinterpret_cast<char*>(dk_ptr + dv_hbm_offset + group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 2);
            *reinterpret_cast<float*>(dk_ptr_tmp) = bit_cast<float>(dk_pack);
            dk_pack = __builtin_amdgcn_perm(bit_cast<uint32_t>(dk_acc[1][i_dk * 4 + 3]), bit_cast<uint32_t>(dk_acc[0][i_dk * 4 + 3]), m1);
            dk_ptr_tmp = reinterpret_cast<char*>(dk_ptr + dv_hbm_offset + group_offset_gemm1_gemm3 * i_dk + reg_offset_gemm1_gemm3 * 3);
            *reinterpret_cast<float*>(dk_ptr_tmp) = bit_cast<float>(dk_pack);
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

        // const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kM0);
        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * 1);

        long_index_t batch_offset_dq = 0;
        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            batch_offset_dq                = query_start * kargs.stride_dq;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
            const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
            kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
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
            const AccDataType* dq_acc_ptr =
                reinterpret_cast<const AccDataType*>(kargs.dq_acc_ptr) +
                static_cast<long_index_t>(i_nhead) * (kargs.nhead_stride_dq) + batch_offset_dq;

            
            dq_ptr += kargs.stride_dq * i_m0;
            dq_acc_ptr += kargs.stride_dq * i_m0;
            constexpr int32_t m1 = 0x07060302;
            int reg_offset = threadIdx.x * 4;
            int reg_offset_acc = reg_offset;
            int num_head_q = kargs.stride_dq / kargs.hdim_q;
            constexpr int dq_unroll = 4;
            float4 dq_acc_reg[dq_unroll];
            float2 dq_reg[dq_unroll];

            char* dq_ptr_tmp;

            for(int i = 0; i < num_head_q; i += 16 * dq_unroll)
            {
#pragma unroll
                for(int j = 0; j < dq_unroll; j++)
                {
                    dq_acc_reg[j] = *reinterpret_cast<const float4*>(dq_acc_ptr + reg_offset_acc);
                    reg_offset_acc += 16 * 64;
                }
#pragma unroll
                for(int j = 0; j < dq_unroll; j++)
                {
                    dq_reg[j].x = bit_cast<float>(__builtin_amdgcn_perm(bit_cast<uint32_t>(dq_acc_reg[j].y), bit_cast<uint32_t>(dq_acc_reg[j].x), m1));
                    dq_reg[j].y = bit_cast<float>(__builtin_amdgcn_perm(bit_cast<uint32_t>(dq_acc_reg[j].w), bit_cast<uint32_t>(dq_acc_reg[j].z), m1));
                }
#pragma unroll
                for(int j = 0; j < dq_unroll; j++)
                {
                    dq_ptr_tmp = reinterpret_cast<char*>(dq_ptr + reg_offset);
                    *reinterpret_cast<float2*>(dq_ptr_tmp) = dq_reg[j];
                    reg_offset += 16 * 64;
                }
            }

            // FmhaBwdConvertQGrad{}(dq_acc_dram_window, dq_dram_window);
        }
    }
};

} // namespace ck_tile
