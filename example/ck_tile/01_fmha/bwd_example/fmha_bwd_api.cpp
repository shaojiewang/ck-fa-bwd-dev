// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

// auto generated by generate.py
#include "fmha_bwd.hpp"

#include <iostream>

template <typename dot_do_o_trait_, typename dq_dk_dv_trait_, typename convert_dq_trait_>
float fmha_bwd_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << fmha_bwd_dq_dk_dv_get_name_<dq_dk_dv_trait_>() << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dq_dk_dv_oneshot_<dq_dk_dv_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

float fmha_bwd(fmha_bwd_traits t, fmha_bwd_args a, const ck_tile::stream_config& s){
    float r = -1;
    if(t.data_type.compare("bf16") == 0){
        if (t.hdim_q <= 64 && t.hdim_v <= 64) {
            if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                        (true) && (true) && (a.hdim_q % 64 != 0) && (a.hdim_v % 64 != 0) && (t.is_deterministic == true)) {
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, ck_tile::bf16_t, true, true, true>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<64, ck_tile::bf16_t, true, ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::BlockDropout<false, true,  false>, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, true, true, true, true>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, ck_tile::bf16_t, true, true, true, true>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }
            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                        (true) && (true) && (a.hdim_q % 64 != 0) && (a.hdim_v % 64 != 0) && (t.is_deterministic == false)) {
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, ck_tile::bf16_t, true, true, true>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<64, ck_tile::bf16_t, true, ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::BlockDropout<false, true,  false>, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, true, true, true, false>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, ck_tile::bf16_t, true, true, true, false>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }
            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                        (true) && (true) && (a.hdim_q % 64 != 0) && (a.hdim_v % 64 == 0) && (t.is_deterministic == true)) {
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, ck_tile::bf16_t, true, true, false>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<64, ck_tile::bf16_t, true, ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::BlockDropout<false, true,  false>, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, true, true, false, true>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, ck_tile::bf16_t, true, true, true, true>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }
            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                        (true) && (true) && (a.hdim_q % 64 != 0) && (a.hdim_v % 64 == 0) && (t.is_deterministic == false)) {
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, ck_tile::bf16_t, true, true, false>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<64, ck_tile::bf16_t, true, ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::BlockDropout<false, true,  false>, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, true, true, false, false>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, ck_tile::bf16_t, true, true, true, false>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }
            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                        (true) && (true) && (a.hdim_q % 64 == 0) && (a.hdim_v % 64 != 0) && (t.is_deterministic == true)) {
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, ck_tile::bf16_t, true, true, true>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<64, ck_tile::bf16_t, true, ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::BlockDropout<false, true,  false>, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, true, false, true, true>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, ck_tile::bf16_t, true, true, false, true>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }
            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                        (true) && (true) && (a.hdim_q % 64 == 0) && (a.hdim_v % 64 != 0) && (t.is_deterministic == false)) {
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, ck_tile::bf16_t, true, true, true>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<64, ck_tile::bf16_t, true, ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::BlockDropout<false, true,  false>, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, true, false, true, false>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, ck_tile::bf16_t, true, true, false, false>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }
            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                        (true) && (true) && (a.hdim_q % 64 == 0) && (a.hdim_v % 64 == 0) && (t.is_deterministic == true)) {
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, ck_tile::bf16_t, true, true, false>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<64, ck_tile::bf16_t, true, ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::BlockDropout<false, true,  false>, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, true, false, false, true>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, ck_tile::bf16_t, true, true, false, true>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }
            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                        (true) && (true) && (a.hdim_q % 64 == 0) && (a.hdim_v % 64 == 0) && (t.is_deterministic == false)) {
                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, ck_tile::bf16_t, true, true, false>;
                using dq_dk_dv_trait_ = fmha_bwd_dq_dk_dv_traits_<64, ck_tile::bf16_t, true, ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR, ck_tile::SimplifiedGenericAttentionMask<false>, ck_tile::BlockDropout<false, true,  false>, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, true, false, false, false>;
                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, ck_tile::bf16_t, true, true, false, false>;
                r = fmha_bwd_<dot_do_o_trait_, dq_dk_dv_trait_, convert_dq_trait_>(s, a);
                return r;
            }

        }

    }

    return r;
}
