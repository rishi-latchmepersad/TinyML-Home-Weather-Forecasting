/**
  ******************************************************************************
  * @file    forecast_temp_ml_model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-12-13T17:21:04-0400
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "forecast_temp_ml_model.h"
#include "forecast_temp_ml_model_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_forecast_temp_ml_model
 
#undef AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE
#define AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE     "0x8bafba61fb61e4202d85dbe56a2922ae"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-12-13T17:21:04-0400"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_FORECAST_TEMP_ML_MODEL_N_BATCHES
#define AI_FORECAST_TEMP_ML_MODEL_N_BATCHES         (1)

static ai_ptr g_forecast_temp_ml_model_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_forecast_temp_ml_model_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gemm_253_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_keras_tensor_60_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 672, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  transpose_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 672, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output2_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output3_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output4_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output5_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output6_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output7_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output8_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output9_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output10_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output11_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output12_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output13_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output14_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output15_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output16_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output17_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output18_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output19_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output20_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output21_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output22_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output23_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output24_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output25_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output26_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output27_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output28_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output29_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output30_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output31_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output32_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output33_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output34_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output35_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output36_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output37_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output38_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output39_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output40_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output41_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output42_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output43_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output44_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output45_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output46_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  unpack_3_output47_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  nl_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conversion_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  gemm_22_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_23_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  nl_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  conversion_25_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  gemm_37_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_38_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  nl_39_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  conversion_40_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  gemm_41_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  gemm_52_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  nl_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  conversion_55_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  gemm_56_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  gemm_67_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_68_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  nl_69_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  conversion_70_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  gemm_71_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_72_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  nl_73_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  conversion_74_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  gemm_75_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_76_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  nl_77_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  conversion_78_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  gemm_79_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_80_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  nl_81_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  conversion_82_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  gemm_83_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_84_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  nl_85_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  conversion_86_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  gemm_87_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_88_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  nl_89_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  conversion_90_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  gemm_91_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_92_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  nl_93_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  conversion_94_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  gemm_95_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  gemm_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_96_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  nl_97_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  conversion_98_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  gemm_99_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_100_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  nl_101_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  conversion_102_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  gemm_103_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  gemm_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_104_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  nl_105_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  conversion_106_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  gemm_107_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  gemm_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_108_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  nl_109_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  conversion_110_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  gemm_111_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  gemm_27_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_112_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  nl_113_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  conversion_114_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  gemm_115_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  gemm_28_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_116_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  nl_117_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  conversion_118_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  gemm_119_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  gemm_29_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_120_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  nl_121_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  conversion_122_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  gemm_123_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  gemm_30_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_124_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  nl_125_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  conversion_126_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  gemm_127_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  gemm_31_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_128_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  nl_129_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  conversion_130_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  gemm_131_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  gemm_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_132_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  nl_133_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  conversion_134_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  gemm_135_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  gemm_33_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_136_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  nl_137_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  conversion_138_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  gemm_139_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  gemm_34_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_140_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  nl_141_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  conversion_142_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  gemm_143_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  gemm_35_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_144_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  nl_145_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  conversion_146_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  gemm_147_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_148_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  nl_149_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  conversion_150_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  gemm_151_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_152_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  nl_153_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  conversion_154_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  gemm_155_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  gemm_43_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_156_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  nl_157_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  conversion_158_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  gemm_159_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  gemm_44_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_160_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  nl_161_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  conversion_162_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  gemm_163_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  gemm_45_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_164_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  nl_165_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  conversion_166_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  gemm_167_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_168_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  nl_169_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  conversion_170_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  gemm_171_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  gemm_47_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_172_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  nl_173_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  conversion_174_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  gemm_175_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_176_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  nl_177_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  conversion_178_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  gemm_179_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#212 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#213 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_180_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#214 */
AI_ARRAY_OBJ_DECLARE(
  nl_181_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#215 */
AI_ARRAY_OBJ_DECLARE(
  conversion_182_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#216 */
AI_ARRAY_OBJ_DECLARE(
  gemm_183_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#217 */
AI_ARRAY_OBJ_DECLARE(
  gemm_50_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#218 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_184_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#219 */
AI_ARRAY_OBJ_DECLARE(
  nl_185_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#220 */
AI_ARRAY_OBJ_DECLARE(
  conversion_186_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#221 */
AI_ARRAY_OBJ_DECLARE(
  gemm_187_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#222 */
AI_ARRAY_OBJ_DECLARE(
  gemm_51_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#223 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_188_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#224 */
AI_ARRAY_OBJ_DECLARE(
  nl_189_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#225 */
AI_ARRAY_OBJ_DECLARE(
  conversion_190_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#226 */
AI_ARRAY_OBJ_DECLARE(
  gemm_191_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#227 */
AI_ARRAY_OBJ_DECLARE(
  gemm_57_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#228 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_192_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#229 */
AI_ARRAY_OBJ_DECLARE(
  nl_193_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#230 */
AI_ARRAY_OBJ_DECLARE(
  conversion_194_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#231 */
AI_ARRAY_OBJ_DECLARE(
  gemm_195_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#232 */
AI_ARRAY_OBJ_DECLARE(
  gemm_58_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#233 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_196_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#234 */
AI_ARRAY_OBJ_DECLARE(
  nl_197_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#235 */
AI_ARRAY_OBJ_DECLARE(
  conversion_198_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#236 */
AI_ARRAY_OBJ_DECLARE(
  gemm_199_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#237 */
AI_ARRAY_OBJ_DECLARE(
  gemm_59_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#238 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_200_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#239 */
AI_ARRAY_OBJ_DECLARE(
  nl_201_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#240 */
AI_ARRAY_OBJ_DECLARE(
  conversion_202_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#241 */
AI_ARRAY_OBJ_DECLARE(
  gemm_203_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#242 */
AI_ARRAY_OBJ_DECLARE(
  gemm_60_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#243 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_204_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#244 */
AI_ARRAY_OBJ_DECLARE(
  nl_205_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#245 */
AI_ARRAY_OBJ_DECLARE(
  conversion_206_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#246 */
AI_ARRAY_OBJ_DECLARE(
  gemm_207_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#247 */
AI_ARRAY_OBJ_DECLARE(
  gemm_61_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#248 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_208_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#249 */
AI_ARRAY_OBJ_DECLARE(
  nl_209_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#250 */
AI_ARRAY_OBJ_DECLARE(
  conversion_210_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#251 */
AI_ARRAY_OBJ_DECLARE(
  gemm_211_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#252 */
AI_ARRAY_OBJ_DECLARE(
  gemm_62_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#253 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_212_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#254 */
AI_ARRAY_OBJ_DECLARE(
  nl_213_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#255 */
AI_ARRAY_OBJ_DECLARE(
  conversion_214_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#256 */
AI_ARRAY_OBJ_DECLARE(
  gemm_215_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#257 */
AI_ARRAY_OBJ_DECLARE(
  gemm_63_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#258 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_216_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#259 */
AI_ARRAY_OBJ_DECLARE(
  nl_217_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#260 */
AI_ARRAY_OBJ_DECLARE(
  conversion_218_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#261 */
AI_ARRAY_OBJ_DECLARE(
  gemm_219_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#262 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#263 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_220_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#264 */
AI_ARRAY_OBJ_DECLARE(
  nl_221_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#265 */
AI_ARRAY_OBJ_DECLARE(
  conversion_222_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#266 */
AI_ARRAY_OBJ_DECLARE(
  gemm_223_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#267 */
AI_ARRAY_OBJ_DECLARE(
  gemm_65_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#268 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_224_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#269 */
AI_ARRAY_OBJ_DECLARE(
  nl_225_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#270 */
AI_ARRAY_OBJ_DECLARE(
  conversion_226_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#271 */
AI_ARRAY_OBJ_DECLARE(
  gemm_227_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#272 */
AI_ARRAY_OBJ_DECLARE(
  gemm_66_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#273 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_228_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#274 */
AI_ARRAY_OBJ_DECLARE(
  nl_229_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#275 */
AI_ARRAY_OBJ_DECLARE(
  conversion_230_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#276 */
AI_ARRAY_OBJ_DECLARE(
  gemm_231_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#277 */
AI_ARRAY_OBJ_DECLARE(
  gemm_232_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#278 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_233_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#279 */
AI_ARRAY_OBJ_DECLARE(
  nl_234_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#280 */
AI_ARRAY_OBJ_DECLARE(
  conversion_235_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#281 */
AI_ARRAY_OBJ_DECLARE(
  gemm_236_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#282 */
AI_ARRAY_OBJ_DECLARE(
  gemm_237_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#283 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_238_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#284 */
AI_ARRAY_OBJ_DECLARE(
  nl_239_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#285 */
AI_ARRAY_OBJ_DECLARE(
  conversion_240_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#286 */
AI_ARRAY_OBJ_DECLARE(
  gemm_241_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#287 */
AI_ARRAY_OBJ_DECLARE(
  gemm_242_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#288 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_243_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#289 */
AI_ARRAY_OBJ_DECLARE(
  nl_244_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#290 */
AI_ARRAY_OBJ_DECLARE(
  conversion_245_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#291 */
AI_ARRAY_OBJ_DECLARE(
  pack_246_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3072, AI_STATIC)

/* Array#292 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#293 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#294 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output2_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#295 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output3_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#296 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output4_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#297 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output5_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#298 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output6_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#299 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output7_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#300 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output8_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#301 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output9_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#302 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output10_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#303 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output11_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#304 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output12_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#305 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output13_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#306 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output14_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#307 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output15_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#308 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output16_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#309 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output17_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#310 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output18_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#311 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output19_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#312 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output20_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#313 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output21_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#314 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output22_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#315 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output23_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#316 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output24_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#317 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output25_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#318 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output26_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#319 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output27_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#320 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output28_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#321 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output29_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#322 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output30_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#323 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output31_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#324 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output32_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#325 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output33_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#326 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output34_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#327 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output35_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#328 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output36_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#329 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output37_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#330 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output38_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#331 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output39_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#332 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output40_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#333 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output41_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#334 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output42_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#335 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output43_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#336 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output44_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#337 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output45_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#338 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output46_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#339 */
AI_ARRAY_OBJ_DECLARE(
  unpack_252_output47_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#340 */
AI_ARRAY_OBJ_DECLARE(
  gemm_254_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#341 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_255_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#342 */
AI_ARRAY_OBJ_DECLARE(
  nl_256_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#343 */
AI_ARRAY_OBJ_DECLARE(
  gemm_257_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#344 */
AI_ARRAY_OBJ_DECLARE(
  gemm_268_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#345 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_269_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#346 */
AI_ARRAY_OBJ_DECLARE(
  nl_270_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#347 */
AI_ARRAY_OBJ_DECLARE(
  gemm_271_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#348 */
AI_ARRAY_OBJ_DECLARE(
  gemm_282_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#349 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_283_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#350 */
AI_ARRAY_OBJ_DECLARE(
  nl_284_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#351 */
AI_ARRAY_OBJ_DECLARE(
  gemm_285_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#352 */
AI_ARRAY_OBJ_DECLARE(
  gemm_296_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#353 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_297_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#354 */
AI_ARRAY_OBJ_DECLARE(
  nl_298_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#355 */
AI_ARRAY_OBJ_DECLARE(
  gemm_299_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#356 */
AI_ARRAY_OBJ_DECLARE(
  gemm_310_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#357 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_311_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#358 */
AI_ARRAY_OBJ_DECLARE(
  nl_312_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#359 */
AI_ARRAY_OBJ_DECLARE(
  gemm_313_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#360 */
AI_ARRAY_OBJ_DECLARE(
  gemm_258_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#361 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_314_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#362 */
AI_ARRAY_OBJ_DECLARE(
  nl_315_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#363 */
AI_ARRAY_OBJ_DECLARE(
  gemm_316_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#364 */
AI_ARRAY_OBJ_DECLARE(
  gemm_259_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#365 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_317_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#366 */
AI_ARRAY_OBJ_DECLARE(
  nl_318_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#367 */
AI_ARRAY_OBJ_DECLARE(
  gemm_319_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#368 */
AI_ARRAY_OBJ_DECLARE(
  gemm_260_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#369 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_320_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#370 */
AI_ARRAY_OBJ_DECLARE(
  nl_321_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#371 */
AI_ARRAY_OBJ_DECLARE(
  gemm_322_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#372 */
AI_ARRAY_OBJ_DECLARE(
  gemm_261_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#373 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_323_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#374 */
AI_ARRAY_OBJ_DECLARE(
  nl_324_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#375 */
AI_ARRAY_OBJ_DECLARE(
  gemm_325_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#376 */
AI_ARRAY_OBJ_DECLARE(
  gemm_262_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#377 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_326_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#378 */
AI_ARRAY_OBJ_DECLARE(
  nl_327_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#379 */
AI_ARRAY_OBJ_DECLARE(
  gemm_328_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#380 */
AI_ARRAY_OBJ_DECLARE(
  gemm_263_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#381 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_329_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#382 */
AI_ARRAY_OBJ_DECLARE(
  nl_330_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#383 */
AI_ARRAY_OBJ_DECLARE(
  gemm_331_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#384 */
AI_ARRAY_OBJ_DECLARE(
  gemm_264_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#385 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_332_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#386 */
AI_ARRAY_OBJ_DECLARE(
  nl_333_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#387 */
AI_ARRAY_OBJ_DECLARE(
  gemm_334_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#388 */
AI_ARRAY_OBJ_DECLARE(
  gemm_265_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#389 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_335_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#390 */
AI_ARRAY_OBJ_DECLARE(
  nl_336_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#391 */
AI_ARRAY_OBJ_DECLARE(
  gemm_337_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#392 */
AI_ARRAY_OBJ_DECLARE(
  gemm_266_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#393 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_338_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#394 */
AI_ARRAY_OBJ_DECLARE(
  nl_339_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#395 */
AI_ARRAY_OBJ_DECLARE(
  gemm_340_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#396 */
AI_ARRAY_OBJ_DECLARE(
  gemm_267_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#397 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_341_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#398 */
AI_ARRAY_OBJ_DECLARE(
  nl_342_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#399 */
AI_ARRAY_OBJ_DECLARE(
  gemm_343_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#400 */
AI_ARRAY_OBJ_DECLARE(
  gemm_272_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#401 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_344_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#402 */
AI_ARRAY_OBJ_DECLARE(
  nl_345_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#403 */
AI_ARRAY_OBJ_DECLARE(
  gemm_346_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#404 */
AI_ARRAY_OBJ_DECLARE(
  gemm_273_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#405 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_347_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#406 */
AI_ARRAY_OBJ_DECLARE(
  nl_348_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#407 */
AI_ARRAY_OBJ_DECLARE(
  gemm_349_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#408 */
AI_ARRAY_OBJ_DECLARE(
  gemm_274_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#409 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_350_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#410 */
AI_ARRAY_OBJ_DECLARE(
  nl_351_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#411 */
AI_ARRAY_OBJ_DECLARE(
  gemm_352_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#412 */
AI_ARRAY_OBJ_DECLARE(
  gemm_275_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#413 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_353_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#414 */
AI_ARRAY_OBJ_DECLARE(
  nl_354_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#415 */
AI_ARRAY_OBJ_DECLARE(
  gemm_355_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#416 */
AI_ARRAY_OBJ_DECLARE(
  gemm_276_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#417 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_356_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#418 */
AI_ARRAY_OBJ_DECLARE(
  nl_357_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#419 */
AI_ARRAY_OBJ_DECLARE(
  gemm_358_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#420 */
AI_ARRAY_OBJ_DECLARE(
  gemm_277_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#421 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_359_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#422 */
AI_ARRAY_OBJ_DECLARE(
  nl_360_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#423 */
AI_ARRAY_OBJ_DECLARE(
  gemm_361_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#424 */
AI_ARRAY_OBJ_DECLARE(
  gemm_278_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#425 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_362_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#426 */
AI_ARRAY_OBJ_DECLARE(
  nl_363_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#427 */
AI_ARRAY_OBJ_DECLARE(
  gemm_364_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#428 */
AI_ARRAY_OBJ_DECLARE(
  gemm_279_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#429 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_365_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#430 */
AI_ARRAY_OBJ_DECLARE(
  nl_366_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#431 */
AI_ARRAY_OBJ_DECLARE(
  gemm_367_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#432 */
AI_ARRAY_OBJ_DECLARE(
  gemm_280_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#433 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_368_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#434 */
AI_ARRAY_OBJ_DECLARE(
  nl_369_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#435 */
AI_ARRAY_OBJ_DECLARE(
  gemm_370_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#436 */
AI_ARRAY_OBJ_DECLARE(
  gemm_281_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#437 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_371_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#438 */
AI_ARRAY_OBJ_DECLARE(
  nl_372_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#439 */
AI_ARRAY_OBJ_DECLARE(
  gemm_373_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#440 */
AI_ARRAY_OBJ_DECLARE(
  gemm_286_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#441 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_374_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#442 */
AI_ARRAY_OBJ_DECLARE(
  nl_375_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#443 */
AI_ARRAY_OBJ_DECLARE(
  gemm_376_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#444 */
AI_ARRAY_OBJ_DECLARE(
  gemm_287_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#445 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_377_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#446 */
AI_ARRAY_OBJ_DECLARE(
  nl_378_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#447 */
AI_ARRAY_OBJ_DECLARE(
  gemm_379_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#448 */
AI_ARRAY_OBJ_DECLARE(
  gemm_288_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#449 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_380_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#450 */
AI_ARRAY_OBJ_DECLARE(
  nl_381_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#451 */
AI_ARRAY_OBJ_DECLARE(
  gemm_382_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#452 */
AI_ARRAY_OBJ_DECLARE(
  gemm_289_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#453 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_383_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#454 */
AI_ARRAY_OBJ_DECLARE(
  nl_384_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#455 */
AI_ARRAY_OBJ_DECLARE(
  gemm_385_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#456 */
AI_ARRAY_OBJ_DECLARE(
  gemm_290_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#457 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_386_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#458 */
AI_ARRAY_OBJ_DECLARE(
  nl_387_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#459 */
AI_ARRAY_OBJ_DECLARE(
  gemm_388_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#460 */
AI_ARRAY_OBJ_DECLARE(
  gemm_291_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#461 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_389_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#462 */
AI_ARRAY_OBJ_DECLARE(
  nl_390_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#463 */
AI_ARRAY_OBJ_DECLARE(
  gemm_391_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#464 */
AI_ARRAY_OBJ_DECLARE(
  gemm_292_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#465 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_392_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#466 */
AI_ARRAY_OBJ_DECLARE(
  nl_393_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#467 */
AI_ARRAY_OBJ_DECLARE(
  gemm_394_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#468 */
AI_ARRAY_OBJ_DECLARE(
  gemm_293_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#469 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_395_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#470 */
AI_ARRAY_OBJ_DECLARE(
  nl_396_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#471 */
AI_ARRAY_OBJ_DECLARE(
  gemm_397_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#472 */
AI_ARRAY_OBJ_DECLARE(
  gemm_294_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#473 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_398_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#474 */
AI_ARRAY_OBJ_DECLARE(
  nl_399_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#475 */
AI_ARRAY_OBJ_DECLARE(
  gemm_400_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#476 */
AI_ARRAY_OBJ_DECLARE(
  gemm_295_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#477 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_401_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#478 */
AI_ARRAY_OBJ_DECLARE(
  nl_402_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#479 */
AI_ARRAY_OBJ_DECLARE(
  gemm_403_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#480 */
AI_ARRAY_OBJ_DECLARE(
  gemm_300_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#481 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_404_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#482 */
AI_ARRAY_OBJ_DECLARE(
  nl_405_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#483 */
AI_ARRAY_OBJ_DECLARE(
  gemm_406_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#484 */
AI_ARRAY_OBJ_DECLARE(
  gemm_301_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#485 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_407_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#486 */
AI_ARRAY_OBJ_DECLARE(
  nl_408_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#487 */
AI_ARRAY_OBJ_DECLARE(
  gemm_409_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#488 */
AI_ARRAY_OBJ_DECLARE(
  gemm_302_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#489 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_410_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#490 */
AI_ARRAY_OBJ_DECLARE(
  nl_411_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#491 */
AI_ARRAY_OBJ_DECLARE(
  gemm_412_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#492 */
AI_ARRAY_OBJ_DECLARE(
  gemm_303_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#493 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_413_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#494 */
AI_ARRAY_OBJ_DECLARE(
  nl_414_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#495 */
AI_ARRAY_OBJ_DECLARE(
  gemm_415_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#496 */
AI_ARRAY_OBJ_DECLARE(
  gemm_304_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#497 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_416_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#498 */
AI_ARRAY_OBJ_DECLARE(
  nl_417_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#499 */
AI_ARRAY_OBJ_DECLARE(
  gemm_418_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#500 */
AI_ARRAY_OBJ_DECLARE(
  gemm_305_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#501 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_419_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#502 */
AI_ARRAY_OBJ_DECLARE(
  nl_420_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#503 */
AI_ARRAY_OBJ_DECLARE(
  gemm_421_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#504 */
AI_ARRAY_OBJ_DECLARE(
  gemm_306_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#505 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_422_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#506 */
AI_ARRAY_OBJ_DECLARE(
  nl_423_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#507 */
AI_ARRAY_OBJ_DECLARE(
  gemm_424_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#508 */
AI_ARRAY_OBJ_DECLARE(
  gemm_307_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#509 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_425_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#510 */
AI_ARRAY_OBJ_DECLARE(
  nl_426_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#511 */
AI_ARRAY_OBJ_DECLARE(
  gemm_427_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#512 */
AI_ARRAY_OBJ_DECLARE(
  gemm_308_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#513 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_428_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#514 */
AI_ARRAY_OBJ_DECLARE(
  nl_429_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#515 */
AI_ARRAY_OBJ_DECLARE(
  gemm_430_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#516 */
AI_ARRAY_OBJ_DECLARE(
  gemm_309_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#517 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_431_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#518 */
AI_ARRAY_OBJ_DECLARE(
  nl_432_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#519 */
AI_ARRAY_OBJ_DECLARE(
  gemm_433_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#520 */
AI_ARRAY_OBJ_DECLARE(
  gemm_434_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#521 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_435_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#522 */
AI_ARRAY_OBJ_DECLARE(
  nl_436_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#523 */
AI_ARRAY_OBJ_DECLARE(
  gemm_437_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#524 */
AI_ARRAY_OBJ_DECLARE(
  gemm_438_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#525 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_439_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#526 */
AI_ARRAY_OBJ_DECLARE(
  nl_440_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#527 */
AI_ARRAY_OBJ_DECLARE(
  gemm_441_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#528 */
AI_ARRAY_OBJ_DECLARE(
  gemm_442_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#529 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_443_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#530 */
AI_ARRAY_OBJ_DECLARE(
  nl_444_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#531 */
AI_ARRAY_OBJ_DECLARE(
  gemm_445_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#532 */
AI_ARRAY_OBJ_DECLARE(
  gemm_446_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 24, AI_STATIC)

/* Array#533 */
AI_ARRAY_OBJ_DECLARE(
  constantofshape_5_const_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#534 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#535 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#536 */
AI_ARRAY_OBJ_DECLARE(
  constantofshape_251_const_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#537 */
AI_ARRAY_OBJ_DECLARE(
  gemm_253_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#538 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#539 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#540 */
AI_ARRAY_OBJ_DECLARE(
  gemm_254_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#541 */
AI_ARRAY_OBJ_DECLARE(
  gemm_254_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#542 */
AI_ARRAY_OBJ_DECLARE(
  gemm_445_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#543 */
AI_ARRAY_OBJ_DECLARE(
  gemm_445_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#544 */
AI_ARRAY_OBJ_DECLARE(
  gemm_446_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#545 */
AI_ARRAY_OBJ_DECLARE(
  gemm_446_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#546 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#547 */
AI_ARRAY_OBJ_DECLARE(
  gemm_253_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#548 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#549 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#550 */
AI_ARRAY_OBJ_DECLARE(
  gemm_22_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#551 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#552 */
AI_ARRAY_OBJ_DECLARE(
  gemm_37_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#553 */
AI_ARRAY_OBJ_DECLARE(
  gemm_41_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#554 */
AI_ARRAY_OBJ_DECLARE(
  gemm_52_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#555 */
AI_ARRAY_OBJ_DECLARE(
  gemm_56_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#556 */
AI_ARRAY_OBJ_DECLARE(
  gemm_67_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#557 */
AI_ARRAY_OBJ_DECLARE(
  gemm_71_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#558 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#559 */
AI_ARRAY_OBJ_DECLARE(
  gemm_75_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#560 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#561 */
AI_ARRAY_OBJ_DECLARE(
  gemm_79_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#562 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#563 */
AI_ARRAY_OBJ_DECLARE(
  gemm_83_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#564 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#565 */
AI_ARRAY_OBJ_DECLARE(
  gemm_87_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#566 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#567 */
AI_ARRAY_OBJ_DECLARE(
  gemm_91_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#568 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#569 */
AI_ARRAY_OBJ_DECLARE(
  gemm_95_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#570 */
AI_ARRAY_OBJ_DECLARE(
  gemm_18_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#571 */
AI_ARRAY_OBJ_DECLARE(
  gemm_99_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#572 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#573 */
AI_ARRAY_OBJ_DECLARE(
  gemm_103_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#574 */
AI_ARRAY_OBJ_DECLARE(
  gemm_20_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#575 */
AI_ARRAY_OBJ_DECLARE(
  gemm_107_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#576 */
AI_ARRAY_OBJ_DECLARE(
  gemm_21_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#577 */
AI_ARRAY_OBJ_DECLARE(
  gemm_111_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#578 */
AI_ARRAY_OBJ_DECLARE(
  gemm_27_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#579 */
AI_ARRAY_OBJ_DECLARE(
  gemm_115_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#580 */
AI_ARRAY_OBJ_DECLARE(
  gemm_28_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#581 */
AI_ARRAY_OBJ_DECLARE(
  gemm_119_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#582 */
AI_ARRAY_OBJ_DECLARE(
  gemm_29_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#583 */
AI_ARRAY_OBJ_DECLARE(
  gemm_123_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#584 */
AI_ARRAY_OBJ_DECLARE(
  gemm_30_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#585 */
AI_ARRAY_OBJ_DECLARE(
  gemm_127_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#586 */
AI_ARRAY_OBJ_DECLARE(
  gemm_31_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#587 */
AI_ARRAY_OBJ_DECLARE(
  gemm_131_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#588 */
AI_ARRAY_OBJ_DECLARE(
  gemm_32_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#589 */
AI_ARRAY_OBJ_DECLARE(
  gemm_135_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#590 */
AI_ARRAY_OBJ_DECLARE(
  gemm_33_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#591 */
AI_ARRAY_OBJ_DECLARE(
  gemm_139_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#592 */
AI_ARRAY_OBJ_DECLARE(
  gemm_34_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#593 */
AI_ARRAY_OBJ_DECLARE(
  gemm_143_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#594 */
AI_ARRAY_OBJ_DECLARE(
  gemm_35_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#595 */
AI_ARRAY_OBJ_DECLARE(
  gemm_147_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#596 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#597 */
AI_ARRAY_OBJ_DECLARE(
  gemm_151_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#598 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#599 */
AI_ARRAY_OBJ_DECLARE(
  gemm_155_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#600 */
AI_ARRAY_OBJ_DECLARE(
  gemm_43_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#601 */
AI_ARRAY_OBJ_DECLARE(
  gemm_159_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#602 */
AI_ARRAY_OBJ_DECLARE(
  gemm_44_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#603 */
AI_ARRAY_OBJ_DECLARE(
  gemm_163_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#604 */
AI_ARRAY_OBJ_DECLARE(
  gemm_45_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#605 */
AI_ARRAY_OBJ_DECLARE(
  gemm_167_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#606 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#607 */
AI_ARRAY_OBJ_DECLARE(
  gemm_171_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#608 */
AI_ARRAY_OBJ_DECLARE(
  gemm_47_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#609 */
AI_ARRAY_OBJ_DECLARE(
  gemm_175_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#610 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#611 */
AI_ARRAY_OBJ_DECLARE(
  gemm_179_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#612 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#613 */
AI_ARRAY_OBJ_DECLARE(
  gemm_183_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#614 */
AI_ARRAY_OBJ_DECLARE(
  gemm_50_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#615 */
AI_ARRAY_OBJ_DECLARE(
  gemm_187_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#616 */
AI_ARRAY_OBJ_DECLARE(
  gemm_51_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#617 */
AI_ARRAY_OBJ_DECLARE(
  gemm_191_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#618 */
AI_ARRAY_OBJ_DECLARE(
  gemm_57_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#619 */
AI_ARRAY_OBJ_DECLARE(
  gemm_195_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#620 */
AI_ARRAY_OBJ_DECLARE(
  gemm_58_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#621 */
AI_ARRAY_OBJ_DECLARE(
  gemm_199_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#622 */
AI_ARRAY_OBJ_DECLARE(
  gemm_59_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#623 */
AI_ARRAY_OBJ_DECLARE(
  gemm_203_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#624 */
AI_ARRAY_OBJ_DECLARE(
  gemm_60_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#625 */
AI_ARRAY_OBJ_DECLARE(
  gemm_207_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#626 */
AI_ARRAY_OBJ_DECLARE(
  gemm_61_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#627 */
AI_ARRAY_OBJ_DECLARE(
  gemm_211_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#628 */
AI_ARRAY_OBJ_DECLARE(
  gemm_62_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#629 */
AI_ARRAY_OBJ_DECLARE(
  gemm_215_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#630 */
AI_ARRAY_OBJ_DECLARE(
  gemm_63_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#631 */
AI_ARRAY_OBJ_DECLARE(
  gemm_219_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#632 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#633 */
AI_ARRAY_OBJ_DECLARE(
  gemm_223_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#634 */
AI_ARRAY_OBJ_DECLARE(
  gemm_65_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#635 */
AI_ARRAY_OBJ_DECLARE(
  gemm_227_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#636 */
AI_ARRAY_OBJ_DECLARE(
  gemm_66_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#637 */
AI_ARRAY_OBJ_DECLARE(
  gemm_231_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#638 */
AI_ARRAY_OBJ_DECLARE(
  gemm_232_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#639 */
AI_ARRAY_OBJ_DECLARE(
  gemm_236_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#640 */
AI_ARRAY_OBJ_DECLARE(
  gemm_237_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#641 */
AI_ARRAY_OBJ_DECLARE(
  gemm_241_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#642 */
AI_ARRAY_OBJ_DECLARE(
  gemm_242_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 334, AI_STATIC)

/* Array#643 */
AI_ARRAY_OBJ_DECLARE(
  gemm_254_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#644 */
AI_ARRAY_OBJ_DECLARE(
  gemm_257_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#645 */
AI_ARRAY_OBJ_DECLARE(
  gemm_268_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#646 */
AI_ARRAY_OBJ_DECLARE(
  gemm_271_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#647 */
AI_ARRAY_OBJ_DECLARE(
  gemm_282_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#648 */
AI_ARRAY_OBJ_DECLARE(
  gemm_285_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#649 */
AI_ARRAY_OBJ_DECLARE(
  gemm_296_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#650 */
AI_ARRAY_OBJ_DECLARE(
  gemm_299_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#651 */
AI_ARRAY_OBJ_DECLARE(
  gemm_310_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#652 */
AI_ARRAY_OBJ_DECLARE(
  gemm_313_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#653 */
AI_ARRAY_OBJ_DECLARE(
  gemm_258_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#654 */
AI_ARRAY_OBJ_DECLARE(
  gemm_316_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#655 */
AI_ARRAY_OBJ_DECLARE(
  gemm_259_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#656 */
AI_ARRAY_OBJ_DECLARE(
  gemm_319_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#657 */
AI_ARRAY_OBJ_DECLARE(
  gemm_260_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#658 */
AI_ARRAY_OBJ_DECLARE(
  gemm_322_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#659 */
AI_ARRAY_OBJ_DECLARE(
  gemm_261_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#660 */
AI_ARRAY_OBJ_DECLARE(
  gemm_325_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#661 */
AI_ARRAY_OBJ_DECLARE(
  gemm_262_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#662 */
AI_ARRAY_OBJ_DECLARE(
  gemm_328_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#663 */
AI_ARRAY_OBJ_DECLARE(
  gemm_263_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#664 */
AI_ARRAY_OBJ_DECLARE(
  gemm_331_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#665 */
AI_ARRAY_OBJ_DECLARE(
  gemm_264_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#666 */
AI_ARRAY_OBJ_DECLARE(
  gemm_334_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#667 */
AI_ARRAY_OBJ_DECLARE(
  gemm_265_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#668 */
AI_ARRAY_OBJ_DECLARE(
  gemm_337_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#669 */
AI_ARRAY_OBJ_DECLARE(
  gemm_266_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#670 */
AI_ARRAY_OBJ_DECLARE(
  gemm_340_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#671 */
AI_ARRAY_OBJ_DECLARE(
  gemm_267_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#672 */
AI_ARRAY_OBJ_DECLARE(
  gemm_343_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#673 */
AI_ARRAY_OBJ_DECLARE(
  gemm_272_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#674 */
AI_ARRAY_OBJ_DECLARE(
  gemm_346_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#675 */
AI_ARRAY_OBJ_DECLARE(
  gemm_273_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#676 */
AI_ARRAY_OBJ_DECLARE(
  gemm_349_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#677 */
AI_ARRAY_OBJ_DECLARE(
  gemm_274_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#678 */
AI_ARRAY_OBJ_DECLARE(
  gemm_352_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#679 */
AI_ARRAY_OBJ_DECLARE(
  gemm_275_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#680 */
AI_ARRAY_OBJ_DECLARE(
  gemm_355_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#681 */
AI_ARRAY_OBJ_DECLARE(
  gemm_276_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#682 */
AI_ARRAY_OBJ_DECLARE(
  gemm_358_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#683 */
AI_ARRAY_OBJ_DECLARE(
  gemm_277_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#684 */
AI_ARRAY_OBJ_DECLARE(
  gemm_361_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#685 */
AI_ARRAY_OBJ_DECLARE(
  gemm_278_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#686 */
AI_ARRAY_OBJ_DECLARE(
  gemm_364_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#687 */
AI_ARRAY_OBJ_DECLARE(
  gemm_279_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#688 */
AI_ARRAY_OBJ_DECLARE(
  gemm_367_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#689 */
AI_ARRAY_OBJ_DECLARE(
  gemm_280_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#690 */
AI_ARRAY_OBJ_DECLARE(
  gemm_370_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#691 */
AI_ARRAY_OBJ_DECLARE(
  gemm_281_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#692 */
AI_ARRAY_OBJ_DECLARE(
  gemm_373_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#693 */
AI_ARRAY_OBJ_DECLARE(
  gemm_286_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#694 */
AI_ARRAY_OBJ_DECLARE(
  gemm_376_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#695 */
AI_ARRAY_OBJ_DECLARE(
  gemm_287_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#696 */
AI_ARRAY_OBJ_DECLARE(
  gemm_379_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#697 */
AI_ARRAY_OBJ_DECLARE(
  gemm_288_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#698 */
AI_ARRAY_OBJ_DECLARE(
  gemm_382_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#699 */
AI_ARRAY_OBJ_DECLARE(
  gemm_289_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#700 */
AI_ARRAY_OBJ_DECLARE(
  gemm_385_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#701 */
AI_ARRAY_OBJ_DECLARE(
  gemm_290_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#702 */
AI_ARRAY_OBJ_DECLARE(
  gemm_388_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#703 */
AI_ARRAY_OBJ_DECLARE(
  gemm_291_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#704 */
AI_ARRAY_OBJ_DECLARE(
  gemm_391_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#705 */
AI_ARRAY_OBJ_DECLARE(
  gemm_292_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#706 */
AI_ARRAY_OBJ_DECLARE(
  gemm_394_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#707 */
AI_ARRAY_OBJ_DECLARE(
  gemm_293_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#708 */
AI_ARRAY_OBJ_DECLARE(
  gemm_397_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#709 */
AI_ARRAY_OBJ_DECLARE(
  gemm_294_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#710 */
AI_ARRAY_OBJ_DECLARE(
  gemm_400_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#711 */
AI_ARRAY_OBJ_DECLARE(
  gemm_295_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#712 */
AI_ARRAY_OBJ_DECLARE(
  gemm_403_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#713 */
AI_ARRAY_OBJ_DECLARE(
  gemm_300_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#714 */
AI_ARRAY_OBJ_DECLARE(
  gemm_406_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#715 */
AI_ARRAY_OBJ_DECLARE(
  gemm_301_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#716 */
AI_ARRAY_OBJ_DECLARE(
  gemm_409_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#717 */
AI_ARRAY_OBJ_DECLARE(
  gemm_302_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#718 */
AI_ARRAY_OBJ_DECLARE(
  gemm_412_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#719 */
AI_ARRAY_OBJ_DECLARE(
  gemm_303_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#720 */
AI_ARRAY_OBJ_DECLARE(
  gemm_415_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#721 */
AI_ARRAY_OBJ_DECLARE(
  gemm_304_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#722 */
AI_ARRAY_OBJ_DECLARE(
  gemm_418_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#723 */
AI_ARRAY_OBJ_DECLARE(
  gemm_305_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#724 */
AI_ARRAY_OBJ_DECLARE(
  gemm_421_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#725 */
AI_ARRAY_OBJ_DECLARE(
  gemm_306_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#726 */
AI_ARRAY_OBJ_DECLARE(
  gemm_424_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#727 */
AI_ARRAY_OBJ_DECLARE(
  gemm_307_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#728 */
AI_ARRAY_OBJ_DECLARE(
  gemm_427_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#729 */
AI_ARRAY_OBJ_DECLARE(
  gemm_308_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#730 */
AI_ARRAY_OBJ_DECLARE(
  gemm_430_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#731 */
AI_ARRAY_OBJ_DECLARE(
  gemm_309_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#732 */
AI_ARRAY_OBJ_DECLARE(
  gemm_433_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#733 */
AI_ARRAY_OBJ_DECLARE(
  gemm_434_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#734 */
AI_ARRAY_OBJ_DECLARE(
  gemm_437_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#735 */
AI_ARRAY_OBJ_DECLARE(
  gemm_438_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#736 */
AI_ARRAY_OBJ_DECLARE(
  gemm_441_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#737 */
AI_ARRAY_OBJ_DECLARE(
  gemm_442_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#738 */
AI_ARRAY_OBJ_DECLARE(
  gemm_445_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#739 */
AI_ARRAY_OBJ_DECLARE(
  gemm_446_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 184, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(constantofshape_251_const_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.84313680668447e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(constantofshape_5_const_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.84313680668447e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_102_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_106_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_110_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_114_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_118_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_122_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_126_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_130_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_134_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_138_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_142_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_146_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_150_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_154_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_158_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_162_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_166_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_170_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_174_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_178_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_182_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_186_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_190_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_194_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_198_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_202_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_206_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_210_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_214_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_218_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_222_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_226_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_230_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_235_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_240_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_245_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_25_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_40_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_55_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_70_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_74_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_78_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_82_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_86_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_90_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_94_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_98_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_100_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039955392479896545f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_104_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04001697897911072f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_108_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04007373005151749f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_112_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03987129032611847f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_116_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03978469967842102f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_120_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03972034156322479f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_124_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039772819727659225f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_128_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03984823077917099f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_132_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03991046920418739f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_136_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03993769362568855f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_140_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03993692994117737f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_144_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0399264320731163f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_148_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03993207961320877f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_152_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03994743898510933f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_156_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03996008262038231f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_160_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03994635492563248f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_164_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03987394645810127f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_168_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03987746685743332f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_172_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03987327218055725f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_176_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03986399248242378f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_180_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03983968123793602f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_184_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03982730954885483f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_188_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039821140468120575f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_192_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039817925542593f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_196_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981635719537735f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_200_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981561213731766f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_204_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981541469693184f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_208_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981555253267288f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_212_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981596231460571f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_216_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981602191925049f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_220_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981608897447586f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_224_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981618583202362f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_228_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981621935963631f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_233_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981621563434601f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_238_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039816223084926605f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_23_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03152075782418251f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_243_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981621935963631f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_255_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03238719329237938f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_269_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04620243236422539f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_283_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0505773089826107f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_297_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05990210920572281f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_311_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06205454841256142f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #92 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_314_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06112903729081154f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #93 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_317_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05966242030262947f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #94 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_320_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05915628373622894f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #95 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_323_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05746467411518097f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #96 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_326_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05694771185517311f),
    AI_PACK_INTQ_ZP(11)))

/* Int quant #97 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_329_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05794944241642952f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #98 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_332_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05784677341580391f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #99 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_335_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05781181901693344f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #100 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_338_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05777494236826897f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #101 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_341_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05775825306773186f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #102 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_344_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05778703838586807f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #103 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_347_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05780813843011856f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #104 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_350_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.057832200080156326f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #105 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_353_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0577940009534359f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #106 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_356_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05739254876971245f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #107 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_359_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.057389210909605026f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #108 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_362_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05738933011889458f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #109 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_365_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05739051476120949f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #110 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_368_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05782401189208031f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #111 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_371_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0585014671087265f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #112 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_374_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05977470427751541f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #113 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_377_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.061436377465724945f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #114 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_380_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06175687536597252f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #115 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_383_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.061826106160879135f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #116 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_386_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06182566657662392f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #117 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_389_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.061826277524232864f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #118 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_38_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0371326208114624f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #119 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_392_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060994066298007965f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #120 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_395_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06054957956075668f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #121 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_398_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052622199058533f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #122 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_401_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060526106506586075f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #123 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_404_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052616983652115f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #124 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_407_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052633374929428f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #125 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_410_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052646040916443f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #126 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_413_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060526326298713684f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #127 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_416_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060525648295879364f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #128 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_419_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052567437291145f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #129 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_422_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052563339471817f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #130 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_425_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060525551438331604f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #131 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_428_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843152999878f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #132 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_431_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843525528908f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #133 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_435_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843525528908f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #134 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_439_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843525528908f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #135 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_443_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843525528908f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #136 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_53_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03791920468211174f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #137 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_68_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.037519752979278564f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #138 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_72_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039380960166454315f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #139 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_76_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04149246960878372f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #140 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_80_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04165453836321831f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #141 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_84_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04044939577579498f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #142 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_88_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03976702690124512f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #143 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_8_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02678738348186016f),
    AI_PACK_INTQ_ZP(20)))

/* Int quant #144 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_92_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03956305980682373f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #145 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_96_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03957647457718849f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #146 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_103_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0283010583370924f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #147 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_107_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02823272906243801f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #148 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_111_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028181064873933792f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #149 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_115_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028150789439678192f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #150 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_119_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028139233589172363f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #151 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017456693574786186f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #152 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_123_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028142599388957024f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #153 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_127_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028137512505054474f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #154 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_12_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039380960166454315f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #155 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_131_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02813105471432209f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #156 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_135_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028121519833803177f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #157 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_139_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02811136096715927f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #158 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04149246960878372f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #159 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_143_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02810332365334034f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #160 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_147_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028098106384277344f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #161 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04165453836321831f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #162 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_151_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02809595689177513f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #163 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_155_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02809719182550907f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #164 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_159_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02809906378388405f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #165 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_15_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04044939577579498f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #166 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_163_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02809874899685383f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #167 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_167_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028096521273255348f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #168 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_16_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03976702690124512f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #169 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_171_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02809411659836769f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #170 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_175_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028094707056879997f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #171 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_179_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028096789494156837f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #172 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_17_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03956305980682373f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #173 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_183_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028097817674279213f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #174 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_187_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028096606954932213f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #175 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03957647457718849f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #176 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_191_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028095727786421776f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #177 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_195_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02809545397758484f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #178 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_199_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028095971792936325f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #179 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_19_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039955392479896545f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #180 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_203_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028096001595258713f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #181 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_207_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028095798566937447f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #182 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_20_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04001697897911072f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #183 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_211_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02809571661055088f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #184 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_215_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028095854446291924f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #185 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_219_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028095973655581474f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #186 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_21_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04007373005151749f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #187 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_223_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028096020221710205f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #188 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_227_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028096020221710205f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #189 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_22_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03152075782418251f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #190 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_231_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02783118188381195f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #191 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_232_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981621563434601f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #192 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_236_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027831198647618294f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #193 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_237_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039816223084926605f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #194 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_241_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02783120609819889f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #195 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_242_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981621935963631f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #196 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_253_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.84313680668447e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #197 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_253_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003187361406162381f, 0.0024567025247961283f, 0.0027069668285548687f, 0.003134588012471795f, 0.0024502717424184084f, 0.002396981930360198f, 0.003380867186933756f, 0.0027817264199256897f, 0.002996000461280346f, 0.0030087728518992662f, 0.0026873459573835135f, 0.002458013128489256f, 0.0030895541422069073f, 0.0029484382830560207f, 0.0025442412588745356f, 0.0035743187181651592f, 0.003400063142180443f, 0.0028292147908359766f, 0.002653735224157572f, 0.002511409344151616f, 0.002243685768917203f, 0.0029110496398061514f, 0.0030313211027532816f, 0.0026328943204134703f, 0.0034708762541413307f, 0.002932195086032152f, 0.0034213601611554623f, 0.003843773854896426f, 0.0024366185534745455f, 0.0028148989658802748f, 0.0028191725723445415f, 0.0030153561383485794f, 0.0031064245849847794f, 0.002509078476577997f, 0.003023486351594329f, 0.003370364662259817f, 0.002672859700396657f, 0.003372891340404749f, 0.0038607388269156218f, 0.0041499813087284565f, 0.0024972930550575256f, 0.002843686379492283f, 0.003038299735635519f, 0.0027554091066122055f, 0.002615749603137374f, 0.0029110335744917393f, 0.0030137524008750916f, 0.0030811254400759935f, 0.003165791044011712f, 0.002518212189897895f, 0.002814434701576829f, 0.0029102598782628775f, 0.003054634667932987f, 0.003130345605313778f, 0.0026620293501764536f, 0.003137109335511923f, 0.0027825222350656986f, 0.002764603588730097f, 0.0028209732845425606f, 0.002681892132386565f, 0.004857912659645081f, 0.003285246202722192f, 0.003000800032168627f, 0.004512525629252195f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #198 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_254_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03238719329237938f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #199 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_254_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017057484947144985f, 0.0024805504363030195f, 0.001734415185637772f, 0.0015828057657927275f, 0.0019604486878961325f, 0.0019368890207260847f, 0.0019542586524039507f, 0.0021452114451676607f, 0.00228892732411623f, 0.0023566726595163345f, 0.0032169027253985405f, 0.0018544943304732442f, 0.001984064932912588f, 0.0018634559819474816f, 0.004117949865758419f, 0.0017258385196328163f, 0.001864640973508358f, 0.001694085425697267f, 0.0019242948619648814f, 0.0018089102813974023f, 0.0018876917893067002f, 0.0017467164434492588f, 0.0017396481707692146f, 0.0017196324188262224f, 0.0019751612562686205f, 0.0018798204837366939f, 0.003174328478053212f, 0.003310407279059291f, 0.0023057544603943825f, 0.002263846807181835f, 0.0019613925833255053f, 0.0017248023068532348f, 0.0017968431347981095f, 0.002186508383601904f, 0.0019656617660075426f, 0.0032871023286134005f, 0.0024557390715926886f, 0.0030758476350456476f, 0.002038600854575634f, 0.003955832216888666f, 0.0020550889894366264f, 0.002138375537469983f, 0.0020099410321563482f, 0.0020707142539322376f, 0.0018241291400045156f, 0.001771741546690464f, 0.0018600600305944681f, 0.005479085259139538f, 0.003685044590383768f, 0.002196230459958315f, 0.001806905260309577f, 0.002037396654486656f, 0.0016263622092083097f, 0.0037565373349934816f, 0.0018892597872763872f, 0.0017876235069707036f, 0.0017632368253543973f, 0.0020123482681810856f, 0.0020784223452210426f, 0.0020199758000671864f, 0.0039251153357326984f, 0.0016566229751333594f, 0.0020616149995476007f, 0.004418596159666777f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #200 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_257_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020396072417497635f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #201 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_258_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06112903729081154f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #202 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_259_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05966242030262947f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #203 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_260_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05915628373622894f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #204 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_261_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05746467411518097f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #205 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_262_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05694771185517311f),
    AI_PACK_INTQ_ZP(11)))

/* Int quant #206 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_263_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05794944241642952f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #207 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_264_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05784677341580391f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #208 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_265_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05781181901693344f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #209 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_266_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05777494236826897f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #210 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_267_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05775825306773186f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #211 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_268_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04620243236422539f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #212 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_26_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.026239117607474327f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #213 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_271_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04455127939581871f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #214 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_272_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05778703838586807f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #215 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_273_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05780813843011856f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #216 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_274_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.057832200080156326f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #217 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_275_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0577940009534359f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #218 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_276_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05739254876971245f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #219 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_277_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.057389210909605026f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #220 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_278_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05738933011889458f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #221 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_279_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05739051476120949f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #222 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_27_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03987129032611847f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #223 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_280_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05782401189208031f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #224 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_281_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0585014671087265f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #225 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_282_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0505773089826107f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #226 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_285_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0483611598610878f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #227 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_286_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05977470427751541f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #228 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_287_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.061436377465724945f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #229 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_288_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06175687536597252f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #230 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_289_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.061826106160879135f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #231 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_28_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03978469967842102f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #232 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_290_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06182566657662392f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #233 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_291_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.061826277524232864f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #234 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_292_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060994066298007965f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #235 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_293_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06054957956075668f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #236 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_294_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052622199058533f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #237 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_295_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060526106506586075f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #238 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_296_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05990210920572281f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #239 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_299_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04967601224780083f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #240 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_29_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03972034156322479f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #241 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_300_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052616983652115f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #242 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_301_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052633374929428f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #243 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_302_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052646040916443f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #244 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_303_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060526326298713684f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #245 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_304_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060525648295879364f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #246 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_305_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052567437291145f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #247 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_306_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06052563339471817f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #248 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_307_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.060525551438331604f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #249 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_308_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843152999878f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #250 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_309_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843525528908f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #251 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_30_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039772819727659225f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #252 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_310_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06205454841256142f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #253 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_313_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049506042152643204f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #254 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_316_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04923376068472862f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #255 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_319_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0492587611079216f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #256 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_31_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03984823077917099f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #257 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_322_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04927799105644226f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #258 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_325_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928703233599663f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #259 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_328_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049286238849163055f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #260 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_32_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03991046920418739f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #261 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_331_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049287039786577225f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #262 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_334_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928737133741379f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #263 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_337_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928788170218468f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #264 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_33_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03993769362568855f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #265 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_340_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049288470298051834f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #266 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_343_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928828030824661f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #267 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_346_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049288321286439896f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #268 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_349_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049288060516119f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #269 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_34_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03993692994117737f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #270 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_352_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049287594854831696f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #271 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_355_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928728938102722f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #272 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_358_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928718879818916f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #273 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_35_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0399264320731163f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #274 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_361_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928716644644737f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #275 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_364_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928715527057648f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #276 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_367_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049242448061704636f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #277 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_36_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03993207961320877f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #278 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_370_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04924234375357628f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #279 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_373_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04921567812561989f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #280 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_376_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04917886108160019f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #281 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_379_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049094267189502716f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #282 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_37_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0371326208114624f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #283 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_382_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04900075122714043f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #284 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_385_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04884793981909752f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #285 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_388_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04875694960355759f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #286 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_391_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04859311506152153f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #287 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_394_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04831907898187637f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #288 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_397_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04803897440433502f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #289 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_400_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04804135859012604f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #290 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_403_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04804238677024841f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #291 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_406_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04804253205657005f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #292 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_409_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04825957119464874f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #293 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_412_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048460524529218674f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #294 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_415_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048681557178497314f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #295 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_418_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04894368723034859f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #296 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_41_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02960411086678505f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #297 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_421_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04912898689508438f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #298 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_424_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049256373196840286f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #299 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_427_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928712919354439f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #300 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_42_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03994743898510933f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #301 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_430_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049287136644124985f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #302 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_433_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049287136644124985f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #303 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_434_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843525528908f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #304 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_437_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049287136644124985f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #305 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_438_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843525528908f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #306 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_43_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03996008262038231f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #307 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_441_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049287136644124985f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #308 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_442_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06015843525528908f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #309 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_445_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04082246124744415f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #310 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_445_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0028351102955639362f, 0.004207029473036528f, 0.0026778678875416517f, 0.007806582376360893f, 0.012614795006811619f, 0.0039825052954256535f, 0.0032406921964138746f, 0.004581972490996122f, 0.0036676404997706413f, 0.003909907303750515f, 0.005728460382670164f, 0.002857112791389227f, 0.0025241354014724493f, 0.0026868400163948536f, 0.0038446150720119476f, 0.004360190127044916f, 0.005078778602182865f, 0.0038581821136176586f, 0.004792279098182917f, 0.005696242209523916f, 0.004273212049156427f, 0.0040998454205691814f, 0.003268496599048376f, 0.0031859958544373512f, 0.004015609622001648f, 0.0029618002008646727f, 0.003117199521511793f, 0.00258365529589355f, 0.003988807089626789f, 0.006349598988890648f, 0.004583791829645634f, 0.004092622548341751f, 0.00285618519410491f, 0.0038385142106562853f, 0.003680119290947914f, 0.00411917082965374f, 0.003290034830570221f, 0.007800028193742037f, 0.00339802959933877f, 0.0030246421229094267f, 0.0036374197807163f, 0.003245255909860134f, 0.0027833750937134027f, 0.003172796219587326f, 0.004674451891332865f, 0.010091546922922134f, 0.0028570496942847967f, 0.003516240045428276f, 0.003268516156822443f, 0.0026605245657265186f, 0.002788200043141842f, 0.002552902791649103f, 0.009418003261089325f, 0.0029603654984384775f, 0.005871579982340336f, 0.00451451912522316f, 0.004068037495017052f, 0.003530818037688732f, 0.00457923486828804f, 0.003277554176747799f, 0.00256025861017406f, 0.004025852773338556f, 0.0027829192113131285f, 0.003648184472694993f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #311 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_446_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14647063612937927f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #312 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_446_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009005625732243061f, 0.007717636413872242f, 0.007756117731332779f, 0.006887555122375488f, 0.00636766804382205f, 0.005460585467517376f, 0.0041117239743471146f, 0.005113943479955196f, 0.004736186470836401f, 0.007185997441411018f, 0.005971736274659634f, 0.006036791019141674f, 0.006068162154406309f, 0.006483448203653097f, 0.005990422796458006f, 0.0071770683862268925f, 0.007897595874965191f, 0.005737804342061281f, 0.006421694997698069f, 0.006900972221046686f, 0.006390801630914211f, 0.00669889897108078f, 0.005595899652689695f, 0.0056700254790484905f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #313 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_44_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03994635492563248f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #314 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_45_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03987394645810127f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #315 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_46_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03987746685743332f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #316 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_47_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03987327218055725f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #317 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_48_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03986399248242378f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #318 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_49_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03983968123793602f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #319 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_50_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03982730954885483f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #320 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_51_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039821140468120575f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #321 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_52_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03791920468211174f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #322 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_56_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029199250042438507f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #323 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_57_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039817925542593f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #324 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_58_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981635719537735f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #325 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_59_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981561213731766f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #326 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_60_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981541469693184f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #327 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_61_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981555253267288f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #328 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_62_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981596231460571f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #329 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_63_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981602191925049f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #330 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_64_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981608897447586f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #331 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_65_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981618583202362f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #332 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_66_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03981621935963631f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #333 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_67_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.037519752979278564f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #334 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_6_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.84313680668447e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #335 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_6_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002684053499251604f, 0.0033634223509579897f, 0.002852396108210087f, 0.003410150296986103f, 0.002313402947038412f, 0.003543325699865818f, 0.0021860015112906694f, 0.0027173245325684547f, 0.002748399041593075f, 0.0026256837882101536f, 0.0024250857532024384f, 0.002878837753087282f, 0.0028642104007303715f, 0.0027120267041027546f, 0.002576041966676712f, 0.0040409499779343605f, 0.0028319410048425198f, 0.0036550257354974747f, 0.002852162579074502f, 0.002798657864332199f, 0.002707861131057143f, 0.0027408795431256294f, 0.002538013970479369f, 0.0039015961810946465f, 0.0021818859968334436f, 0.003838414791971445f, 0.002332766307517886f, 0.0029538783710449934f, 0.005032298155128956f, 0.0021231581922620535f, 0.0027878449764102697f, 0.003023292636498809f, 0.0029409355483949184f, 0.0021875135134905577f, 0.0032609242480248213f, 0.0022455581929534674f, 0.002834347542375326f, 0.0026335804723203182f, 0.0024941430892795324f, 0.002734585665166378f, 0.002376043237745762f, 0.0026949800085276365f, 0.0026488094590604305f, 0.0026238008867949247f, 0.0030605692882090807f, 0.0025041713379323483f, 0.0026125474832952023f, 0.0027187266387045383f, 0.0030007322784513235f, 0.0023436560295522213f, 0.002893011551350355f, 0.0027936988044530153f, 0.0034564712550491095f, 0.003321335418149829f, 0.002162178745493293f, 0.0023144015576690435f, 0.002205462660640478f, 0.002713931491598487f, 0.0024610820692032576f, 0.002680090954527259f, 0.003327122190967202f, 0.0028530729468911886f, 0.0023881285451352596f, 0.0025351555086672306f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #336 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_71_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02866779826581478f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #337 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_75_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029236262664198875f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #338 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_79_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02870718203485012f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #339 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02678738348186016f),
    AI_PACK_INTQ_ZP(20)))

/* Int quant #340 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_7_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0031847828067839146f, 0.0025500566698610783f, 0.0018348906887695193f, 0.002076149685308337f, 0.0024165953509509563f, 0.0030774374026805162f, 0.0019978040363639593f, 0.0026880260556936264f, 0.0028723350260406733f, 0.001828155480325222f, 0.001994994468986988f, 0.003265181090682745f, 0.002832979429513216f, 0.0021320651285350323f, 0.0019397832220420241f, 0.002283936133608222f, 0.0034051977563649416f, 0.003045987104997039f, 0.003108780598267913f, 0.0021710374858230352f, 0.0027694357559084892f, 0.0028192067984491587f, 0.002167430240660906f, 0.0038375307340174913f, 0.0020215539261698723f, 0.0037857573479413986f, 0.001901164068840444f, 0.002648396650329232f, 0.0026190888602286577f, 0.002724104095250368f, 0.0020998630207031965f, 0.0023218640126287937f, 0.00201708753593266f, 0.002023314358666539f, 0.0028946506790816784f, 0.0020550922490656376f, 0.002198736649006605f, 0.0017349842237308621f, 0.002087291097268462f, 0.001570865511894226f, 0.0027904489543288946f, 0.0021452095825225115f, 0.0019293664954602718f, 0.0024731343146413565f, 0.001989704789593816f, 0.0032468175049871206f, 0.0024410029873251915f, 0.002285014372318983f, 0.002315678633749485f, 0.00200580689124763f, 0.0021046032197773457f, 0.0028361426666378975f, 0.0020596925169229507f, 0.002131967106834054f, 0.0018035289831459522f, 0.002493753330782056f, 0.001972159370779991f, 0.002507682191208005f, 0.0034484213683754206f, 0.002001682762056589f, 0.003057356458157301f, 0.002236950444057584f, 0.002577692735940218f, 0.0023549559991806746f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #341 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_83_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028481490910053253f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #342 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_87_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028230801224708557f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #343 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_91_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02828642539680004f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #344 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_95_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028316227719187737f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #345 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_99_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028305552899837494f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #346 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_101_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #347 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_105_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #348 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_109_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #349 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_113_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #350 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_117_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #351 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_121_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #352 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_125_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #353 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_129_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #354 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_133_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #355 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_137_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #356 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_141_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #357 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_145_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #358 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_149_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #359 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_153_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #360 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_157_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #361 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_161_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #362 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_165_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #363 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_169_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #364 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_173_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #365 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_177_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #366 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_181_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #367 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_185_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #368 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_189_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #369 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_193_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #370 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_197_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #371 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_201_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #372 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_205_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #373 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_209_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #374 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_213_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #375 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_217_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #376 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_221_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #377 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_225_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #378 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_229_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #379 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_234_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #380 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_239_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #381 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_244_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #382 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_24_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #383 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_256_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #384 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_270_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #385 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_284_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #386 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_298_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #387 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_312_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #388 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_315_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #389 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_318_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #390 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_321_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #391 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_324_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #392 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_327_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #393 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_330_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #394 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_333_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #395 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_336_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #396 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_339_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #397 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_342_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #398 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_345_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #399 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_348_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #400 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_351_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #401 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_354_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #402 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_357_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #403 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_360_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #404 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_363_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #405 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_366_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #406 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_369_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #407 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_372_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #408 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_375_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #409 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_378_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #410 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_381_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #411 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_384_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #412 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_387_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #413 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_390_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #414 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_393_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #415 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_396_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #416 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_399_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #417 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_39_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #418 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_402_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #419 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_405_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #420 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_408_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #421 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_411_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #422 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_414_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #423 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_417_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #424 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_420_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #425 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_423_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #426 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_426_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #427 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_429_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #428 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_432_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #429 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_436_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #430 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_440_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #431 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_444_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #432 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_54_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #433 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_69_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #434 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_73_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #435 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_77_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #436 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_81_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #437 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_85_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #438 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_89_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #439 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_93_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #440 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_97_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #441 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_9_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #442 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pack_246_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #443 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_keras_tensor_60_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #444 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #445 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output0_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #446 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #447 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output10_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #448 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output11_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #449 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output12_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #450 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output13_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #451 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output14_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #452 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output15_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #453 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output16_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #454 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output17_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #455 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output18_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #456 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output19_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #457 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output2_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #458 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output20_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #459 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output21_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #460 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output22_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #461 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output23_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #462 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output24_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #463 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output25_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #464 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output26_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #465 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output27_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #466 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output28_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #467 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output29_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #468 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output3_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #469 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output30_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #470 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output31_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #471 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output32_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #472 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output33_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #473 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output34_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #474 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output35_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #475 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output36_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #476 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output37_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #477 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output38_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #478 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output39_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #479 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output4_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #480 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output40_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #481 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output41_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #482 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output42_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #483 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output43_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #484 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output44_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #485 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output45_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #486 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output46_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #487 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output47_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #488 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output5_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #489 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output6_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #490 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output7_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #491 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output8_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #492 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_252_output9_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842672988772392f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #493 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output0_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #494 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #495 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output10_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #496 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output11_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #497 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output12_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #498 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output13_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #499 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output14_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #500 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output15_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #501 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output16_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #502 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output17_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #503 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output18_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #504 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output19_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #505 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output2_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #506 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output20_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #507 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output21_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #508 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output22_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #509 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output23_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #510 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output24_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #511 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output25_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #512 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output26_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #513 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output27_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #514 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output28_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #515 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output29_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #516 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output3_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #517 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output30_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #518 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output31_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #519 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output32_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #520 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output33_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #521 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output34_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #522 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output35_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #523 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output36_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #524 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output37_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #525 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output38_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #526 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output39_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #527 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output4_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #528 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output40_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #529 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output41_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #530 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output42_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #531 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output43_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #532 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output44_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #533 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output45_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #534 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output46_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #535 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output47_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #536 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output5_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #537 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output6_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #538 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output7_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #539 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output8_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #540 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_3_output9_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030718522146344185f),
    AI_PACK_INTQ_ZP(14)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  constantofshape_251_const, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &constantofshape_251_const_array, &constantofshape_251_const_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  constantofshape_5_const, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &constantofshape_5_const_array, &constantofshape_5_const_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conversion_102_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_102_output_array, &conversion_102_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conversion_106_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_106_output_array, &conversion_106_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conversion_10_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_10_output_array, &conversion_10_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conversion_110_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_110_output_array, &conversion_110_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conversion_114_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_114_output_array, &conversion_114_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conversion_118_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_118_output_array, &conversion_118_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conversion_122_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_122_output_array, &conversion_122_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conversion_126_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_126_output_array, &conversion_126_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conversion_130_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_130_output_array, &conversion_130_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conversion_134_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_134_output_array, &conversion_134_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conversion_138_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_138_output_array, &conversion_138_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conversion_142_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_142_output_array, &conversion_142_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conversion_146_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_146_output_array, &conversion_146_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conversion_150_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_150_output_array, &conversion_150_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conversion_154_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_154_output_array, &conversion_154_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conversion_158_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_158_output_array, &conversion_158_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conversion_162_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_162_output_array, &conversion_162_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conversion_166_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_166_output_array, &conversion_166_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conversion_170_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_170_output_array, &conversion_170_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conversion_174_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_174_output_array, &conversion_174_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conversion_178_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_178_output_array, &conversion_178_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conversion_182_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_182_output_array, &conversion_182_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conversion_186_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_186_output_array, &conversion_186_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conversion_190_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_190_output_array, &conversion_190_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conversion_194_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_194_output_array, &conversion_194_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conversion_198_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_198_output_array, &conversion_198_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conversion_202_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_202_output_array, &conversion_202_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conversion_206_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_206_output_array, &conversion_206_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conversion_210_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_210_output_array, &conversion_210_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conversion_214_output, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_214_output_array, &conversion_214_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conversion_218_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_218_output_array, &conversion_218_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conversion_222_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_222_output_array, &conversion_222_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conversion_226_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_226_output_array, &conversion_226_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conversion_230_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_230_output_array, &conversion_230_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conversion_235_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_235_output_array, &conversion_235_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conversion_240_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_240_output_array, &conversion_240_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conversion_245_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_245_output_array, &conversion_245_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conversion_25_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_25_output_array, &conversion_25_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conversion_40_output, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_40_output_array, &conversion_40_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conversion_55_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_55_output_array, &conversion_55_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conversion_70_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_70_output_array, &conversion_70_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conversion_74_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_74_output_array, &conversion_74_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conversion_78_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_78_output_array, &conversion_78_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conversion_82_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_82_output_array, &conversion_82_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conversion_86_output, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_86_output_array, &conversion_86_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conversion_90_output, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_90_output_array, &conversion_90_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conversion_94_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_94_output_array, &conversion_94_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conversion_98_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conversion_98_output_array, &conversion_98_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_100_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_100_output_array, &eltwise_100_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_104_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_104_output_array, &eltwise_104_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_108_output, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_108_output_array, &eltwise_108_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_112_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_112_output_array, &eltwise_112_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_116_output, AI_STATIC,
  54, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_116_output_array, &eltwise_116_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_120_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_120_output_array, &eltwise_120_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_124_output, AI_STATIC,
  56, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_124_output_array, &eltwise_124_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_128_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_128_output_array, &eltwise_128_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_132_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_132_output_array, &eltwise_132_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_136_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_136_output_array, &eltwise_136_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_140_output, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_140_output_array, &eltwise_140_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_144_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_144_output_array, &eltwise_144_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_148_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_148_output_array, &eltwise_148_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_152_output, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_152_output_array, &eltwise_152_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_156_output, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_156_output_array, &eltwise_156_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_160_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_160_output_array, &eltwise_160_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_164_output, AI_STATIC,
  66, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_164_output_array, &eltwise_164_output_array_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_168_output, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_168_output_array, &eltwise_168_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_172_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_172_output_array, &eltwise_172_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_176_output, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_176_output_array, &eltwise_176_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_180_output, AI_STATIC,
  70, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_180_output_array, &eltwise_180_output_array_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_184_output, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_184_output_array, &eltwise_184_output_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_188_output, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_188_output_array, &eltwise_188_output_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_192_output, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_192_output_array, &eltwise_192_output_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_196_output, AI_STATIC,
  74, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_196_output_array, &eltwise_196_output_array_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_200_output, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_200_output_array, &eltwise_200_output_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_204_output, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_204_output_array, &eltwise_204_output_array_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_208_output, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_208_output_array, &eltwise_208_output_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_212_output, AI_STATIC,
  78, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_212_output_array, &eltwise_212_output_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_216_output, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_216_output_array, &eltwise_216_output_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_220_output, AI_STATIC,
  80, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_220_output_array, &eltwise_220_output_array_intq)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_224_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_224_output_array, &eltwise_224_output_array_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_228_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_228_output_array, &eltwise_228_output_array_intq)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_233_output, AI_STATIC,
  83, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_233_output_array, &eltwise_233_output_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_238_output, AI_STATIC,
  84, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_238_output_array, &eltwise_238_output_array_intq)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_23_output, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_23_output_array, &eltwise_23_output_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_243_output, AI_STATIC,
  86, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_243_output_array, &eltwise_243_output_array_intq)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_255_output, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_255_output_array, &eltwise_255_output_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_269_output, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_269_output_array, &eltwise_269_output_array_intq)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_283_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_283_output_array, &eltwise_283_output_array_intq)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_297_output, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_297_output_array, &eltwise_297_output_array_intq)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_311_output, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_311_output_array, &eltwise_311_output_array_intq)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_314_output, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_314_output_array, &eltwise_314_output_array_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_317_output, AI_STATIC,
  93, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_317_output_array, &eltwise_317_output_array_intq)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_320_output, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_320_output_array, &eltwise_320_output_array_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_323_output, AI_STATIC,
  95, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_323_output_array, &eltwise_323_output_array_intq)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_326_output, AI_STATIC,
  96, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_326_output_array, &eltwise_326_output_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_329_output, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_329_output_array, &eltwise_329_output_array_intq)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_332_output, AI_STATIC,
  98, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_332_output_array, &eltwise_332_output_array_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_335_output, AI_STATIC,
  99, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_335_output_array, &eltwise_335_output_array_intq)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_338_output, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_338_output_array, &eltwise_338_output_array_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_341_output, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_341_output_array, &eltwise_341_output_array_intq)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_344_output, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_344_output_array, &eltwise_344_output_array_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_347_output, AI_STATIC,
  103, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_347_output_array, &eltwise_347_output_array_intq)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_350_output, AI_STATIC,
  104, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_350_output_array, &eltwise_350_output_array_intq)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_353_output, AI_STATIC,
  105, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_353_output_array, &eltwise_353_output_array_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_356_output, AI_STATIC,
  106, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_356_output_array, &eltwise_356_output_array_intq)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_359_output, AI_STATIC,
  107, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_359_output_array, &eltwise_359_output_array_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_362_output, AI_STATIC,
  108, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_362_output_array, &eltwise_362_output_array_intq)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_365_output, AI_STATIC,
  109, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_365_output_array, &eltwise_365_output_array_intq)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_368_output, AI_STATIC,
  110, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_368_output_array, &eltwise_368_output_array_intq)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_371_output, AI_STATIC,
  111, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_371_output_array, &eltwise_371_output_array_intq)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_374_output, AI_STATIC,
  112, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_374_output_array, &eltwise_374_output_array_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_377_output, AI_STATIC,
  113, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_377_output_array, &eltwise_377_output_array_intq)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_380_output, AI_STATIC,
  114, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_380_output_array, &eltwise_380_output_array_intq)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_383_output, AI_STATIC,
  115, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_383_output_array, &eltwise_383_output_array_intq)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_386_output, AI_STATIC,
  116, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_386_output_array, &eltwise_386_output_array_intq)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_389_output, AI_STATIC,
  117, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_389_output_array, &eltwise_389_output_array_intq)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_38_output, AI_STATIC,
  118, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_38_output_array, &eltwise_38_output_array_intq)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_392_output, AI_STATIC,
  119, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_392_output_array, &eltwise_392_output_array_intq)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_395_output, AI_STATIC,
  120, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_395_output_array, &eltwise_395_output_array_intq)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_398_output, AI_STATIC,
  121, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_398_output_array, &eltwise_398_output_array_intq)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_401_output, AI_STATIC,
  122, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_401_output_array, &eltwise_401_output_array_intq)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_404_output, AI_STATIC,
  123, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_404_output_array, &eltwise_404_output_array_intq)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_407_output, AI_STATIC,
  124, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_407_output_array, &eltwise_407_output_array_intq)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_410_output, AI_STATIC,
  125, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_410_output_array, &eltwise_410_output_array_intq)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_413_output, AI_STATIC,
  126, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_413_output_array, &eltwise_413_output_array_intq)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_416_output, AI_STATIC,
  127, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_416_output_array, &eltwise_416_output_array_intq)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_419_output, AI_STATIC,
  128, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_419_output_array, &eltwise_419_output_array_intq)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_422_output, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_422_output_array, &eltwise_422_output_array_intq)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_425_output, AI_STATIC,
  130, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_425_output_array, &eltwise_425_output_array_intq)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_428_output, AI_STATIC,
  131, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_428_output_array, &eltwise_428_output_array_intq)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_431_output, AI_STATIC,
  132, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_431_output_array, &eltwise_431_output_array_intq)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_435_output, AI_STATIC,
  133, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_435_output_array, &eltwise_435_output_array_intq)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_439_output, AI_STATIC,
  134, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_439_output_array, &eltwise_439_output_array_intq)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_443_output, AI_STATIC,
  135, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_443_output_array, &eltwise_443_output_array_intq)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_53_output, AI_STATIC,
  136, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_53_output_array, &eltwise_53_output_array_intq)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_68_output, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_68_output_array, &eltwise_68_output_array_intq)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_72_output, AI_STATIC,
  138, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_72_output_array, &eltwise_72_output_array_intq)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_76_output, AI_STATIC,
  139, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_76_output_array, &eltwise_76_output_array_intq)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_80_output, AI_STATIC,
  140, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_80_output_array, &eltwise_80_output_array_intq)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_84_output, AI_STATIC,
  141, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_84_output_array, &eltwise_84_output_array_intq)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_88_output, AI_STATIC,
  142, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_88_output_array, &eltwise_88_output_array_intq)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_8_output, AI_STATIC,
  143, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_8_output_array, &eltwise_8_output_array_intq)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_92_output, AI_STATIC,
  144, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_92_output_array, &eltwise_92_output_array_intq)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_96_output, AI_STATIC,
  145, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_96_output_array, &eltwise_96_output_array_intq)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  gemm_103_output, AI_STATIC,
  146, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_103_output_array, &gemm_103_output_array_intq)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  gemm_103_scratch0, AI_STATIC,
  147, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_103_scratch0_array, NULL)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  gemm_107_output, AI_STATIC,
  148, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_107_output_array, &gemm_107_output_array_intq)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  gemm_107_scratch0, AI_STATIC,
  149, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_107_scratch0_array, NULL)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  gemm_111_output, AI_STATIC,
  150, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_111_output_array, &gemm_111_output_array_intq)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  gemm_111_scratch0, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_111_scratch0_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  gemm_115_output, AI_STATIC,
  152, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_115_output_array, &gemm_115_output_array_intq)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  gemm_115_scratch0, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_115_scratch0_array, NULL)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  gemm_119_output, AI_STATIC,
  154, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_119_output_array, &gemm_119_output_array_intq)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  gemm_119_scratch0, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_119_scratch0_array, NULL)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_output, AI_STATIC,
  156, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_11_output_array, &gemm_11_output_array_intq)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_scratch0, AI_STATIC,
  157, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_11_scratch0_array, NULL)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  gemm_123_output, AI_STATIC,
  158, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_123_output_array, &gemm_123_output_array_intq)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  gemm_123_scratch0, AI_STATIC,
  159, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_123_scratch0_array, NULL)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  gemm_127_output, AI_STATIC,
  160, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_127_output_array, &gemm_127_output_array_intq)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  gemm_127_scratch0, AI_STATIC,
  161, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_127_scratch0_array, NULL)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_output, AI_STATIC,
  162, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_12_output_array, &gemm_12_output_array_intq)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_scratch0, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_12_scratch0_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  gemm_131_output, AI_STATIC,
  164, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_131_output_array, &gemm_131_output_array_intq)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  gemm_131_scratch0, AI_STATIC,
  165, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_131_scratch0_array, NULL)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  gemm_135_output, AI_STATIC,
  166, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_135_output_array, &gemm_135_output_array_intq)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  gemm_135_scratch0, AI_STATIC,
  167, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_135_scratch0_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  gemm_139_output, AI_STATIC,
  168, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_139_output_array, &gemm_139_output_array_intq)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  gemm_139_scratch0, AI_STATIC,
  169, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_139_scratch0_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_output, AI_STATIC,
  170, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_13_output_array, &gemm_13_output_array_intq)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_scratch0, AI_STATIC,
  171, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_13_scratch0_array, NULL)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  gemm_143_output, AI_STATIC,
  172, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_143_output_array, &gemm_143_output_array_intq)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  gemm_143_scratch0, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_143_scratch0_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  gemm_147_output, AI_STATIC,
  174, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_147_output_array, &gemm_147_output_array_intq)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  gemm_147_scratch0, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_147_scratch0_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output, AI_STATIC,
  176, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_14_output_array, &gemm_14_output_array_intq)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_scratch0, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_14_scratch0_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  gemm_151_output, AI_STATIC,
  178, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_151_output_array, &gemm_151_output_array_intq)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  gemm_151_scratch0, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_151_scratch0_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  gemm_155_output, AI_STATIC,
  180, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_155_output_array, &gemm_155_output_array_intq)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  gemm_155_scratch0, AI_STATIC,
  181, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_155_scratch0_array, NULL)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  gemm_159_output, AI_STATIC,
  182, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_159_output_array, &gemm_159_output_array_intq)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  gemm_159_scratch0, AI_STATIC,
  183, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_159_scratch0_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_output, AI_STATIC,
  184, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_15_output_array, &gemm_15_output_array_intq)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_scratch0, AI_STATIC,
  185, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_15_scratch0_array, NULL)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  gemm_163_output, AI_STATIC,
  186, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_163_output_array, &gemm_163_output_array_intq)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  gemm_163_scratch0, AI_STATIC,
  187, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_163_scratch0_array, NULL)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  gemm_167_output, AI_STATIC,
  188, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_167_output_array, &gemm_167_output_array_intq)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  gemm_167_scratch0, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_167_scratch0_array, NULL)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_output, AI_STATIC,
  190, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_16_output_array, &gemm_16_output_array_intq)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_scratch0, AI_STATIC,
  191, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_16_scratch0_array, NULL)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  gemm_171_output, AI_STATIC,
  192, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_171_output_array, &gemm_171_output_array_intq)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  gemm_171_scratch0, AI_STATIC,
  193, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_171_scratch0_array, NULL)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  gemm_175_output, AI_STATIC,
  194, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_175_output_array, &gemm_175_output_array_intq)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  gemm_175_scratch0, AI_STATIC,
  195, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_175_scratch0_array, NULL)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  gemm_179_output, AI_STATIC,
  196, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_179_output_array, &gemm_179_output_array_intq)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  gemm_179_scratch0, AI_STATIC,
  197, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_179_scratch0_array, NULL)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_output, AI_STATIC,
  198, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_17_output_array, &gemm_17_output_array_intq)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_scratch0, AI_STATIC,
  199, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_17_scratch0_array, NULL)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  gemm_183_output, AI_STATIC,
  200, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_183_output_array, &gemm_183_output_array_intq)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  gemm_183_scratch0, AI_STATIC,
  201, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_183_scratch0_array, NULL)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  gemm_187_output, AI_STATIC,
  202, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_187_output_array, &gemm_187_output_array_intq)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  gemm_187_scratch0, AI_STATIC,
  203, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_187_scratch0_array, NULL)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  gemm_18_output, AI_STATIC,
  204, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_18_output_array, &gemm_18_output_array_intq)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  gemm_18_scratch0, AI_STATIC,
  205, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_18_scratch0_array, NULL)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  gemm_191_output, AI_STATIC,
  206, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_191_output_array, &gemm_191_output_array_intq)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  gemm_191_scratch0, AI_STATIC,
  207, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_191_scratch0_array, NULL)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  gemm_195_output, AI_STATIC,
  208, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_195_output_array, &gemm_195_output_array_intq)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  gemm_195_scratch0, AI_STATIC,
  209, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_195_scratch0_array, NULL)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  gemm_199_output, AI_STATIC,
  210, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_199_output_array, &gemm_199_output_array_intq)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  gemm_199_scratch0, AI_STATIC,
  211, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_199_scratch0_array, NULL)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_output, AI_STATIC,
  212, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_19_output_array, &gemm_19_output_array_intq)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_scratch0, AI_STATIC,
  213, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_19_scratch0_array, NULL)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  gemm_203_output, AI_STATIC,
  214, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_203_output_array, &gemm_203_output_array_intq)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  gemm_203_scratch0, AI_STATIC,
  215, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_203_scratch0_array, NULL)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  gemm_207_output, AI_STATIC,
  216, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_207_output_array, &gemm_207_output_array_intq)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  gemm_207_scratch0, AI_STATIC,
  217, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_207_scratch0_array, NULL)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  gemm_20_output, AI_STATIC,
  218, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_20_output_array, &gemm_20_output_array_intq)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  gemm_20_scratch0, AI_STATIC,
  219, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_20_scratch0_array, NULL)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  gemm_211_output, AI_STATIC,
  220, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_211_output_array, &gemm_211_output_array_intq)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  gemm_211_scratch0, AI_STATIC,
  221, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_211_scratch0_array, NULL)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  gemm_215_output, AI_STATIC,
  222, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_215_output_array, &gemm_215_output_array_intq)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  gemm_215_scratch0, AI_STATIC,
  223, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_215_scratch0_array, NULL)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  gemm_219_output, AI_STATIC,
  224, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_219_output_array, &gemm_219_output_array_intq)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  gemm_219_scratch0, AI_STATIC,
  225, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_219_scratch0_array, NULL)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  gemm_21_output, AI_STATIC,
  226, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_21_output_array, &gemm_21_output_array_intq)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  gemm_21_scratch0, AI_STATIC,
  227, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_21_scratch0_array, NULL)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  gemm_223_output, AI_STATIC,
  228, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_223_output_array, &gemm_223_output_array_intq)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  gemm_223_scratch0, AI_STATIC,
  229, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_223_scratch0_array, NULL)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  gemm_227_output, AI_STATIC,
  230, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_227_output_array, &gemm_227_output_array_intq)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  gemm_227_scratch0, AI_STATIC,
  231, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_227_scratch0_array, NULL)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  gemm_22_output, AI_STATIC,
  232, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_22_output_array, &gemm_22_output_array_intq)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  gemm_22_scratch0, AI_STATIC,
  233, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_22_scratch0_array, NULL)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  gemm_231_output, AI_STATIC,
  234, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_231_output_array, &gemm_231_output_array_intq)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  gemm_231_scratch0, AI_STATIC,
  235, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_231_scratch0_array, NULL)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  gemm_232_output, AI_STATIC,
  236, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_232_output_array, &gemm_232_output_array_intq)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  gemm_232_scratch0, AI_STATIC,
  237, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_232_scratch0_array, NULL)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  gemm_236_output, AI_STATIC,
  238, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_236_output_array, &gemm_236_output_array_intq)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  gemm_236_scratch0, AI_STATIC,
  239, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_236_scratch0_array, NULL)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  gemm_237_output, AI_STATIC,
  240, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_237_output_array, &gemm_237_output_array_intq)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  gemm_237_scratch0, AI_STATIC,
  241, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_237_scratch0_array, NULL)

/* Tensor #242 */
AI_TENSOR_OBJ_DECLARE(
  gemm_241_output, AI_STATIC,
  242, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_241_output_array, &gemm_241_output_array_intq)

/* Tensor #243 */
AI_TENSOR_OBJ_DECLARE(
  gemm_241_scratch0, AI_STATIC,
  243, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_241_scratch0_array, NULL)

/* Tensor #244 */
AI_TENSOR_OBJ_DECLARE(
  gemm_242_output, AI_STATIC,
  244, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_242_output_array, &gemm_242_output_array_intq)

/* Tensor #245 */
AI_TENSOR_OBJ_DECLARE(
  gemm_242_scratch0, AI_STATIC,
  245, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_242_scratch0_array, NULL)

/* Tensor #246 */
AI_TENSOR_OBJ_DECLARE(
  gemm_253_output, AI_STATIC,
  246, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_253_output_array, &gemm_253_output_array_intq)

/* Tensor #247 */
AI_TENSOR_OBJ_DECLARE(
  gemm_253_scratch0, AI_STATIC,
  247, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_253_scratch0_array, NULL)

/* Tensor #248 */
AI_TENSOR_OBJ_DECLARE(
  gemm_253_weights, AI_STATIC,
  248, 0x1,
  AI_SHAPE_INIT(4, 64, 64, 1, 1), AI_STRIDE_INIT(4, 1, 64, 4096, 4096),
  1, &gemm_253_weights_array, &gemm_253_weights_array_intq)

/* Tensor #249 */
AI_TENSOR_OBJ_DECLARE(
  gemm_254_bias, AI_STATIC,
  249, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_254_bias_array, NULL)

/* Tensor #250 */
AI_TENSOR_OBJ_DECLARE(
  gemm_254_output, AI_STATIC,
  250, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_254_output_array, &gemm_254_output_array_intq)

/* Tensor #251 */
AI_TENSOR_OBJ_DECLARE(
  gemm_254_scratch0, AI_STATIC,
  251, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_254_scratch0_array, NULL)

/* Tensor #252 */
AI_TENSOR_OBJ_DECLARE(
  gemm_254_weights, AI_STATIC,
  252, 0x1,
  AI_SHAPE_INIT(4, 64, 64, 1, 1), AI_STRIDE_INIT(4, 1, 64, 4096, 4096),
  1, &gemm_254_weights_array, &gemm_254_weights_array_intq)

/* Tensor #253 */
AI_TENSOR_OBJ_DECLARE(
  gemm_257_output, AI_STATIC,
  253, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_257_output_array, &gemm_257_output_array_intq)

/* Tensor #254 */
AI_TENSOR_OBJ_DECLARE(
  gemm_257_scratch0, AI_STATIC,
  254, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_257_scratch0_array, NULL)

/* Tensor #255 */
AI_TENSOR_OBJ_DECLARE(
  gemm_258_output, AI_STATIC,
  255, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_258_output_array, &gemm_258_output_array_intq)

/* Tensor #256 */
AI_TENSOR_OBJ_DECLARE(
  gemm_258_scratch0, AI_STATIC,
  256, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_258_scratch0_array, NULL)

/* Tensor #257 */
AI_TENSOR_OBJ_DECLARE(
  gemm_259_output, AI_STATIC,
  257, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_259_output_array, &gemm_259_output_array_intq)

/* Tensor #258 */
AI_TENSOR_OBJ_DECLARE(
  gemm_259_scratch0, AI_STATIC,
  258, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_259_scratch0_array, NULL)

/* Tensor #259 */
AI_TENSOR_OBJ_DECLARE(
  gemm_260_output, AI_STATIC,
  259, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_260_output_array, &gemm_260_output_array_intq)

/* Tensor #260 */
AI_TENSOR_OBJ_DECLARE(
  gemm_260_scratch0, AI_STATIC,
  260, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_260_scratch0_array, NULL)

/* Tensor #261 */
AI_TENSOR_OBJ_DECLARE(
  gemm_261_output, AI_STATIC,
  261, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_261_output_array, &gemm_261_output_array_intq)

/* Tensor #262 */
AI_TENSOR_OBJ_DECLARE(
  gemm_261_scratch0, AI_STATIC,
  262, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_261_scratch0_array, NULL)

/* Tensor #263 */
AI_TENSOR_OBJ_DECLARE(
  gemm_262_output, AI_STATIC,
  263, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_262_output_array, &gemm_262_output_array_intq)

/* Tensor #264 */
AI_TENSOR_OBJ_DECLARE(
  gemm_262_scratch0, AI_STATIC,
  264, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_262_scratch0_array, NULL)

/* Tensor #265 */
AI_TENSOR_OBJ_DECLARE(
  gemm_263_output, AI_STATIC,
  265, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_263_output_array, &gemm_263_output_array_intq)

/* Tensor #266 */
AI_TENSOR_OBJ_DECLARE(
  gemm_263_scratch0, AI_STATIC,
  266, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_263_scratch0_array, NULL)

/* Tensor #267 */
AI_TENSOR_OBJ_DECLARE(
  gemm_264_output, AI_STATIC,
  267, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_264_output_array, &gemm_264_output_array_intq)

/* Tensor #268 */
AI_TENSOR_OBJ_DECLARE(
  gemm_264_scratch0, AI_STATIC,
  268, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_264_scratch0_array, NULL)

/* Tensor #269 */
AI_TENSOR_OBJ_DECLARE(
  gemm_265_output, AI_STATIC,
  269, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_265_output_array, &gemm_265_output_array_intq)

/* Tensor #270 */
AI_TENSOR_OBJ_DECLARE(
  gemm_265_scratch0, AI_STATIC,
  270, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_265_scratch0_array, NULL)

/* Tensor #271 */
AI_TENSOR_OBJ_DECLARE(
  gemm_266_output, AI_STATIC,
  271, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_266_output_array, &gemm_266_output_array_intq)

/* Tensor #272 */
AI_TENSOR_OBJ_DECLARE(
  gemm_266_scratch0, AI_STATIC,
  272, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_266_scratch0_array, NULL)

/* Tensor #273 */
AI_TENSOR_OBJ_DECLARE(
  gemm_267_output, AI_STATIC,
  273, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_267_output_array, &gemm_267_output_array_intq)

/* Tensor #274 */
AI_TENSOR_OBJ_DECLARE(
  gemm_267_scratch0, AI_STATIC,
  274, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_267_scratch0_array, NULL)

/* Tensor #275 */
AI_TENSOR_OBJ_DECLARE(
  gemm_268_output, AI_STATIC,
  275, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_268_output_array, &gemm_268_output_array_intq)

/* Tensor #276 */
AI_TENSOR_OBJ_DECLARE(
  gemm_268_scratch0, AI_STATIC,
  276, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_268_scratch0_array, NULL)

/* Tensor #277 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_output, AI_STATIC,
  277, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_26_output_array, &gemm_26_output_array_intq)

/* Tensor #278 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_scratch0, AI_STATIC,
  278, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_26_scratch0_array, NULL)

/* Tensor #279 */
AI_TENSOR_OBJ_DECLARE(
  gemm_271_output, AI_STATIC,
  279, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_271_output_array, &gemm_271_output_array_intq)

/* Tensor #280 */
AI_TENSOR_OBJ_DECLARE(
  gemm_271_scratch0, AI_STATIC,
  280, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_271_scratch0_array, NULL)

/* Tensor #281 */
AI_TENSOR_OBJ_DECLARE(
  gemm_272_output, AI_STATIC,
  281, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_272_output_array, &gemm_272_output_array_intq)

/* Tensor #282 */
AI_TENSOR_OBJ_DECLARE(
  gemm_272_scratch0, AI_STATIC,
  282, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_272_scratch0_array, NULL)

/* Tensor #283 */
AI_TENSOR_OBJ_DECLARE(
  gemm_273_output, AI_STATIC,
  283, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_273_output_array, &gemm_273_output_array_intq)

/* Tensor #284 */
AI_TENSOR_OBJ_DECLARE(
  gemm_273_scratch0, AI_STATIC,
  284, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_273_scratch0_array, NULL)

/* Tensor #285 */
AI_TENSOR_OBJ_DECLARE(
  gemm_274_output, AI_STATIC,
  285, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_274_output_array, &gemm_274_output_array_intq)

/* Tensor #286 */
AI_TENSOR_OBJ_DECLARE(
  gemm_274_scratch0, AI_STATIC,
  286, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_274_scratch0_array, NULL)

/* Tensor #287 */
AI_TENSOR_OBJ_DECLARE(
  gemm_275_output, AI_STATIC,
  287, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_275_output_array, &gemm_275_output_array_intq)

/* Tensor #288 */
AI_TENSOR_OBJ_DECLARE(
  gemm_275_scratch0, AI_STATIC,
  288, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_275_scratch0_array, NULL)

/* Tensor #289 */
AI_TENSOR_OBJ_DECLARE(
  gemm_276_output, AI_STATIC,
  289, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_276_output_array, &gemm_276_output_array_intq)

/* Tensor #290 */
AI_TENSOR_OBJ_DECLARE(
  gemm_276_scratch0, AI_STATIC,
  290, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_276_scratch0_array, NULL)

/* Tensor #291 */
AI_TENSOR_OBJ_DECLARE(
  gemm_277_output, AI_STATIC,
  291, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_277_output_array, &gemm_277_output_array_intq)

/* Tensor #292 */
AI_TENSOR_OBJ_DECLARE(
  gemm_277_scratch0, AI_STATIC,
  292, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_277_scratch0_array, NULL)

/* Tensor #293 */
AI_TENSOR_OBJ_DECLARE(
  gemm_278_output, AI_STATIC,
  293, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_278_output_array, &gemm_278_output_array_intq)

/* Tensor #294 */
AI_TENSOR_OBJ_DECLARE(
  gemm_278_scratch0, AI_STATIC,
  294, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_278_scratch0_array, NULL)

/* Tensor #295 */
AI_TENSOR_OBJ_DECLARE(
  gemm_279_output, AI_STATIC,
  295, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_279_output_array, &gemm_279_output_array_intq)

/* Tensor #296 */
AI_TENSOR_OBJ_DECLARE(
  gemm_279_scratch0, AI_STATIC,
  296, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_279_scratch0_array, NULL)

/* Tensor #297 */
AI_TENSOR_OBJ_DECLARE(
  gemm_27_output, AI_STATIC,
  297, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_27_output_array, &gemm_27_output_array_intq)

/* Tensor #298 */
AI_TENSOR_OBJ_DECLARE(
  gemm_27_scratch0, AI_STATIC,
  298, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_27_scratch0_array, NULL)

/* Tensor #299 */
AI_TENSOR_OBJ_DECLARE(
  gemm_280_output, AI_STATIC,
  299, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_280_output_array, &gemm_280_output_array_intq)

/* Tensor #300 */
AI_TENSOR_OBJ_DECLARE(
  gemm_280_scratch0, AI_STATIC,
  300, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_280_scratch0_array, NULL)

/* Tensor #301 */
AI_TENSOR_OBJ_DECLARE(
  gemm_281_output, AI_STATIC,
  301, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_281_output_array, &gemm_281_output_array_intq)

/* Tensor #302 */
AI_TENSOR_OBJ_DECLARE(
  gemm_281_scratch0, AI_STATIC,
  302, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_281_scratch0_array, NULL)

/* Tensor #303 */
AI_TENSOR_OBJ_DECLARE(
  gemm_282_output, AI_STATIC,
  303, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_282_output_array, &gemm_282_output_array_intq)

/* Tensor #304 */
AI_TENSOR_OBJ_DECLARE(
  gemm_282_scratch0, AI_STATIC,
  304, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_282_scratch0_array, NULL)

/* Tensor #305 */
AI_TENSOR_OBJ_DECLARE(
  gemm_285_output, AI_STATIC,
  305, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_285_output_array, &gemm_285_output_array_intq)

/* Tensor #306 */
AI_TENSOR_OBJ_DECLARE(
  gemm_285_scratch0, AI_STATIC,
  306, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_285_scratch0_array, NULL)

/* Tensor #307 */
AI_TENSOR_OBJ_DECLARE(
  gemm_286_output, AI_STATIC,
  307, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_286_output_array, &gemm_286_output_array_intq)

/* Tensor #308 */
AI_TENSOR_OBJ_DECLARE(
  gemm_286_scratch0, AI_STATIC,
  308, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_286_scratch0_array, NULL)

/* Tensor #309 */
AI_TENSOR_OBJ_DECLARE(
  gemm_287_output, AI_STATIC,
  309, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_287_output_array, &gemm_287_output_array_intq)

/* Tensor #310 */
AI_TENSOR_OBJ_DECLARE(
  gemm_287_scratch0, AI_STATIC,
  310, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_287_scratch0_array, NULL)

/* Tensor #311 */
AI_TENSOR_OBJ_DECLARE(
  gemm_288_output, AI_STATIC,
  311, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_288_output_array, &gemm_288_output_array_intq)

/* Tensor #312 */
AI_TENSOR_OBJ_DECLARE(
  gemm_288_scratch0, AI_STATIC,
  312, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_288_scratch0_array, NULL)

/* Tensor #313 */
AI_TENSOR_OBJ_DECLARE(
  gemm_289_output, AI_STATIC,
  313, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_289_output_array, &gemm_289_output_array_intq)

/* Tensor #314 */
AI_TENSOR_OBJ_DECLARE(
  gemm_289_scratch0, AI_STATIC,
  314, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_289_scratch0_array, NULL)

/* Tensor #315 */
AI_TENSOR_OBJ_DECLARE(
  gemm_28_output, AI_STATIC,
  315, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_28_output_array, &gemm_28_output_array_intq)

/* Tensor #316 */
AI_TENSOR_OBJ_DECLARE(
  gemm_28_scratch0, AI_STATIC,
  316, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_28_scratch0_array, NULL)

/* Tensor #317 */
AI_TENSOR_OBJ_DECLARE(
  gemm_290_output, AI_STATIC,
  317, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_290_output_array, &gemm_290_output_array_intq)

/* Tensor #318 */
AI_TENSOR_OBJ_DECLARE(
  gemm_290_scratch0, AI_STATIC,
  318, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_290_scratch0_array, NULL)

/* Tensor #319 */
AI_TENSOR_OBJ_DECLARE(
  gemm_291_output, AI_STATIC,
  319, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_291_output_array, &gemm_291_output_array_intq)

/* Tensor #320 */
AI_TENSOR_OBJ_DECLARE(
  gemm_291_scratch0, AI_STATIC,
  320, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_291_scratch0_array, NULL)

/* Tensor #321 */
AI_TENSOR_OBJ_DECLARE(
  gemm_292_output, AI_STATIC,
  321, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_292_output_array, &gemm_292_output_array_intq)

/* Tensor #322 */
AI_TENSOR_OBJ_DECLARE(
  gemm_292_scratch0, AI_STATIC,
  322, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_292_scratch0_array, NULL)

/* Tensor #323 */
AI_TENSOR_OBJ_DECLARE(
  gemm_293_output, AI_STATIC,
  323, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_293_output_array, &gemm_293_output_array_intq)

/* Tensor #324 */
AI_TENSOR_OBJ_DECLARE(
  gemm_293_scratch0, AI_STATIC,
  324, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_293_scratch0_array, NULL)

/* Tensor #325 */
AI_TENSOR_OBJ_DECLARE(
  gemm_294_output, AI_STATIC,
  325, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_294_output_array, &gemm_294_output_array_intq)

/* Tensor #326 */
AI_TENSOR_OBJ_DECLARE(
  gemm_294_scratch0, AI_STATIC,
  326, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_294_scratch0_array, NULL)

/* Tensor #327 */
AI_TENSOR_OBJ_DECLARE(
  gemm_295_output, AI_STATIC,
  327, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_295_output_array, &gemm_295_output_array_intq)

/* Tensor #328 */
AI_TENSOR_OBJ_DECLARE(
  gemm_295_scratch0, AI_STATIC,
  328, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_295_scratch0_array, NULL)

/* Tensor #329 */
AI_TENSOR_OBJ_DECLARE(
  gemm_296_output, AI_STATIC,
  329, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_296_output_array, &gemm_296_output_array_intq)

/* Tensor #330 */
AI_TENSOR_OBJ_DECLARE(
  gemm_296_scratch0, AI_STATIC,
  330, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_296_scratch0_array, NULL)

/* Tensor #331 */
AI_TENSOR_OBJ_DECLARE(
  gemm_299_output, AI_STATIC,
  331, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_299_output_array, &gemm_299_output_array_intq)

/* Tensor #332 */
AI_TENSOR_OBJ_DECLARE(
  gemm_299_scratch0, AI_STATIC,
  332, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_299_scratch0_array, NULL)

/* Tensor #333 */
AI_TENSOR_OBJ_DECLARE(
  gemm_29_output, AI_STATIC,
  333, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_29_output_array, &gemm_29_output_array_intq)

/* Tensor #334 */
AI_TENSOR_OBJ_DECLARE(
  gemm_29_scratch0, AI_STATIC,
  334, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_29_scratch0_array, NULL)

/* Tensor #335 */
AI_TENSOR_OBJ_DECLARE(
  gemm_300_output, AI_STATIC,
  335, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_300_output_array, &gemm_300_output_array_intq)

/* Tensor #336 */
AI_TENSOR_OBJ_DECLARE(
  gemm_300_scratch0, AI_STATIC,
  336, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_300_scratch0_array, NULL)

/* Tensor #337 */
AI_TENSOR_OBJ_DECLARE(
  gemm_301_output, AI_STATIC,
  337, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_301_output_array, &gemm_301_output_array_intq)

/* Tensor #338 */
AI_TENSOR_OBJ_DECLARE(
  gemm_301_scratch0, AI_STATIC,
  338, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_301_scratch0_array, NULL)

/* Tensor #339 */
AI_TENSOR_OBJ_DECLARE(
  gemm_302_output, AI_STATIC,
  339, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_302_output_array, &gemm_302_output_array_intq)

/* Tensor #340 */
AI_TENSOR_OBJ_DECLARE(
  gemm_302_scratch0, AI_STATIC,
  340, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_302_scratch0_array, NULL)

/* Tensor #341 */
AI_TENSOR_OBJ_DECLARE(
  gemm_303_output, AI_STATIC,
  341, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_303_output_array, &gemm_303_output_array_intq)

/* Tensor #342 */
AI_TENSOR_OBJ_DECLARE(
  gemm_303_scratch0, AI_STATIC,
  342, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_303_scratch0_array, NULL)

/* Tensor #343 */
AI_TENSOR_OBJ_DECLARE(
  gemm_304_output, AI_STATIC,
  343, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_304_output_array, &gemm_304_output_array_intq)

/* Tensor #344 */
AI_TENSOR_OBJ_DECLARE(
  gemm_304_scratch0, AI_STATIC,
  344, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_304_scratch0_array, NULL)

/* Tensor #345 */
AI_TENSOR_OBJ_DECLARE(
  gemm_305_output, AI_STATIC,
  345, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_305_output_array, &gemm_305_output_array_intq)

/* Tensor #346 */
AI_TENSOR_OBJ_DECLARE(
  gemm_305_scratch0, AI_STATIC,
  346, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_305_scratch0_array, NULL)

/* Tensor #347 */
AI_TENSOR_OBJ_DECLARE(
  gemm_306_output, AI_STATIC,
  347, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_306_output_array, &gemm_306_output_array_intq)

/* Tensor #348 */
AI_TENSOR_OBJ_DECLARE(
  gemm_306_scratch0, AI_STATIC,
  348, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_306_scratch0_array, NULL)

/* Tensor #349 */
AI_TENSOR_OBJ_DECLARE(
  gemm_307_output, AI_STATIC,
  349, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_307_output_array, &gemm_307_output_array_intq)

/* Tensor #350 */
AI_TENSOR_OBJ_DECLARE(
  gemm_307_scratch0, AI_STATIC,
  350, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_307_scratch0_array, NULL)

/* Tensor #351 */
AI_TENSOR_OBJ_DECLARE(
  gemm_308_output, AI_STATIC,
  351, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_308_output_array, &gemm_308_output_array_intq)

/* Tensor #352 */
AI_TENSOR_OBJ_DECLARE(
  gemm_308_scratch0, AI_STATIC,
  352, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_308_scratch0_array, NULL)

/* Tensor #353 */
AI_TENSOR_OBJ_DECLARE(
  gemm_309_output, AI_STATIC,
  353, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_309_output_array, &gemm_309_output_array_intq)

/* Tensor #354 */
AI_TENSOR_OBJ_DECLARE(
  gemm_309_scratch0, AI_STATIC,
  354, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_309_scratch0_array, NULL)

/* Tensor #355 */
AI_TENSOR_OBJ_DECLARE(
  gemm_30_output, AI_STATIC,
  355, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_30_output_array, &gemm_30_output_array_intq)

/* Tensor #356 */
AI_TENSOR_OBJ_DECLARE(
  gemm_30_scratch0, AI_STATIC,
  356, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_30_scratch0_array, NULL)

/* Tensor #357 */
AI_TENSOR_OBJ_DECLARE(
  gemm_310_output, AI_STATIC,
  357, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_310_output_array, &gemm_310_output_array_intq)

/* Tensor #358 */
AI_TENSOR_OBJ_DECLARE(
  gemm_310_scratch0, AI_STATIC,
  358, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_310_scratch0_array, NULL)

/* Tensor #359 */
AI_TENSOR_OBJ_DECLARE(
  gemm_313_output, AI_STATIC,
  359, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_313_output_array, &gemm_313_output_array_intq)

/* Tensor #360 */
AI_TENSOR_OBJ_DECLARE(
  gemm_313_scratch0, AI_STATIC,
  360, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_313_scratch0_array, NULL)

/* Tensor #361 */
AI_TENSOR_OBJ_DECLARE(
  gemm_316_output, AI_STATIC,
  361, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_316_output_array, &gemm_316_output_array_intq)

/* Tensor #362 */
AI_TENSOR_OBJ_DECLARE(
  gemm_316_scratch0, AI_STATIC,
  362, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_316_scratch0_array, NULL)

/* Tensor #363 */
AI_TENSOR_OBJ_DECLARE(
  gemm_319_output, AI_STATIC,
  363, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_319_output_array, &gemm_319_output_array_intq)

/* Tensor #364 */
AI_TENSOR_OBJ_DECLARE(
  gemm_319_scratch0, AI_STATIC,
  364, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_319_scratch0_array, NULL)

/* Tensor #365 */
AI_TENSOR_OBJ_DECLARE(
  gemm_31_output, AI_STATIC,
  365, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_31_output_array, &gemm_31_output_array_intq)

/* Tensor #366 */
AI_TENSOR_OBJ_DECLARE(
  gemm_31_scratch0, AI_STATIC,
  366, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_31_scratch0_array, NULL)

/* Tensor #367 */
AI_TENSOR_OBJ_DECLARE(
  gemm_322_output, AI_STATIC,
  367, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_322_output_array, &gemm_322_output_array_intq)

/* Tensor #368 */
AI_TENSOR_OBJ_DECLARE(
  gemm_322_scratch0, AI_STATIC,
  368, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_322_scratch0_array, NULL)

/* Tensor #369 */
AI_TENSOR_OBJ_DECLARE(
  gemm_325_output, AI_STATIC,
  369, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_325_output_array, &gemm_325_output_array_intq)

/* Tensor #370 */
AI_TENSOR_OBJ_DECLARE(
  gemm_325_scratch0, AI_STATIC,
  370, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_325_scratch0_array, NULL)

/* Tensor #371 */
AI_TENSOR_OBJ_DECLARE(
  gemm_328_output, AI_STATIC,
  371, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_328_output_array, &gemm_328_output_array_intq)

/* Tensor #372 */
AI_TENSOR_OBJ_DECLARE(
  gemm_328_scratch0, AI_STATIC,
  372, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_328_scratch0_array, NULL)

/* Tensor #373 */
AI_TENSOR_OBJ_DECLARE(
  gemm_32_output, AI_STATIC,
  373, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_32_output_array, &gemm_32_output_array_intq)

/* Tensor #374 */
AI_TENSOR_OBJ_DECLARE(
  gemm_32_scratch0, AI_STATIC,
  374, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_32_scratch0_array, NULL)

/* Tensor #375 */
AI_TENSOR_OBJ_DECLARE(
  gemm_331_output, AI_STATIC,
  375, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_331_output_array, &gemm_331_output_array_intq)

/* Tensor #376 */
AI_TENSOR_OBJ_DECLARE(
  gemm_331_scratch0, AI_STATIC,
  376, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_331_scratch0_array, NULL)

/* Tensor #377 */
AI_TENSOR_OBJ_DECLARE(
  gemm_334_output, AI_STATIC,
  377, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_334_output_array, &gemm_334_output_array_intq)

/* Tensor #378 */
AI_TENSOR_OBJ_DECLARE(
  gemm_334_scratch0, AI_STATIC,
  378, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_334_scratch0_array, NULL)

/* Tensor #379 */
AI_TENSOR_OBJ_DECLARE(
  gemm_337_output, AI_STATIC,
  379, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_337_output_array, &gemm_337_output_array_intq)

/* Tensor #380 */
AI_TENSOR_OBJ_DECLARE(
  gemm_337_scratch0, AI_STATIC,
  380, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_337_scratch0_array, NULL)

/* Tensor #381 */
AI_TENSOR_OBJ_DECLARE(
  gemm_33_output, AI_STATIC,
  381, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_33_output_array, &gemm_33_output_array_intq)

/* Tensor #382 */
AI_TENSOR_OBJ_DECLARE(
  gemm_33_scratch0, AI_STATIC,
  382, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_33_scratch0_array, NULL)

/* Tensor #383 */
AI_TENSOR_OBJ_DECLARE(
  gemm_340_output, AI_STATIC,
  383, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_340_output_array, &gemm_340_output_array_intq)

/* Tensor #384 */
AI_TENSOR_OBJ_DECLARE(
  gemm_340_scratch0, AI_STATIC,
  384, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_340_scratch0_array, NULL)

/* Tensor #385 */
AI_TENSOR_OBJ_DECLARE(
  gemm_343_output, AI_STATIC,
  385, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_343_output_array, &gemm_343_output_array_intq)

/* Tensor #386 */
AI_TENSOR_OBJ_DECLARE(
  gemm_343_scratch0, AI_STATIC,
  386, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_343_scratch0_array, NULL)

/* Tensor #387 */
AI_TENSOR_OBJ_DECLARE(
  gemm_346_output, AI_STATIC,
  387, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_346_output_array, &gemm_346_output_array_intq)

/* Tensor #388 */
AI_TENSOR_OBJ_DECLARE(
  gemm_346_scratch0, AI_STATIC,
  388, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_346_scratch0_array, NULL)

/* Tensor #389 */
AI_TENSOR_OBJ_DECLARE(
  gemm_349_output, AI_STATIC,
  389, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_349_output_array, &gemm_349_output_array_intq)

/* Tensor #390 */
AI_TENSOR_OBJ_DECLARE(
  gemm_349_scratch0, AI_STATIC,
  390, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_349_scratch0_array, NULL)

/* Tensor #391 */
AI_TENSOR_OBJ_DECLARE(
  gemm_34_output, AI_STATIC,
  391, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_34_output_array, &gemm_34_output_array_intq)

/* Tensor #392 */
AI_TENSOR_OBJ_DECLARE(
  gemm_34_scratch0, AI_STATIC,
  392, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_34_scratch0_array, NULL)

/* Tensor #393 */
AI_TENSOR_OBJ_DECLARE(
  gemm_352_output, AI_STATIC,
  393, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_352_output_array, &gemm_352_output_array_intq)

/* Tensor #394 */
AI_TENSOR_OBJ_DECLARE(
  gemm_352_scratch0, AI_STATIC,
  394, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_352_scratch0_array, NULL)

/* Tensor #395 */
AI_TENSOR_OBJ_DECLARE(
  gemm_355_output, AI_STATIC,
  395, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_355_output_array, &gemm_355_output_array_intq)

/* Tensor #396 */
AI_TENSOR_OBJ_DECLARE(
  gemm_355_scratch0, AI_STATIC,
  396, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_355_scratch0_array, NULL)

/* Tensor #397 */
AI_TENSOR_OBJ_DECLARE(
  gemm_358_output, AI_STATIC,
  397, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_358_output_array, &gemm_358_output_array_intq)

/* Tensor #398 */
AI_TENSOR_OBJ_DECLARE(
  gemm_358_scratch0, AI_STATIC,
  398, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_358_scratch0_array, NULL)

/* Tensor #399 */
AI_TENSOR_OBJ_DECLARE(
  gemm_35_output, AI_STATIC,
  399, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_35_output_array, &gemm_35_output_array_intq)

/* Tensor #400 */
AI_TENSOR_OBJ_DECLARE(
  gemm_35_scratch0, AI_STATIC,
  400, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_35_scratch0_array, NULL)

/* Tensor #401 */
AI_TENSOR_OBJ_DECLARE(
  gemm_361_output, AI_STATIC,
  401, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_361_output_array, &gemm_361_output_array_intq)

/* Tensor #402 */
AI_TENSOR_OBJ_DECLARE(
  gemm_361_scratch0, AI_STATIC,
  402, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_361_scratch0_array, NULL)

/* Tensor #403 */
AI_TENSOR_OBJ_DECLARE(
  gemm_364_output, AI_STATIC,
  403, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_364_output_array, &gemm_364_output_array_intq)

/* Tensor #404 */
AI_TENSOR_OBJ_DECLARE(
  gemm_364_scratch0, AI_STATIC,
  404, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_364_scratch0_array, NULL)

/* Tensor #405 */
AI_TENSOR_OBJ_DECLARE(
  gemm_367_output, AI_STATIC,
  405, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_367_output_array, &gemm_367_output_array_intq)

/* Tensor #406 */
AI_TENSOR_OBJ_DECLARE(
  gemm_367_scratch0, AI_STATIC,
  406, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_367_scratch0_array, NULL)

/* Tensor #407 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_output, AI_STATIC,
  407, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_36_output_array, &gemm_36_output_array_intq)

/* Tensor #408 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_scratch0, AI_STATIC,
  408, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_36_scratch0_array, NULL)

/* Tensor #409 */
AI_TENSOR_OBJ_DECLARE(
  gemm_370_output, AI_STATIC,
  409, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_370_output_array, &gemm_370_output_array_intq)

/* Tensor #410 */
AI_TENSOR_OBJ_DECLARE(
  gemm_370_scratch0, AI_STATIC,
  410, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_370_scratch0_array, NULL)

/* Tensor #411 */
AI_TENSOR_OBJ_DECLARE(
  gemm_373_output, AI_STATIC,
  411, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_373_output_array, &gemm_373_output_array_intq)

/* Tensor #412 */
AI_TENSOR_OBJ_DECLARE(
  gemm_373_scratch0, AI_STATIC,
  412, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_373_scratch0_array, NULL)

/* Tensor #413 */
AI_TENSOR_OBJ_DECLARE(
  gemm_376_output, AI_STATIC,
  413, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_376_output_array, &gemm_376_output_array_intq)

/* Tensor #414 */
AI_TENSOR_OBJ_DECLARE(
  gemm_376_scratch0, AI_STATIC,
  414, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_376_scratch0_array, NULL)

/* Tensor #415 */
AI_TENSOR_OBJ_DECLARE(
  gemm_379_output, AI_STATIC,
  415, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_379_output_array, &gemm_379_output_array_intq)

/* Tensor #416 */
AI_TENSOR_OBJ_DECLARE(
  gemm_379_scratch0, AI_STATIC,
  416, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_379_scratch0_array, NULL)

/* Tensor #417 */
AI_TENSOR_OBJ_DECLARE(
  gemm_37_output, AI_STATIC,
  417, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_37_output_array, &gemm_37_output_array_intq)

/* Tensor #418 */
AI_TENSOR_OBJ_DECLARE(
  gemm_37_scratch0, AI_STATIC,
  418, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_37_scratch0_array, NULL)

/* Tensor #419 */
AI_TENSOR_OBJ_DECLARE(
  gemm_382_output, AI_STATIC,
  419, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_382_output_array, &gemm_382_output_array_intq)

/* Tensor #420 */
AI_TENSOR_OBJ_DECLARE(
  gemm_382_scratch0, AI_STATIC,
  420, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_382_scratch0_array, NULL)

/* Tensor #421 */
AI_TENSOR_OBJ_DECLARE(
  gemm_385_output, AI_STATIC,
  421, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_385_output_array, &gemm_385_output_array_intq)

/* Tensor #422 */
AI_TENSOR_OBJ_DECLARE(
  gemm_385_scratch0, AI_STATIC,
  422, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_385_scratch0_array, NULL)

/* Tensor #423 */
AI_TENSOR_OBJ_DECLARE(
  gemm_388_output, AI_STATIC,
  423, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_388_output_array, &gemm_388_output_array_intq)

/* Tensor #424 */
AI_TENSOR_OBJ_DECLARE(
  gemm_388_scratch0, AI_STATIC,
  424, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_388_scratch0_array, NULL)

/* Tensor #425 */
AI_TENSOR_OBJ_DECLARE(
  gemm_391_output, AI_STATIC,
  425, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_391_output_array, &gemm_391_output_array_intq)

/* Tensor #426 */
AI_TENSOR_OBJ_DECLARE(
  gemm_391_scratch0, AI_STATIC,
  426, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_391_scratch0_array, NULL)

/* Tensor #427 */
AI_TENSOR_OBJ_DECLARE(
  gemm_394_output, AI_STATIC,
  427, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_394_output_array, &gemm_394_output_array_intq)

/* Tensor #428 */
AI_TENSOR_OBJ_DECLARE(
  gemm_394_scratch0, AI_STATIC,
  428, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_394_scratch0_array, NULL)

/* Tensor #429 */
AI_TENSOR_OBJ_DECLARE(
  gemm_397_output, AI_STATIC,
  429, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_397_output_array, &gemm_397_output_array_intq)

/* Tensor #430 */
AI_TENSOR_OBJ_DECLARE(
  gemm_397_scratch0, AI_STATIC,
  430, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_397_scratch0_array, NULL)

/* Tensor #431 */
AI_TENSOR_OBJ_DECLARE(
  gemm_400_output, AI_STATIC,
  431, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_400_output_array, &gemm_400_output_array_intq)

/* Tensor #432 */
AI_TENSOR_OBJ_DECLARE(
  gemm_400_scratch0, AI_STATIC,
  432, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_400_scratch0_array, NULL)

/* Tensor #433 */
AI_TENSOR_OBJ_DECLARE(
  gemm_403_output, AI_STATIC,
  433, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_403_output_array, &gemm_403_output_array_intq)

/* Tensor #434 */
AI_TENSOR_OBJ_DECLARE(
  gemm_403_scratch0, AI_STATIC,
  434, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_403_scratch0_array, NULL)

/* Tensor #435 */
AI_TENSOR_OBJ_DECLARE(
  gemm_406_output, AI_STATIC,
  435, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_406_output_array, &gemm_406_output_array_intq)

/* Tensor #436 */
AI_TENSOR_OBJ_DECLARE(
  gemm_406_scratch0, AI_STATIC,
  436, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_406_scratch0_array, NULL)

/* Tensor #437 */
AI_TENSOR_OBJ_DECLARE(
  gemm_409_output, AI_STATIC,
  437, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_409_output_array, &gemm_409_output_array_intq)

/* Tensor #438 */
AI_TENSOR_OBJ_DECLARE(
  gemm_409_scratch0, AI_STATIC,
  438, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_409_scratch0_array, NULL)

/* Tensor #439 */
AI_TENSOR_OBJ_DECLARE(
  gemm_412_output, AI_STATIC,
  439, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_412_output_array, &gemm_412_output_array_intq)

/* Tensor #440 */
AI_TENSOR_OBJ_DECLARE(
  gemm_412_scratch0, AI_STATIC,
  440, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_412_scratch0_array, NULL)

/* Tensor #441 */
AI_TENSOR_OBJ_DECLARE(
  gemm_415_output, AI_STATIC,
  441, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_415_output_array, &gemm_415_output_array_intq)

/* Tensor #442 */
AI_TENSOR_OBJ_DECLARE(
  gemm_415_scratch0, AI_STATIC,
  442, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_415_scratch0_array, NULL)

/* Tensor #443 */
AI_TENSOR_OBJ_DECLARE(
  gemm_418_output, AI_STATIC,
  443, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_418_output_array, &gemm_418_output_array_intq)

/* Tensor #444 */
AI_TENSOR_OBJ_DECLARE(
  gemm_418_scratch0, AI_STATIC,
  444, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_418_scratch0_array, NULL)

/* Tensor #445 */
AI_TENSOR_OBJ_DECLARE(
  gemm_41_output, AI_STATIC,
  445, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_41_output_array, &gemm_41_output_array_intq)

/* Tensor #446 */
AI_TENSOR_OBJ_DECLARE(
  gemm_41_scratch0, AI_STATIC,
  446, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_41_scratch0_array, NULL)

/* Tensor #447 */
AI_TENSOR_OBJ_DECLARE(
  gemm_421_output, AI_STATIC,
  447, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_421_output_array, &gemm_421_output_array_intq)

/* Tensor #448 */
AI_TENSOR_OBJ_DECLARE(
  gemm_421_scratch0, AI_STATIC,
  448, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_421_scratch0_array, NULL)

/* Tensor #449 */
AI_TENSOR_OBJ_DECLARE(
  gemm_424_output, AI_STATIC,
  449, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_424_output_array, &gemm_424_output_array_intq)

/* Tensor #450 */
AI_TENSOR_OBJ_DECLARE(
  gemm_424_scratch0, AI_STATIC,
  450, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_424_scratch0_array, NULL)

/* Tensor #451 */
AI_TENSOR_OBJ_DECLARE(
  gemm_427_output, AI_STATIC,
  451, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_427_output_array, &gemm_427_output_array_intq)

/* Tensor #452 */
AI_TENSOR_OBJ_DECLARE(
  gemm_427_scratch0, AI_STATIC,
  452, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_427_scratch0_array, NULL)

/* Tensor #453 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_output, AI_STATIC,
  453, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_42_output_array, &gemm_42_output_array_intq)

/* Tensor #454 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_scratch0, AI_STATIC,
  454, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_42_scratch0_array, NULL)

/* Tensor #455 */
AI_TENSOR_OBJ_DECLARE(
  gemm_430_output, AI_STATIC,
  455, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_430_output_array, &gemm_430_output_array_intq)

/* Tensor #456 */
AI_TENSOR_OBJ_DECLARE(
  gemm_430_scratch0, AI_STATIC,
  456, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_430_scratch0_array, NULL)

/* Tensor #457 */
AI_TENSOR_OBJ_DECLARE(
  gemm_433_output, AI_STATIC,
  457, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_433_output_array, &gemm_433_output_array_intq)

/* Tensor #458 */
AI_TENSOR_OBJ_DECLARE(
  gemm_433_scratch0, AI_STATIC,
  458, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_433_scratch0_array, NULL)

/* Tensor #459 */
AI_TENSOR_OBJ_DECLARE(
  gemm_434_output, AI_STATIC,
  459, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_434_output_array, &gemm_434_output_array_intq)

/* Tensor #460 */
AI_TENSOR_OBJ_DECLARE(
  gemm_434_scratch0, AI_STATIC,
  460, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_434_scratch0_array, NULL)

/* Tensor #461 */
AI_TENSOR_OBJ_DECLARE(
  gemm_437_output, AI_STATIC,
  461, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_437_output_array, &gemm_437_output_array_intq)

/* Tensor #462 */
AI_TENSOR_OBJ_DECLARE(
  gemm_437_scratch0, AI_STATIC,
  462, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_437_scratch0_array, NULL)

/* Tensor #463 */
AI_TENSOR_OBJ_DECLARE(
  gemm_438_output, AI_STATIC,
  463, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_438_output_array, &gemm_438_output_array_intq)

/* Tensor #464 */
AI_TENSOR_OBJ_DECLARE(
  gemm_438_scratch0, AI_STATIC,
  464, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_438_scratch0_array, NULL)

/* Tensor #465 */
AI_TENSOR_OBJ_DECLARE(
  gemm_43_output, AI_STATIC,
  465, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_43_output_array, &gemm_43_output_array_intq)

/* Tensor #466 */
AI_TENSOR_OBJ_DECLARE(
  gemm_43_scratch0, AI_STATIC,
  466, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_43_scratch0_array, NULL)

/* Tensor #467 */
AI_TENSOR_OBJ_DECLARE(
  gemm_441_output, AI_STATIC,
  467, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_441_output_array, &gemm_441_output_array_intq)

/* Tensor #468 */
AI_TENSOR_OBJ_DECLARE(
  gemm_441_scratch0, AI_STATIC,
  468, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_441_scratch0_array, NULL)

/* Tensor #469 */
AI_TENSOR_OBJ_DECLARE(
  gemm_442_output, AI_STATIC,
  469, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_442_output_array, &gemm_442_output_array_intq)

/* Tensor #470 */
AI_TENSOR_OBJ_DECLARE(
  gemm_442_scratch0, AI_STATIC,
  470, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_442_scratch0_array, NULL)

/* Tensor #471 */
AI_TENSOR_OBJ_DECLARE(
  gemm_445_bias, AI_STATIC,
  471, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_445_bias_array, NULL)

/* Tensor #472 */
AI_TENSOR_OBJ_DECLARE(
  gemm_445_output, AI_STATIC,
  472, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_445_output_array, &gemm_445_output_array_intq)

/* Tensor #473 */
AI_TENSOR_OBJ_DECLARE(
  gemm_445_scratch0, AI_STATIC,
  473, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_445_scratch0_array, NULL)

/* Tensor #474 */
AI_TENSOR_OBJ_DECLARE(
  gemm_445_weights, AI_STATIC,
  474, 0x1,
  AI_SHAPE_INIT(4, 64, 64, 1, 1), AI_STRIDE_INIT(4, 1, 64, 4096, 4096),
  1, &gemm_445_weights_array, &gemm_445_weights_array_intq)

/* Tensor #475 */
AI_TENSOR_OBJ_DECLARE(
  gemm_446_bias, AI_STATIC,
  475, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &gemm_446_bias_array, NULL)

/* Tensor #476 */
AI_TENSOR_OBJ_DECLARE(
  gemm_446_output, AI_STATIC,
  476, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_446_output_array, &gemm_446_output_array_intq)

/* Tensor #477 */
AI_TENSOR_OBJ_DECLARE(
  gemm_446_scratch0, AI_STATIC,
  477, 0x0,
  AI_SHAPE_INIT(4, 1, 184, 1, 1), AI_STRIDE_INIT(4, 2, 2, 368, 368),
  1, &gemm_446_scratch0_array, NULL)

/* Tensor #478 */
AI_TENSOR_OBJ_DECLARE(
  gemm_446_weights, AI_STATIC,
  478, 0x1,
  AI_SHAPE_INIT(4, 64, 24, 1, 1), AI_STRIDE_INIT(4, 1, 64, 1536, 1536),
  1, &gemm_446_weights_array, &gemm_446_weights_array_intq)

/* Tensor #479 */
AI_TENSOR_OBJ_DECLARE(
  gemm_44_output, AI_STATIC,
  479, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_44_output_array, &gemm_44_output_array_intq)

/* Tensor #480 */
AI_TENSOR_OBJ_DECLARE(
  gemm_44_scratch0, AI_STATIC,
  480, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_44_scratch0_array, NULL)

/* Tensor #481 */
AI_TENSOR_OBJ_DECLARE(
  gemm_45_output, AI_STATIC,
  481, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_45_output_array, &gemm_45_output_array_intq)

/* Tensor #482 */
AI_TENSOR_OBJ_DECLARE(
  gemm_45_scratch0, AI_STATIC,
  482, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_45_scratch0_array, NULL)

/* Tensor #483 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_output, AI_STATIC,
  483, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_46_output_array, &gemm_46_output_array_intq)

/* Tensor #484 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_scratch0, AI_STATIC,
  484, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_46_scratch0_array, NULL)

/* Tensor #485 */
AI_TENSOR_OBJ_DECLARE(
  gemm_47_output, AI_STATIC,
  485, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_47_output_array, &gemm_47_output_array_intq)

/* Tensor #486 */
AI_TENSOR_OBJ_DECLARE(
  gemm_47_scratch0, AI_STATIC,
  486, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_47_scratch0_array, NULL)

/* Tensor #487 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_output, AI_STATIC,
  487, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_48_output_array, &gemm_48_output_array_intq)

/* Tensor #488 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_scratch0, AI_STATIC,
  488, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_48_scratch0_array, NULL)

/* Tensor #489 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_output, AI_STATIC,
  489, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_49_output_array, &gemm_49_output_array_intq)

/* Tensor #490 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_scratch0, AI_STATIC,
  490, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_49_scratch0_array, NULL)

/* Tensor #491 */
AI_TENSOR_OBJ_DECLARE(
  gemm_50_output, AI_STATIC,
  491, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_50_output_array, &gemm_50_output_array_intq)

/* Tensor #492 */
AI_TENSOR_OBJ_DECLARE(
  gemm_50_scratch0, AI_STATIC,
  492, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_50_scratch0_array, NULL)

/* Tensor #493 */
AI_TENSOR_OBJ_DECLARE(
  gemm_51_output, AI_STATIC,
  493, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_51_output_array, &gemm_51_output_array_intq)

/* Tensor #494 */
AI_TENSOR_OBJ_DECLARE(
  gemm_51_scratch0, AI_STATIC,
  494, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_51_scratch0_array, NULL)

/* Tensor #495 */
AI_TENSOR_OBJ_DECLARE(
  gemm_52_output, AI_STATIC,
  495, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_52_output_array, &gemm_52_output_array_intq)

/* Tensor #496 */
AI_TENSOR_OBJ_DECLARE(
  gemm_52_scratch0, AI_STATIC,
  496, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_52_scratch0_array, NULL)

/* Tensor #497 */
AI_TENSOR_OBJ_DECLARE(
  gemm_56_output, AI_STATIC,
  497, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_56_output_array, &gemm_56_output_array_intq)

/* Tensor #498 */
AI_TENSOR_OBJ_DECLARE(
  gemm_56_scratch0, AI_STATIC,
  498, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_56_scratch0_array, NULL)

/* Tensor #499 */
AI_TENSOR_OBJ_DECLARE(
  gemm_57_output, AI_STATIC,
  499, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_57_output_array, &gemm_57_output_array_intq)

/* Tensor #500 */
AI_TENSOR_OBJ_DECLARE(
  gemm_57_scratch0, AI_STATIC,
  500, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_57_scratch0_array, NULL)

/* Tensor #501 */
AI_TENSOR_OBJ_DECLARE(
  gemm_58_output, AI_STATIC,
  501, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_58_output_array, &gemm_58_output_array_intq)

/* Tensor #502 */
AI_TENSOR_OBJ_DECLARE(
  gemm_58_scratch0, AI_STATIC,
  502, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_58_scratch0_array, NULL)

/* Tensor #503 */
AI_TENSOR_OBJ_DECLARE(
  gemm_59_output, AI_STATIC,
  503, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_59_output_array, &gemm_59_output_array_intq)

/* Tensor #504 */
AI_TENSOR_OBJ_DECLARE(
  gemm_59_scratch0, AI_STATIC,
  504, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_59_scratch0_array, NULL)

/* Tensor #505 */
AI_TENSOR_OBJ_DECLARE(
  gemm_60_output, AI_STATIC,
  505, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_60_output_array, &gemm_60_output_array_intq)

/* Tensor #506 */
AI_TENSOR_OBJ_DECLARE(
  gemm_60_scratch0, AI_STATIC,
  506, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_60_scratch0_array, NULL)

/* Tensor #507 */
AI_TENSOR_OBJ_DECLARE(
  gemm_61_output, AI_STATIC,
  507, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_61_output_array, &gemm_61_output_array_intq)

/* Tensor #508 */
AI_TENSOR_OBJ_DECLARE(
  gemm_61_scratch0, AI_STATIC,
  508, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_61_scratch0_array, NULL)

/* Tensor #509 */
AI_TENSOR_OBJ_DECLARE(
  gemm_62_output, AI_STATIC,
  509, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_62_output_array, &gemm_62_output_array_intq)

/* Tensor #510 */
AI_TENSOR_OBJ_DECLARE(
  gemm_62_scratch0, AI_STATIC,
  510, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_62_scratch0_array, NULL)

/* Tensor #511 */
AI_TENSOR_OBJ_DECLARE(
  gemm_63_output, AI_STATIC,
  511, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_63_output_array, &gemm_63_output_array_intq)

/* Tensor #512 */
AI_TENSOR_OBJ_DECLARE(
  gemm_63_scratch0, AI_STATIC,
  512, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_63_scratch0_array, NULL)

/* Tensor #513 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_output, AI_STATIC,
  513, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_64_output_array, &gemm_64_output_array_intq)

/* Tensor #514 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_scratch0, AI_STATIC,
  514, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_64_scratch0_array, NULL)

/* Tensor #515 */
AI_TENSOR_OBJ_DECLARE(
  gemm_65_output, AI_STATIC,
  515, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_65_output_array, &gemm_65_output_array_intq)

/* Tensor #516 */
AI_TENSOR_OBJ_DECLARE(
  gemm_65_scratch0, AI_STATIC,
  516, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_65_scratch0_array, NULL)

/* Tensor #517 */
AI_TENSOR_OBJ_DECLARE(
  gemm_66_output, AI_STATIC,
  517, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_66_output_array, &gemm_66_output_array_intq)

/* Tensor #518 */
AI_TENSOR_OBJ_DECLARE(
  gemm_66_scratch0, AI_STATIC,
  518, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_66_scratch0_array, NULL)

/* Tensor #519 */
AI_TENSOR_OBJ_DECLARE(
  gemm_67_output, AI_STATIC,
  519, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_67_output_array, &gemm_67_output_array_intq)

/* Tensor #520 */
AI_TENSOR_OBJ_DECLARE(
  gemm_67_scratch0, AI_STATIC,
  520, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_67_scratch0_array, NULL)

/* Tensor #521 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_bias, AI_STATIC,
  521, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_6_bias_array, NULL)

/* Tensor #522 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_output, AI_STATIC,
  522, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_6_output_array, &gemm_6_output_array_intq)

/* Tensor #523 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_scratch0, AI_STATIC,
  523, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_6_scratch0_array, NULL)

/* Tensor #524 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_weights, AI_STATIC,
  524, 0x1,
  AI_SHAPE_INIT(4, 64, 64, 1, 1), AI_STRIDE_INIT(4, 1, 64, 4096, 4096),
  1, &gemm_6_weights_array, &gemm_6_weights_array_intq)

/* Tensor #525 */
AI_TENSOR_OBJ_DECLARE(
  gemm_71_output, AI_STATIC,
  525, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_71_output_array, &gemm_71_output_array_intq)

/* Tensor #526 */
AI_TENSOR_OBJ_DECLARE(
  gemm_71_scratch0, AI_STATIC,
  526, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_71_scratch0_array, NULL)

/* Tensor #527 */
AI_TENSOR_OBJ_DECLARE(
  gemm_75_output, AI_STATIC,
  527, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_75_output_array, &gemm_75_output_array_intq)

/* Tensor #528 */
AI_TENSOR_OBJ_DECLARE(
  gemm_75_scratch0, AI_STATIC,
  528, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_75_scratch0_array, NULL)

/* Tensor #529 */
AI_TENSOR_OBJ_DECLARE(
  gemm_79_output, AI_STATIC,
  529, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_79_output_array, &gemm_79_output_array_intq)

/* Tensor #530 */
AI_TENSOR_OBJ_DECLARE(
  gemm_79_scratch0, AI_STATIC,
  530, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_79_scratch0_array, NULL)

/* Tensor #531 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_bias, AI_STATIC,
  531, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_7_bias_array, NULL)

/* Tensor #532 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_output, AI_STATIC,
  532, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_7_output_array, &gemm_7_output_array_intq)

/* Tensor #533 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_scratch0, AI_STATIC,
  533, 0x0,
  AI_SHAPE_INIT(4, 1, 334, 1, 1), AI_STRIDE_INIT(4, 2, 2, 668, 668),
  1, &gemm_7_scratch0_array, NULL)

/* Tensor #534 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_weights, AI_STATIC,
  534, 0x1,
  AI_SHAPE_INIT(4, 14, 64, 1, 1), AI_STRIDE_INIT(4, 1, 14, 896, 896),
  1, &gemm_7_weights_array, &gemm_7_weights_array_intq)

/* Tensor #535 */
AI_TENSOR_OBJ_DECLARE(
  gemm_83_output, AI_STATIC,
  535, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_83_output_array, &gemm_83_output_array_intq)

/* Tensor #536 */
AI_TENSOR_OBJ_DECLARE(
  gemm_83_scratch0, AI_STATIC,
  536, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_83_scratch0_array, NULL)

/* Tensor #537 */
AI_TENSOR_OBJ_DECLARE(
  gemm_87_output, AI_STATIC,
  537, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_87_output_array, &gemm_87_output_array_intq)

/* Tensor #538 */
AI_TENSOR_OBJ_DECLARE(
  gemm_87_scratch0, AI_STATIC,
  538, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_87_scratch0_array, NULL)

/* Tensor #539 */
AI_TENSOR_OBJ_DECLARE(
  gemm_91_output, AI_STATIC,
  539, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_91_output_array, &gemm_91_output_array_intq)

/* Tensor #540 */
AI_TENSOR_OBJ_DECLARE(
  gemm_91_scratch0, AI_STATIC,
  540, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_91_scratch0_array, NULL)

/* Tensor #541 */
AI_TENSOR_OBJ_DECLARE(
  gemm_95_output, AI_STATIC,
  541, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_95_output_array, &gemm_95_output_array_intq)

/* Tensor #542 */
AI_TENSOR_OBJ_DECLARE(
  gemm_95_scratch0, AI_STATIC,
  542, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_95_scratch0_array, NULL)

/* Tensor #543 */
AI_TENSOR_OBJ_DECLARE(
  gemm_99_output, AI_STATIC,
  543, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_99_output_array, &gemm_99_output_array_intq)

/* Tensor #544 */
AI_TENSOR_OBJ_DECLARE(
  gemm_99_scratch0, AI_STATIC,
  544, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_99_scratch0_array, NULL)

/* Tensor #545 */
AI_TENSOR_OBJ_DECLARE(
  nl_101_output, AI_STATIC,
  545, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_101_output_array, &nl_101_output_array_intq)

/* Tensor #546 */
AI_TENSOR_OBJ_DECLARE(
  nl_105_output, AI_STATIC,
  546, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_105_output_array, &nl_105_output_array_intq)

/* Tensor #547 */
AI_TENSOR_OBJ_DECLARE(
  nl_109_output, AI_STATIC,
  547, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_109_output_array, &nl_109_output_array_intq)

/* Tensor #548 */
AI_TENSOR_OBJ_DECLARE(
  nl_113_output, AI_STATIC,
  548, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_113_output_array, &nl_113_output_array_intq)

/* Tensor #549 */
AI_TENSOR_OBJ_DECLARE(
  nl_117_output, AI_STATIC,
  549, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_117_output_array, &nl_117_output_array_intq)

/* Tensor #550 */
AI_TENSOR_OBJ_DECLARE(
  nl_121_output, AI_STATIC,
  550, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_121_output_array, &nl_121_output_array_intq)

/* Tensor #551 */
AI_TENSOR_OBJ_DECLARE(
  nl_125_output, AI_STATIC,
  551, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_125_output_array, &nl_125_output_array_intq)

/* Tensor #552 */
AI_TENSOR_OBJ_DECLARE(
  nl_129_output, AI_STATIC,
  552, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_129_output_array, &nl_129_output_array_intq)

/* Tensor #553 */
AI_TENSOR_OBJ_DECLARE(
  nl_133_output, AI_STATIC,
  553, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_133_output_array, &nl_133_output_array_intq)

/* Tensor #554 */
AI_TENSOR_OBJ_DECLARE(
  nl_137_output, AI_STATIC,
  554, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_137_output_array, &nl_137_output_array_intq)

/* Tensor #555 */
AI_TENSOR_OBJ_DECLARE(
  nl_141_output, AI_STATIC,
  555, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_141_output_array, &nl_141_output_array_intq)

/* Tensor #556 */
AI_TENSOR_OBJ_DECLARE(
  nl_145_output, AI_STATIC,
  556, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_145_output_array, &nl_145_output_array_intq)

/* Tensor #557 */
AI_TENSOR_OBJ_DECLARE(
  nl_149_output, AI_STATIC,
  557, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_149_output_array, &nl_149_output_array_intq)

/* Tensor #558 */
AI_TENSOR_OBJ_DECLARE(
  nl_153_output, AI_STATIC,
  558, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_153_output_array, &nl_153_output_array_intq)

/* Tensor #559 */
AI_TENSOR_OBJ_DECLARE(
  nl_157_output, AI_STATIC,
  559, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_157_output_array, &nl_157_output_array_intq)

/* Tensor #560 */
AI_TENSOR_OBJ_DECLARE(
  nl_161_output, AI_STATIC,
  560, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_161_output_array, &nl_161_output_array_intq)

/* Tensor #561 */
AI_TENSOR_OBJ_DECLARE(
  nl_165_output, AI_STATIC,
  561, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_165_output_array, &nl_165_output_array_intq)

/* Tensor #562 */
AI_TENSOR_OBJ_DECLARE(
  nl_169_output, AI_STATIC,
  562, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_169_output_array, &nl_169_output_array_intq)

/* Tensor #563 */
AI_TENSOR_OBJ_DECLARE(
  nl_173_output, AI_STATIC,
  563, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_173_output_array, &nl_173_output_array_intq)

/* Tensor #564 */
AI_TENSOR_OBJ_DECLARE(
  nl_177_output, AI_STATIC,
  564, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_177_output_array, &nl_177_output_array_intq)

/* Tensor #565 */
AI_TENSOR_OBJ_DECLARE(
  nl_181_output, AI_STATIC,
  565, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_181_output_array, &nl_181_output_array_intq)

/* Tensor #566 */
AI_TENSOR_OBJ_DECLARE(
  nl_185_output, AI_STATIC,
  566, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_185_output_array, &nl_185_output_array_intq)

/* Tensor #567 */
AI_TENSOR_OBJ_DECLARE(
  nl_189_output, AI_STATIC,
  567, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_189_output_array, &nl_189_output_array_intq)

/* Tensor #568 */
AI_TENSOR_OBJ_DECLARE(
  nl_193_output, AI_STATIC,
  568, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_193_output_array, &nl_193_output_array_intq)

/* Tensor #569 */
AI_TENSOR_OBJ_DECLARE(
  nl_197_output, AI_STATIC,
  569, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_197_output_array, &nl_197_output_array_intq)

/* Tensor #570 */
AI_TENSOR_OBJ_DECLARE(
  nl_201_output, AI_STATIC,
  570, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_201_output_array, &nl_201_output_array_intq)

/* Tensor #571 */
AI_TENSOR_OBJ_DECLARE(
  nl_205_output, AI_STATIC,
  571, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_205_output_array, &nl_205_output_array_intq)

/* Tensor #572 */
AI_TENSOR_OBJ_DECLARE(
  nl_209_output, AI_STATIC,
  572, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_209_output_array, &nl_209_output_array_intq)

/* Tensor #573 */
AI_TENSOR_OBJ_DECLARE(
  nl_213_output, AI_STATIC,
  573, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_213_output_array, &nl_213_output_array_intq)

/* Tensor #574 */
AI_TENSOR_OBJ_DECLARE(
  nl_217_output, AI_STATIC,
  574, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_217_output_array, &nl_217_output_array_intq)

/* Tensor #575 */
AI_TENSOR_OBJ_DECLARE(
  nl_221_output, AI_STATIC,
  575, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_221_output_array, &nl_221_output_array_intq)

/* Tensor #576 */
AI_TENSOR_OBJ_DECLARE(
  nl_225_output, AI_STATIC,
  576, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_225_output_array, &nl_225_output_array_intq)

/* Tensor #577 */
AI_TENSOR_OBJ_DECLARE(
  nl_229_output, AI_STATIC,
  577, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_229_output_array, &nl_229_output_array_intq)

/* Tensor #578 */
AI_TENSOR_OBJ_DECLARE(
  nl_234_output, AI_STATIC,
  578, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_234_output_array, &nl_234_output_array_intq)

/* Tensor #579 */
AI_TENSOR_OBJ_DECLARE(
  nl_239_output, AI_STATIC,
  579, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_239_output_array, &nl_239_output_array_intq)

/* Tensor #580 */
AI_TENSOR_OBJ_DECLARE(
  nl_244_output, AI_STATIC,
  580, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_244_output_array, &nl_244_output_array_intq)

/* Tensor #581 */
AI_TENSOR_OBJ_DECLARE(
  nl_24_output, AI_STATIC,
  581, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_24_output_array, &nl_24_output_array_intq)

/* Tensor #582 */
AI_TENSOR_OBJ_DECLARE(
  nl_256_output, AI_STATIC,
  582, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_256_output_array, &nl_256_output_array_intq)

/* Tensor #583 */
AI_TENSOR_OBJ_DECLARE(
  nl_270_output, AI_STATIC,
  583, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_270_output_array, &nl_270_output_array_intq)

/* Tensor #584 */
AI_TENSOR_OBJ_DECLARE(
  nl_284_output, AI_STATIC,
  584, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_284_output_array, &nl_284_output_array_intq)

/* Tensor #585 */
AI_TENSOR_OBJ_DECLARE(
  nl_298_output, AI_STATIC,
  585, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_298_output_array, &nl_298_output_array_intq)

/* Tensor #586 */
AI_TENSOR_OBJ_DECLARE(
  nl_312_output, AI_STATIC,
  586, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_312_output_array, &nl_312_output_array_intq)

/* Tensor #587 */
AI_TENSOR_OBJ_DECLARE(
  nl_315_output, AI_STATIC,
  587, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_315_output_array, &nl_315_output_array_intq)

/* Tensor #588 */
AI_TENSOR_OBJ_DECLARE(
  nl_318_output, AI_STATIC,
  588, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_318_output_array, &nl_318_output_array_intq)

/* Tensor #589 */
AI_TENSOR_OBJ_DECLARE(
  nl_321_output, AI_STATIC,
  589, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_321_output_array, &nl_321_output_array_intq)

/* Tensor #590 */
AI_TENSOR_OBJ_DECLARE(
  nl_324_output, AI_STATIC,
  590, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_324_output_array, &nl_324_output_array_intq)

/* Tensor #591 */
AI_TENSOR_OBJ_DECLARE(
  nl_327_output, AI_STATIC,
  591, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_327_output_array, &nl_327_output_array_intq)

/* Tensor #592 */
AI_TENSOR_OBJ_DECLARE(
  nl_330_output, AI_STATIC,
  592, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_330_output_array, &nl_330_output_array_intq)

/* Tensor #593 */
AI_TENSOR_OBJ_DECLARE(
  nl_333_output, AI_STATIC,
  593, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_333_output_array, &nl_333_output_array_intq)

/* Tensor #594 */
AI_TENSOR_OBJ_DECLARE(
  nl_336_output, AI_STATIC,
  594, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_336_output_array, &nl_336_output_array_intq)

/* Tensor #595 */
AI_TENSOR_OBJ_DECLARE(
  nl_339_output, AI_STATIC,
  595, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_339_output_array, &nl_339_output_array_intq)

/* Tensor #596 */
AI_TENSOR_OBJ_DECLARE(
  nl_342_output, AI_STATIC,
  596, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_342_output_array, &nl_342_output_array_intq)

/* Tensor #597 */
AI_TENSOR_OBJ_DECLARE(
  nl_345_output, AI_STATIC,
  597, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_345_output_array, &nl_345_output_array_intq)

/* Tensor #598 */
AI_TENSOR_OBJ_DECLARE(
  nl_348_output, AI_STATIC,
  598, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_348_output_array, &nl_348_output_array_intq)

/* Tensor #599 */
AI_TENSOR_OBJ_DECLARE(
  nl_351_output, AI_STATIC,
  599, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_351_output_array, &nl_351_output_array_intq)

/* Tensor #600 */
AI_TENSOR_OBJ_DECLARE(
  nl_354_output, AI_STATIC,
  600, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_354_output_array, &nl_354_output_array_intq)

/* Tensor #601 */
AI_TENSOR_OBJ_DECLARE(
  nl_357_output, AI_STATIC,
  601, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_357_output_array, &nl_357_output_array_intq)

/* Tensor #602 */
AI_TENSOR_OBJ_DECLARE(
  nl_360_output, AI_STATIC,
  602, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_360_output_array, &nl_360_output_array_intq)

/* Tensor #603 */
AI_TENSOR_OBJ_DECLARE(
  nl_363_output, AI_STATIC,
  603, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_363_output_array, &nl_363_output_array_intq)

/* Tensor #604 */
AI_TENSOR_OBJ_DECLARE(
  nl_366_output, AI_STATIC,
  604, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_366_output_array, &nl_366_output_array_intq)

/* Tensor #605 */
AI_TENSOR_OBJ_DECLARE(
  nl_369_output, AI_STATIC,
  605, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_369_output_array, &nl_369_output_array_intq)

/* Tensor #606 */
AI_TENSOR_OBJ_DECLARE(
  nl_372_output, AI_STATIC,
  606, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_372_output_array, &nl_372_output_array_intq)

/* Tensor #607 */
AI_TENSOR_OBJ_DECLARE(
  nl_375_output, AI_STATIC,
  607, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_375_output_array, &nl_375_output_array_intq)

/* Tensor #608 */
AI_TENSOR_OBJ_DECLARE(
  nl_378_output, AI_STATIC,
  608, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_378_output_array, &nl_378_output_array_intq)

/* Tensor #609 */
AI_TENSOR_OBJ_DECLARE(
  nl_381_output, AI_STATIC,
  609, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_381_output_array, &nl_381_output_array_intq)

/* Tensor #610 */
AI_TENSOR_OBJ_DECLARE(
  nl_384_output, AI_STATIC,
  610, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_384_output_array, &nl_384_output_array_intq)

/* Tensor #611 */
AI_TENSOR_OBJ_DECLARE(
  nl_387_output, AI_STATIC,
  611, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_387_output_array, &nl_387_output_array_intq)

/* Tensor #612 */
AI_TENSOR_OBJ_DECLARE(
  nl_390_output, AI_STATIC,
  612, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_390_output_array, &nl_390_output_array_intq)

/* Tensor #613 */
AI_TENSOR_OBJ_DECLARE(
  nl_393_output, AI_STATIC,
  613, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_393_output_array, &nl_393_output_array_intq)

/* Tensor #614 */
AI_TENSOR_OBJ_DECLARE(
  nl_396_output, AI_STATIC,
  614, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_396_output_array, &nl_396_output_array_intq)

/* Tensor #615 */
AI_TENSOR_OBJ_DECLARE(
  nl_399_output, AI_STATIC,
  615, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_399_output_array, &nl_399_output_array_intq)

/* Tensor #616 */
AI_TENSOR_OBJ_DECLARE(
  nl_39_output, AI_STATIC,
  616, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_39_output_array, &nl_39_output_array_intq)

/* Tensor #617 */
AI_TENSOR_OBJ_DECLARE(
  nl_402_output, AI_STATIC,
  617, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_402_output_array, &nl_402_output_array_intq)

/* Tensor #618 */
AI_TENSOR_OBJ_DECLARE(
  nl_405_output, AI_STATIC,
  618, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_405_output_array, &nl_405_output_array_intq)

/* Tensor #619 */
AI_TENSOR_OBJ_DECLARE(
  nl_408_output, AI_STATIC,
  619, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_408_output_array, &nl_408_output_array_intq)

/* Tensor #620 */
AI_TENSOR_OBJ_DECLARE(
  nl_411_output, AI_STATIC,
  620, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_411_output_array, &nl_411_output_array_intq)

/* Tensor #621 */
AI_TENSOR_OBJ_DECLARE(
  nl_414_output, AI_STATIC,
  621, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_414_output_array, &nl_414_output_array_intq)

/* Tensor #622 */
AI_TENSOR_OBJ_DECLARE(
  nl_417_output, AI_STATIC,
  622, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_417_output_array, &nl_417_output_array_intq)

/* Tensor #623 */
AI_TENSOR_OBJ_DECLARE(
  nl_420_output, AI_STATIC,
  623, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_420_output_array, &nl_420_output_array_intq)

/* Tensor #624 */
AI_TENSOR_OBJ_DECLARE(
  nl_423_output, AI_STATIC,
  624, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_423_output_array, &nl_423_output_array_intq)

/* Tensor #625 */
AI_TENSOR_OBJ_DECLARE(
  nl_426_output, AI_STATIC,
  625, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_426_output_array, &nl_426_output_array_intq)

/* Tensor #626 */
AI_TENSOR_OBJ_DECLARE(
  nl_429_output, AI_STATIC,
  626, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_429_output_array, &nl_429_output_array_intq)

/* Tensor #627 */
AI_TENSOR_OBJ_DECLARE(
  nl_432_output, AI_STATIC,
  627, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_432_output_array, &nl_432_output_array_intq)

/* Tensor #628 */
AI_TENSOR_OBJ_DECLARE(
  nl_436_output, AI_STATIC,
  628, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_436_output_array, &nl_436_output_array_intq)

/* Tensor #629 */
AI_TENSOR_OBJ_DECLARE(
  nl_440_output, AI_STATIC,
  629, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_440_output_array, &nl_440_output_array_intq)

/* Tensor #630 */
AI_TENSOR_OBJ_DECLARE(
  nl_444_output, AI_STATIC,
  630, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_444_output_array, &nl_444_output_array_intq)

/* Tensor #631 */
AI_TENSOR_OBJ_DECLARE(
  nl_54_output, AI_STATIC,
  631, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_54_output_array, &nl_54_output_array_intq)

/* Tensor #632 */
AI_TENSOR_OBJ_DECLARE(
  nl_69_output, AI_STATIC,
  632, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_69_output_array, &nl_69_output_array_intq)

/* Tensor #633 */
AI_TENSOR_OBJ_DECLARE(
  nl_73_output, AI_STATIC,
  633, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_73_output_array, &nl_73_output_array_intq)

/* Tensor #634 */
AI_TENSOR_OBJ_DECLARE(
  nl_77_output, AI_STATIC,
  634, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_77_output_array, &nl_77_output_array_intq)

/* Tensor #635 */
AI_TENSOR_OBJ_DECLARE(
  nl_81_output, AI_STATIC,
  635, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_81_output_array, &nl_81_output_array_intq)

/* Tensor #636 */
AI_TENSOR_OBJ_DECLARE(
  nl_85_output, AI_STATIC,
  636, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_85_output_array, &nl_85_output_array_intq)

/* Tensor #637 */
AI_TENSOR_OBJ_DECLARE(
  nl_89_output, AI_STATIC,
  637, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_89_output_array, &nl_89_output_array_intq)

/* Tensor #638 */
AI_TENSOR_OBJ_DECLARE(
  nl_93_output, AI_STATIC,
  638, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_93_output_array, &nl_93_output_array_intq)

/* Tensor #639 */
AI_TENSOR_OBJ_DECLARE(
  nl_97_output, AI_STATIC,
  639, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_97_output_array, &nl_97_output_array_intq)

/* Tensor #640 */
AI_TENSOR_OBJ_DECLARE(
  nl_9_output, AI_STATIC,
  640, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_9_output_array, &nl_9_output_array_intq)

/* Tensor #641 */
AI_TENSOR_OBJ_DECLARE(
  pack_246_output, AI_STATIC,
  641, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 48), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pack_246_output_array, &pack_246_output_array_intq)

/* Tensor #642 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_keras_tensor_60_output, AI_STATIC,
  642, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 48), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &serving_default_keras_tensor_60_output_array, &serving_default_keras_tensor_60_output_array_intq)

/* Tensor #643 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_keras_tensor_60_output0, AI_STATIC,
  643, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 48, 1), AI_STRIDE_INIT(4, 1, 1, 14, 672),
  1, &serving_default_keras_tensor_60_output_array, &serving_default_keras_tensor_60_output_array_intq)

/* Tensor #644 */
AI_TENSOR_OBJ_DECLARE(
  transpose_2_output, AI_STATIC,
  644, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 48), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &transpose_2_output_array, &transpose_2_output_array_intq)

/* Tensor #645 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output0, AI_STATIC,
  645, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output0_array, &unpack_252_output0_array_intq)

/* Tensor #646 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output1, AI_STATIC,
  646, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output1_array, &unpack_252_output1_array_intq)

/* Tensor #647 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output10, AI_STATIC,
  647, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output10_array, &unpack_252_output10_array_intq)

/* Tensor #648 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output11, AI_STATIC,
  648, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output11_array, &unpack_252_output11_array_intq)

/* Tensor #649 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output12, AI_STATIC,
  649, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output12_array, &unpack_252_output12_array_intq)

/* Tensor #650 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output13, AI_STATIC,
  650, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output13_array, &unpack_252_output13_array_intq)

/* Tensor #651 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output14, AI_STATIC,
  651, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output14_array, &unpack_252_output14_array_intq)

/* Tensor #652 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output15, AI_STATIC,
  652, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output15_array, &unpack_252_output15_array_intq)

/* Tensor #653 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output16, AI_STATIC,
  653, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output16_array, &unpack_252_output16_array_intq)

/* Tensor #654 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output17, AI_STATIC,
  654, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output17_array, &unpack_252_output17_array_intq)

/* Tensor #655 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output18, AI_STATIC,
  655, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output18_array, &unpack_252_output18_array_intq)

/* Tensor #656 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output19, AI_STATIC,
  656, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output19_array, &unpack_252_output19_array_intq)

/* Tensor #657 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output2, AI_STATIC,
  657, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output2_array, &unpack_252_output2_array_intq)

/* Tensor #658 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output20, AI_STATIC,
  658, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output20_array, &unpack_252_output20_array_intq)

/* Tensor #659 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output21, AI_STATIC,
  659, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output21_array, &unpack_252_output21_array_intq)

/* Tensor #660 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output22, AI_STATIC,
  660, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output22_array, &unpack_252_output22_array_intq)

/* Tensor #661 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output23, AI_STATIC,
  661, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output23_array, &unpack_252_output23_array_intq)

/* Tensor #662 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output24, AI_STATIC,
  662, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output24_array, &unpack_252_output24_array_intq)

/* Tensor #663 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output25, AI_STATIC,
  663, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output25_array, &unpack_252_output25_array_intq)

/* Tensor #664 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output26, AI_STATIC,
  664, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output26_array, &unpack_252_output26_array_intq)

/* Tensor #665 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output27, AI_STATIC,
  665, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output27_array, &unpack_252_output27_array_intq)

/* Tensor #666 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output28, AI_STATIC,
  666, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output28_array, &unpack_252_output28_array_intq)

/* Tensor #667 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output29, AI_STATIC,
  667, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output29_array, &unpack_252_output29_array_intq)

/* Tensor #668 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output3, AI_STATIC,
  668, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output3_array, &unpack_252_output3_array_intq)

/* Tensor #669 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output30, AI_STATIC,
  669, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output30_array, &unpack_252_output30_array_intq)

/* Tensor #670 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output31, AI_STATIC,
  670, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output31_array, &unpack_252_output31_array_intq)

/* Tensor #671 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output32, AI_STATIC,
  671, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output32_array, &unpack_252_output32_array_intq)

/* Tensor #672 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output33, AI_STATIC,
  672, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output33_array, &unpack_252_output33_array_intq)

/* Tensor #673 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output34, AI_STATIC,
  673, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output34_array, &unpack_252_output34_array_intq)

/* Tensor #674 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output35, AI_STATIC,
  674, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output35_array, &unpack_252_output35_array_intq)

/* Tensor #675 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output36, AI_STATIC,
  675, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output36_array, &unpack_252_output36_array_intq)

/* Tensor #676 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output37, AI_STATIC,
  676, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output37_array, &unpack_252_output37_array_intq)

/* Tensor #677 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output38, AI_STATIC,
  677, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output38_array, &unpack_252_output38_array_intq)

/* Tensor #678 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output39, AI_STATIC,
  678, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output39_array, &unpack_252_output39_array_intq)

/* Tensor #679 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output4, AI_STATIC,
  679, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output4_array, &unpack_252_output4_array_intq)

/* Tensor #680 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output40, AI_STATIC,
  680, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output40_array, &unpack_252_output40_array_intq)

/* Tensor #681 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output41, AI_STATIC,
  681, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output41_array, &unpack_252_output41_array_intq)

/* Tensor #682 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output42, AI_STATIC,
  682, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output42_array, &unpack_252_output42_array_intq)

/* Tensor #683 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output43, AI_STATIC,
  683, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output43_array, &unpack_252_output43_array_intq)

/* Tensor #684 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output44, AI_STATIC,
  684, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output44_array, &unpack_252_output44_array_intq)

/* Tensor #685 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output45, AI_STATIC,
  685, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output45_array, &unpack_252_output45_array_intq)

/* Tensor #686 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output46, AI_STATIC,
  686, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output46_array, &unpack_252_output46_array_intq)

/* Tensor #687 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output47, AI_STATIC,
  687, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output47_array, &unpack_252_output47_array_intq)

/* Tensor #688 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output5, AI_STATIC,
  688, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output5_array, &unpack_252_output5_array_intq)

/* Tensor #689 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output6, AI_STATIC,
  689, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output6_array, &unpack_252_output6_array_intq)

/* Tensor #690 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output7, AI_STATIC,
  690, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output7_array, &unpack_252_output7_array_intq)

/* Tensor #691 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output8, AI_STATIC,
  691, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output8_array, &unpack_252_output8_array_intq)

/* Tensor #692 */
AI_TENSOR_OBJ_DECLARE(
  unpack_252_output9, AI_STATIC,
  692, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &unpack_252_output9_array, &unpack_252_output9_array_intq)

/* Tensor #693 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output0, AI_STATIC,
  693, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output0_array, &unpack_3_output0_array_intq)

/* Tensor #694 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output1, AI_STATIC,
  694, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output1_array, &unpack_3_output1_array_intq)

/* Tensor #695 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output10, AI_STATIC,
  695, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output10_array, &unpack_3_output10_array_intq)

/* Tensor #696 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output11, AI_STATIC,
  696, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output11_array, &unpack_3_output11_array_intq)

/* Tensor #697 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output12, AI_STATIC,
  697, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output12_array, &unpack_3_output12_array_intq)

/* Tensor #698 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output13, AI_STATIC,
  698, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output13_array, &unpack_3_output13_array_intq)

/* Tensor #699 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output14, AI_STATIC,
  699, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output14_array, &unpack_3_output14_array_intq)

/* Tensor #700 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output15, AI_STATIC,
  700, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output15_array, &unpack_3_output15_array_intq)

/* Tensor #701 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output16, AI_STATIC,
  701, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output16_array, &unpack_3_output16_array_intq)

/* Tensor #702 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output17, AI_STATIC,
  702, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output17_array, &unpack_3_output17_array_intq)

/* Tensor #703 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output18, AI_STATIC,
  703, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output18_array, &unpack_3_output18_array_intq)

/* Tensor #704 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output19, AI_STATIC,
  704, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output19_array, &unpack_3_output19_array_intq)

/* Tensor #705 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output2, AI_STATIC,
  705, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output2_array, &unpack_3_output2_array_intq)

/* Tensor #706 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output20, AI_STATIC,
  706, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output20_array, &unpack_3_output20_array_intq)

/* Tensor #707 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output21, AI_STATIC,
  707, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output21_array, &unpack_3_output21_array_intq)

/* Tensor #708 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output22, AI_STATIC,
  708, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output22_array, &unpack_3_output22_array_intq)

/* Tensor #709 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output23, AI_STATIC,
  709, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output23_array, &unpack_3_output23_array_intq)

/* Tensor #710 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output24, AI_STATIC,
  710, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output24_array, &unpack_3_output24_array_intq)

/* Tensor #711 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output25, AI_STATIC,
  711, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output25_array, &unpack_3_output25_array_intq)

/* Tensor #712 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output26, AI_STATIC,
  712, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output26_array, &unpack_3_output26_array_intq)

/* Tensor #713 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output27, AI_STATIC,
  713, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output27_array, &unpack_3_output27_array_intq)

/* Tensor #714 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output28, AI_STATIC,
  714, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output28_array, &unpack_3_output28_array_intq)

/* Tensor #715 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output29, AI_STATIC,
  715, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output29_array, &unpack_3_output29_array_intq)

/* Tensor #716 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output3, AI_STATIC,
  716, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output3_array, &unpack_3_output3_array_intq)

/* Tensor #717 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output30, AI_STATIC,
  717, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output30_array, &unpack_3_output30_array_intq)

/* Tensor #718 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output31, AI_STATIC,
  718, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output31_array, &unpack_3_output31_array_intq)

/* Tensor #719 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output32, AI_STATIC,
  719, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output32_array, &unpack_3_output32_array_intq)

/* Tensor #720 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output33, AI_STATIC,
  720, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output33_array, &unpack_3_output33_array_intq)

/* Tensor #721 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output34, AI_STATIC,
  721, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output34_array, &unpack_3_output34_array_intq)

/* Tensor #722 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output35, AI_STATIC,
  722, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output35_array, &unpack_3_output35_array_intq)

/* Tensor #723 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output36, AI_STATIC,
  723, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output36_array, &unpack_3_output36_array_intq)

/* Tensor #724 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output37, AI_STATIC,
  724, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output37_array, &unpack_3_output37_array_intq)

/* Tensor #725 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output38, AI_STATIC,
  725, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output38_array, &unpack_3_output38_array_intq)

/* Tensor #726 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output39, AI_STATIC,
  726, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output39_array, &unpack_3_output39_array_intq)

/* Tensor #727 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output4, AI_STATIC,
  727, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output4_array, &unpack_3_output4_array_intq)

/* Tensor #728 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output40, AI_STATIC,
  728, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output40_array, &unpack_3_output40_array_intq)

/* Tensor #729 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output41, AI_STATIC,
  729, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output41_array, &unpack_3_output41_array_intq)

/* Tensor #730 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output42, AI_STATIC,
  730, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output42_array, &unpack_3_output42_array_intq)

/* Tensor #731 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output43, AI_STATIC,
  731, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output43_array, &unpack_3_output43_array_intq)

/* Tensor #732 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output44, AI_STATIC,
  732, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output44_array, &unpack_3_output44_array_intq)

/* Tensor #733 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output45, AI_STATIC,
  733, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output45_array, &unpack_3_output45_array_intq)

/* Tensor #734 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output46, AI_STATIC,
  734, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output46_array, &unpack_3_output46_array_intq)

/* Tensor #735 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output47, AI_STATIC,
  735, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output47_array, &unpack_3_output47_array_intq)

/* Tensor #736 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output5, AI_STATIC,
  736, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output5_array, &unpack_3_output5_array_intq)

/* Tensor #737 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output6, AI_STATIC,
  737, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output6_array, &unpack_3_output6_array_intq)

/* Tensor #738 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output7, AI_STATIC,
  738, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output7_array, &unpack_3_output7_array_intq)

/* Tensor #739 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output8, AI_STATIC,
  739, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output8_array, &unpack_3_output8_array_intq)

/* Tensor #740 */
AI_TENSOR_OBJ_DECLARE(
  unpack_3_output9, AI_STATIC,
  740, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 1, 1, 14, 14),
  1, &unpack_3_output9_array, &unpack_3_output9_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_446_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_445_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_446_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_446_weights, &gemm_446_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_446_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_446_layer, 446,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_446_chain,
  NULL, &gemm_446_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_445_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_444_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_445_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_445_weights, &gemm_445_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_445_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_445_layer, 445,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_445_chain,
  NULL, &gemm_446_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_444_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -104, -102, -99, -95, -92, -88, -84, -79, -74, -69, -63, -57, -51, -44, -37, -30, -23, -15, -8, 0, 8, 15, 23, 30, 37, 44, 51, 57, 63, 69, 74, 79, 84, 88, 92, 95, 99, 102, 104, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_444_nl_params, AI_ARRAY_FORMAT_S8,
    nl_444_nl_params_data, nl_444_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_444_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_443_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_444_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_444_layer, 444,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_444_chain,
  NULL, &gemm_445_layer, AI_STATIC, 
  .nl_params = &nl_444_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_443_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_442_output, &gemm_441_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_443_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_443_layer, 443,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_443_chain,
  NULL, &nl_444_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_442_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output47),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_442_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_442_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_442_layer, 442,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_442_chain,
  NULL, &eltwise_443_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_441_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_440_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_441_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_441_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_441_layer, 441,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_441_chain,
  NULL, &gemm_442_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_440_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -104, -102, -99, -95, -92, -88, -84, -79, -74, -69, -63, -57, -51, -44, -37, -30, -23, -15, -8, 0, 8, 15, 23, 30, 37, 44, 51, 57, 63, 69, 74, 79, 84, 88, 92, 95, 99, 102, 104, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_440_nl_params, AI_ARRAY_FORMAT_S8,
    nl_440_nl_params_data, nl_440_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_440_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_439_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_440_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_440_layer, 440,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_440_chain,
  NULL, &gemm_441_layer, AI_STATIC, 
  .nl_params = &nl_440_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_439_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_438_output, &gemm_437_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_439_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_439_layer, 439,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_439_chain,
  NULL, &nl_440_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_438_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output46),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_438_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_438_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_438_layer, 438,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_438_chain,
  NULL, &eltwise_439_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_437_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_436_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_437_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_437_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_437_layer, 437,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_437_chain,
  NULL, &gemm_438_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_436_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -104, -102, -99, -95, -92, -88, -84, -79, -74, -69, -63, -57, -51, -44, -37, -30, -23, -15, -8, 0, 8, 15, 23, 30, 37, 44, 51, 57, 63, 69, 74, 79, 84, 88, 92, 95, 99, 102, 104, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_436_nl_params, AI_ARRAY_FORMAT_S8,
    nl_436_nl_params_data, nl_436_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_436_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_435_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_436_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_436_layer, 436,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_436_chain,
  NULL, &gemm_437_layer, AI_STATIC, 
  .nl_params = &nl_436_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_435_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_434_output, &gemm_433_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_435_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_435_layer, 435,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_435_chain,
  NULL, &nl_436_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_434_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output45),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_434_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_434_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_434_layer, 434,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_434_chain,
  NULL, &eltwise_435_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_433_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_432_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_433_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_433_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_433_layer, 433,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_433_chain,
  NULL, &gemm_434_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_432_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -104, -102, -99, -95, -92, -88, -84, -79, -74, -69, -63, -57, -51, -44, -37, -30, -23, -15, -8, 0, 8, 15, 23, 30, 37, 44, 51, 57, 63, 69, 74, 79, 84, 88, 92, 95, 99, 102, 104, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_432_nl_params, AI_ARRAY_FORMAT_S8,
    nl_432_nl_params_data, nl_432_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_432_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_431_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_432_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_432_layer, 432,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_432_chain,
  NULL, &gemm_433_layer, AI_STATIC, 
  .nl_params = &nl_432_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_431_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_309_output, &gemm_430_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_431_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_431_layer, 431,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_431_chain,
  NULL, &nl_432_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_309_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output44),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_309_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_309_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_309_layer, 309,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_309_chain,
  NULL, &eltwise_431_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_430_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_429_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_430_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_430_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_430_layer, 430,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_430_chain,
  NULL, &gemm_309_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_429_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -104, -102, -99, -95, -92, -88, -84, -79, -74, -69, -63, -57, -51, -44, -37, -30, -23, -15, -8, 0, 8, 15, 23, 30, 37, 44, 51, 57, 63, 69, 74, 79, 84, 88, 92, 95, 99, 102, 104, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_429_nl_params, AI_ARRAY_FORMAT_S8,
    nl_429_nl_params_data, nl_429_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_429_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_428_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_429_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_429_layer, 429,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_429_chain,
  NULL, &gemm_430_layer, AI_STATIC, 
  .nl_params = &nl_429_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_428_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_308_output, &gemm_427_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_428_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_428_layer, 428,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_428_chain,
  NULL, &nl_429_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_308_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output43),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_308_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_308_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_308_layer, 308,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_308_chain,
  NULL, &eltwise_428_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_427_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_426_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_427_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_427_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_427_layer, 427,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_427_chain,
  NULL, &gemm_308_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_426_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_426_nl_params, AI_ARRAY_FORMAT_S8,
    nl_426_nl_params_data, nl_426_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_426_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_425_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_426_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_426_layer, 426,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_426_chain,
  NULL, &gemm_427_layer, AI_STATIC, 
  .nl_params = &nl_426_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_425_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_307_output, &gemm_424_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_425_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_425_layer, 425,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_425_chain,
  NULL, &nl_426_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_307_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output42),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_307_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_307_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_307_layer, 307,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_307_chain,
  NULL, &eltwise_425_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_424_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_423_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_424_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_424_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_424_layer, 424,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_424_chain,
  NULL, &gemm_307_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_423_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_423_nl_params, AI_ARRAY_FORMAT_S8,
    nl_423_nl_params_data, nl_423_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_423_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_422_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_423_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_423_layer, 423,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_423_chain,
  NULL, &gemm_424_layer, AI_STATIC, 
  .nl_params = &nl_423_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_422_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_306_output, &gemm_421_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_422_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_422_layer, 422,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_422_chain,
  NULL, &nl_423_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_306_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output41),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_306_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_306_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_306_layer, 306,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_306_chain,
  NULL, &eltwise_422_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_421_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_420_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_421_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_421_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_421_layer, 421,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_421_chain,
  NULL, &gemm_306_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_420_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_420_nl_params, AI_ARRAY_FORMAT_S8,
    nl_420_nl_params_data, nl_420_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_420_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_419_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_420_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_420_layer, 420,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_420_chain,
  NULL, &gemm_421_layer, AI_STATIC, 
  .nl_params = &nl_420_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_419_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_305_output, &gemm_418_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_419_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_419_layer, 419,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_419_chain,
  NULL, &nl_420_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_305_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output40),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_305_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_305_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_305_layer, 305,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_305_chain,
  NULL, &eltwise_419_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_418_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_417_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_418_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_418_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_418_layer, 418,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_418_chain,
  NULL, &gemm_305_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_417_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_417_nl_params, AI_ARRAY_FORMAT_S8,
    nl_417_nl_params_data, nl_417_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_417_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_416_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_417_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_417_layer, 417,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_417_chain,
  NULL, &gemm_418_layer, AI_STATIC, 
  .nl_params = &nl_417_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_416_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_304_output, &gemm_415_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_416_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_416_layer, 416,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_416_chain,
  NULL, &nl_417_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_304_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output39),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_304_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_304_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_304_layer, 304,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_304_chain,
  NULL, &eltwise_416_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_415_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_414_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_415_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_415_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_415_layer, 415,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_415_chain,
  NULL, &gemm_304_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_414_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_414_nl_params, AI_ARRAY_FORMAT_S8,
    nl_414_nl_params_data, nl_414_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_414_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_413_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_414_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_414_layer, 414,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_414_chain,
  NULL, &gemm_415_layer, AI_STATIC, 
  .nl_params = &nl_414_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_413_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_303_output, &gemm_412_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_413_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_413_layer, 413,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_413_chain,
  NULL, &nl_414_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_303_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output38),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_303_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_303_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_303_layer, 303,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_303_chain,
  NULL, &eltwise_413_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_412_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_411_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_412_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_412_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_412_layer, 412,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_412_chain,
  NULL, &gemm_303_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_411_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_411_nl_params, AI_ARRAY_FORMAT_S8,
    nl_411_nl_params_data, nl_411_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_411_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_410_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_411_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_411_layer, 411,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_411_chain,
  NULL, &gemm_412_layer, AI_STATIC, 
  .nl_params = &nl_411_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_410_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_302_output, &gemm_409_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_410_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_410_layer, 410,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_410_chain,
  NULL, &nl_411_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_302_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output37),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_302_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_302_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_302_layer, 302,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_302_chain,
  NULL, &eltwise_410_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_409_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_408_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_409_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_409_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_409_layer, 409,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_409_chain,
  NULL, &gemm_302_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_408_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_408_nl_params, AI_ARRAY_FORMAT_S8,
    nl_408_nl_params_data, nl_408_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_408_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_407_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_408_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_408_layer, 408,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_408_chain,
  NULL, &gemm_409_layer, AI_STATIC, 
  .nl_params = &nl_408_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_407_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_301_output, &gemm_406_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_407_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_407_layer, 407,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_407_chain,
  NULL, &nl_408_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_301_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output36),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_301_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_301_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_301_layer, 301,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_301_chain,
  NULL, &eltwise_407_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_406_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_405_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_406_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_406_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_406_layer, 406,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_406_chain,
  NULL, &gemm_301_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_405_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_405_nl_params, AI_ARRAY_FORMAT_S8,
    nl_405_nl_params_data, nl_405_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_405_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_404_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_405_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_405_layer, 405,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_405_chain,
  NULL, &gemm_406_layer, AI_STATIC, 
  .nl_params = &nl_405_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_404_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_300_output, &gemm_403_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_404_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_404_layer, 404,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_404_chain,
  NULL, &nl_405_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_300_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output35),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_300_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_300_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_300_layer, 300,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_300_chain,
  NULL, &eltwise_404_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_403_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_402_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_403_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_403_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_403_layer, 403,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_403_chain,
  NULL, &gemm_300_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_402_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_402_nl_params, AI_ARRAY_FORMAT_S8,
    nl_402_nl_params_data, nl_402_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_402_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_401_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_402_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_402_layer, 402,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_402_chain,
  NULL, &gemm_403_layer, AI_STATIC, 
  .nl_params = &nl_402_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_401_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_295_output, &gemm_400_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_401_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_401_layer, 401,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_401_chain,
  NULL, &nl_402_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_295_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output34),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_295_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_295_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_295_layer, 295,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_295_chain,
  NULL, &eltwise_401_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_400_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_399_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_400_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_400_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_400_layer, 400,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_400_chain,
  NULL, &gemm_295_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_399_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_399_nl_params, AI_ARRAY_FORMAT_S8,
    nl_399_nl_params_data, nl_399_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_399_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_398_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_399_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_399_layer, 399,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_399_chain,
  NULL, &gemm_400_layer, AI_STATIC, 
  .nl_params = &nl_399_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_398_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_294_output, &gemm_397_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_398_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_398_layer, 398,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_398_chain,
  NULL, &nl_399_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_294_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output33),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_294_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_294_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_294_layer, 294,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_294_chain,
  NULL, &eltwise_398_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_397_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_396_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_397_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_397_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_397_layer, 397,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_397_chain,
  NULL, &gemm_294_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_396_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -117, -116, -115, -113, -111, -109, -107, -105, -102, -99, -96, -92, -88, -84, -79, -75, -69, -64, -58, -51, -45, -38, -30, -23, -15, -8, 0, 8, 15, 23, 30, 38, 45, 51, 58, 64, 69, 75, 79, 84, 88, 92, 96, 99, 102, 105, 107, 109, 111, 113, 115, 116, 117, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_396_nl_params, AI_ARRAY_FORMAT_S8,
    nl_396_nl_params_data, nl_396_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_396_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_395_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_396_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_396_layer, 396,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_396_chain,
  NULL, &gemm_397_layer, AI_STATIC, 
  .nl_params = &nl_396_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_395_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_293_output, &gemm_394_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_395_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_395_layer, 395,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_395_chain,
  NULL, &nl_396_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_293_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output32),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_293_layer, 293,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_293_chain,
  NULL, &eltwise_395_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_394_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_393_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_394_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_394_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_394_layer, 394,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_394_chain,
  NULL, &gemm_293_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_393_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -124, -124, -124, -123, -122, -122, -121, -120, -119, -118, -116, -115, -113, -112, -110, -107, -105, -102, -99, -96, -93, -89, -84, -80, -75, -70, -64, -58, -52, -45, -38, -31, -23, -16, -8, 0, 8, 16, 23, 31, 38, 45, 52, 58, 64, 70, 75, 80, 84, 89, 93, 96, 99, 102, 105, 107, 110, 112, 113, 115, 116, 118, 119, 120, 121, 122, 122, 123, 124, 124, 124, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_393_nl_params, AI_ARRAY_FORMAT_S8,
    nl_393_nl_params_data, nl_393_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_393_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_392_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_393_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_393_layer, 393,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_393_chain,
  NULL, &gemm_394_layer, AI_STATIC, 
  .nl_params = &nl_393_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_392_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_292_output, &gemm_391_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_392_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_392_layer, 392,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_392_chain,
  NULL, &nl_393_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_292_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output31),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_292_layer, 292,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_292_chain,
  NULL, &eltwise_392_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_391_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_390_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_391_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_391_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_391_layer, 391,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_391_chain,
  NULL, &gemm_292_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_390_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -115, -114, -112, -110, -108, -106, -103, -100, -97, -93, -89, -85, -81, -76, -70, -65, -59, -52, -45, -38, -31, -23, -16, -8, 0, 8, 16, 23, 31, 38, 45, 52, 59, 65, 70, 76, 81, 85, 89, 93, 97, 100, 103, 106, 108, 110, 112, 114, 115, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_390_nl_params, AI_ARRAY_FORMAT_S8,
    nl_390_nl_params_data, nl_390_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_390_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_389_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_390_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_390_layer, 390,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_390_chain,
  NULL, &gemm_391_layer, AI_STATIC, 
  .nl_params = &nl_390_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_389_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_291_output, &gemm_388_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_389_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_389_layer, 389,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_389_chain,
  NULL, &nl_390_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_291_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output30),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_291_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_291_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_291_layer, 291,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_291_chain,
  NULL, &eltwise_389_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_388_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_387_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_388_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_388_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_388_layer, 388,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_388_chain,
  NULL, &gemm_291_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_387_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -115, -114, -112, -110, -108, -106, -103, -100, -97, -93, -89, -85, -81, -76, -70, -65, -59, -52, -45, -38, -31, -23, -16, -8, 0, 8, 16, 23, 31, 38, 45, 52, 59, 65, 70, 76, 81, 85, 89, 93, 97, 100, 103, 106, 108, 110, 112, 114, 115, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_387_nl_params, AI_ARRAY_FORMAT_S8,
    nl_387_nl_params_data, nl_387_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_387_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_386_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_387_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_387_layer, 387,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_387_chain,
  NULL, &gemm_388_layer, AI_STATIC, 
  .nl_params = &nl_387_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_386_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_290_output, &gemm_385_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_386_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_386_layer, 386,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_386_chain,
  NULL, &nl_387_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_290_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output29),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_290_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_290_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_290_layer, 290,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_290_chain,
  NULL, &eltwise_386_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_385_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_384_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_385_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_385_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_385_layer, 385,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_385_chain,
  NULL, &gemm_290_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_384_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -115, -114, -112, -110, -108, -106, -103, -100, -97, -93, -89, -85, -81, -76, -70, -65, -59, -52, -45, -38, -31, -23, -16, -8, 0, 8, 16, 23, 31, 38, 45, 52, 59, 65, 70, 76, 81, 85, 89, 93, 97, 100, 103, 106, 108, 110, 112, 114, 115, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_384_nl_params, AI_ARRAY_FORMAT_S8,
    nl_384_nl_params_data, nl_384_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_384_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_383_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_384_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_384_layer, 384,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_384_chain,
  NULL, &gemm_385_layer, AI_STATIC, 
  .nl_params = &nl_384_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_383_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_289_output, &gemm_382_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_383_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_383_layer, 383,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_383_chain,
  NULL, &nl_384_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_289_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output28),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_289_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_289_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_289_layer, 289,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_289_chain,
  NULL, &eltwise_383_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_382_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_381_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_382_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_382_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_382_layer, 382,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_382_chain,
  NULL, &gemm_289_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_381_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -115, -114, -112, -110, -108, -106, -103, -100, -97, -93, -89, -85, -81, -76, -70, -65, -59, -52, -45, -38, -31, -23, -16, -8, 0, 8, 16, 23, 31, 38, 45, 52, 59, 65, 70, 76, 81, 85, 89, 93, 97, 100, 103, 106, 108, 110, 112, 114, 115, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_381_nl_params, AI_ARRAY_FORMAT_S8,
    nl_381_nl_params_data, nl_381_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_381_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_380_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_381_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_381_layer, 381,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_381_chain,
  NULL, &gemm_382_layer, AI_STATIC, 
  .nl_params = &nl_381_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_380_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_288_output, &gemm_379_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_380_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_380_layer, 380,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_380_chain,
  NULL, &nl_381_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_288_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output27),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_288_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_288_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_288_layer, 288,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_288_chain,
  NULL, &eltwise_380_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_379_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_378_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_379_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_379_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_379_layer, 379,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_379_chain,
  NULL, &gemm_288_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_378_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -115, -114, -112, -110, -108, -105, -103, -100, -97, -93, -89, -85, -80, -75, -70, -64, -58, -52, -45, -38, -31, -23, -16, -8, 0, 8, 16, 23, 31, 38, 45, 52, 58, 64, 70, 75, 80, 85, 89, 93, 97, 100, 103, 105, 108, 110, 112, 114, 115, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_378_nl_params, AI_ARRAY_FORMAT_S8,
    nl_378_nl_params_data, nl_378_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_378_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_377_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_378_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_378_layer, 378,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_378_chain,
  NULL, &gemm_379_layer, AI_STATIC, 
  .nl_params = &nl_378_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_377_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_287_output, &gemm_376_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_377_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_377_layer, 377,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_377_chain,
  NULL, &nl_378_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_287_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output26),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_287_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_287_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_287_layer, 287,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_287_chain,
  NULL, &eltwise_377_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_376_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_375_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_376_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_376_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_376_layer, 376,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_376_chain,
  NULL, &gemm_287_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_375_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -104, -101, -98, -95, -91, -88, -83, -79, -74, -69, -63, -57, -51, -44, -37, -30, -23, -15, -8, 0, 8, 15, 23, 30, 37, 44, 51, 57, 63, 69, 74, 79, 83, 88, 91, 95, 98, 101, 104, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_375_nl_params, AI_ARRAY_FORMAT_S8,
    nl_375_nl_params_data, nl_375_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_375_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_374_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_375_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_375_layer, 375,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_375_chain,
  NULL, &gemm_376_layer, AI_STATIC, 
  .nl_params = &nl_375_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_374_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_286_output, &gemm_373_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_374_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_374_layer, 374,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_374_chain,
  NULL, &nl_375_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_286_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output25),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_286_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_286_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_286_layer, 286,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_286_chain,
  NULL, &eltwise_374_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_373_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_372_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_373_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_373_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_373_layer, 373,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_373_chain,
  NULL, &gemm_286_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_372_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -118, -116, -115, -113, -112, -110, -108, -106, -103, -100, -97, -94, -90, -86, -82, -78, -73, -67, -62, -56, -50, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 50, 56, 62, 67, 73, 78, 82, 86, 90, 94, 97, 100, 103, 106, 108, 110, 112, 113, 115, 116, 118, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_372_nl_params, AI_ARRAY_FORMAT_S8,
    nl_372_nl_params_data, nl_372_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_372_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_371_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_372_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_372_layer, 372,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_372_chain,
  NULL, &gemm_373_layer, AI_STATIC, 
  .nl_params = &nl_372_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_371_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_281_output, &gemm_370_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_371_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_371_layer, 371,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_371_chain,
  NULL, &nl_372_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_281_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output24),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_281_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_281_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_281_layer, 281,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_281_chain,
  NULL, &eltwise_371_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_370_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_369_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_370_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_370_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_370_layer, 370,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_370_chain,
  NULL, &gemm_281_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_369_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -102, -100, -97, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 97, 100, 102, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_369_nl_params, AI_ARRAY_FORMAT_S8,
    nl_369_nl_params_data, nl_369_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_369_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_368_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_369_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_369_layer, 369,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_369_chain,
  NULL, &gemm_370_layer, AI_STATIC, 
  .nl_params = &nl_369_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_368_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_280_output, &gemm_367_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_368_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_368_layer, 368,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_368_chain,
  NULL, &nl_369_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_280_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output23),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_280_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_280_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_280_layer, 280,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_280_chain,
  NULL, &eltwise_368_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_367_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_366_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_367_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_367_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_367_layer, 367,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_367_chain,
  NULL, &gemm_280_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_366_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -105, -102, -99, -96, -93, -89, -85, -81, -76, -72, -66, -61, -55, -49, -42, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 42, 49, 55, 61, 66, 72, 76, 81, 85, 89, 93, 96, 99, 102, 105, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_366_nl_params, AI_ARRAY_FORMAT_S8,
    nl_366_nl_params_data, nl_366_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_366_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_365_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_366_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_366_layer, 366,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_366_chain,
  NULL, &gemm_367_layer, AI_STATIC, 
  .nl_params = &nl_366_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_365_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_279_output, &gemm_364_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_365_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_365_layer, 365,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_365_chain,
  NULL, &nl_366_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_279_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output22),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_279_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_279_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_279_layer, 279,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_279_chain,
  NULL, &eltwise_365_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_364_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_363_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_364_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_364_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_364_layer, 364,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_364_chain,
  NULL, &gemm_279_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_363_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -105, -102, -99, -96, -93, -89, -85, -81, -76, -72, -66, -61, -55, -49, -42, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 42, 49, 55, 61, 66, 72, 76, 81, 85, 89, 93, 96, 99, 102, 105, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_363_nl_params, AI_ARRAY_FORMAT_S8,
    nl_363_nl_params_data, nl_363_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_363_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_362_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_363_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_363_layer, 363,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_363_chain,
  NULL, &gemm_364_layer, AI_STATIC, 
  .nl_params = &nl_363_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_362_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_278_output, &gemm_361_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_362_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_362_layer, 362,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_362_chain,
  NULL, &nl_363_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_278_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output21),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_278_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_278_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_278_layer, 278,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_278_chain,
  NULL, &eltwise_362_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_361_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_360_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_361_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_361_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_361_layer, 361,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_361_chain,
  NULL, &gemm_278_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_360_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -105, -102, -99, -96, -93, -89, -85, -81, -76, -72, -66, -61, -55, -49, -42, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 42, 49, 55, 61, 66, 72, 76, 81, 85, 89, 93, 96, 99, 102, 105, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_360_nl_params, AI_ARRAY_FORMAT_S8,
    nl_360_nl_params_data, nl_360_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_360_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_359_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_360_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_360_layer, 360,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_360_chain,
  NULL, &gemm_361_layer, AI_STATIC, 
  .nl_params = &nl_360_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_359_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_277_output, &gemm_358_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_359_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_359_layer, 359,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_359_chain,
  NULL, &nl_360_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_277_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output20),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_277_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_277_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_277_layer, 277,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_277_chain,
  NULL, &eltwise_359_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_358_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_357_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_358_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_358_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_358_layer, 358,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_358_chain,
  NULL, &gemm_277_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_357_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -105, -102, -99, -96, -93, -89, -85, -81, -76, -72, -66, -61, -55, -49, -42, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 42, 49, 55, 61, 66, 72, 76, 81, 85, 89, 93, 96, 99, 102, 105, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_357_nl_params, AI_ARRAY_FORMAT_S8,
    nl_357_nl_params_data, nl_357_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_357_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_356_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_357_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_357_layer, 357,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_357_chain,
  NULL, &gemm_358_layer, AI_STATIC, 
  .nl_params = &nl_357_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_356_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_276_output, &gemm_355_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_356_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_356_layer, 356,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_356_chain,
  NULL, &nl_357_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_276_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output19),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_276_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_276_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_276_layer, 276,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_276_chain,
  NULL, &eltwise_356_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_355_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_354_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_355_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_355_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_355_layer, 355,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_355_chain,
  NULL, &gemm_276_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_354_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -124, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -102, -100, -97, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 97, 100, 102, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 124, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_354_nl_params, AI_ARRAY_FORMAT_S8,
    nl_354_nl_params_data, nl_354_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_354_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_353_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_354_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_354_layer, 354,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_354_chain,
  NULL, &gemm_355_layer, AI_STATIC, 
  .nl_params = &nl_354_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_353_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_275_output, &gemm_352_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_353_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_353_layer, 353,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_353_chain,
  NULL, &nl_354_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_275_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output18),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_275_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_275_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_275_layer, 275,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_275_chain,
  NULL, &eltwise_353_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_352_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_351_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_352_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_352_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_352_layer, 352,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_352_chain,
  NULL, &gemm_275_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_351_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -102, -100, -97, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 97, 100, 102, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_351_nl_params, AI_ARRAY_FORMAT_S8,
    nl_351_nl_params_data, nl_351_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_351_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_350_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_351_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_351_layer, 351,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_351_chain,
  NULL, &gemm_352_layer, AI_STATIC, 
  .nl_params = &nl_351_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_350_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_274_output, &gemm_349_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_350_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_350_layer, 350,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_350_chain,
  NULL, &nl_351_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_274_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output17),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_274_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_274_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_274_layer, 274,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_274_chain,
  NULL, &eltwise_350_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_349_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_348_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_349_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_349_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_349_layer, 349,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_349_chain,
  NULL, &gemm_274_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_348_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -124, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -102, -100, -97, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 97, 100, 102, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 124, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_348_nl_params, AI_ARRAY_FORMAT_S8,
    nl_348_nl_params_data, nl_348_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_348_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_347_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_348_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_348_layer, 348,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_348_chain,
  NULL, &gemm_349_layer, AI_STATIC, 
  .nl_params = &nl_348_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_347_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_273_output, &gemm_346_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_347_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_347_layer, 347,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_347_chain,
  NULL, &nl_348_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_273_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output16),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_273_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_273_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_273_layer, 273,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_273_chain,
  NULL, &eltwise_347_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_346_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_345_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_346_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_346_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_346_layer, 346,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_346_chain,
  NULL, &gemm_273_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_345_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -124, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -102, -100, -97, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 97, 100, 102, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 124, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_345_nl_params, AI_ARRAY_FORMAT_S8,
    nl_345_nl_params_data, nl_345_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_345_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_344_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_345_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_345_layer, 345,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_345_chain,
  NULL, &gemm_346_layer, AI_STATIC, 
  .nl_params = &nl_345_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_344_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_272_output, &gemm_343_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_344_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_344_layer, 344,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_344_chain,
  NULL, &nl_345_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_272_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output15),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_272_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_272_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_272_layer, 272,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_272_chain,
  NULL, &eltwise_344_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_343_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_342_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_343_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_343_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_343_layer, 343,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_343_chain,
  NULL, &gemm_272_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_342_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -124, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -105, -102, -100, -96, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 96, 100, 102, 105, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 124, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_342_nl_params, AI_ARRAY_FORMAT_S8,
    nl_342_nl_params_data, nl_342_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_342_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_341_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_342_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_342_layer, 342,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_342_chain,
  NULL, &gemm_343_layer, AI_STATIC, 
  .nl_params = &nl_342_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_341_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_267_output, &gemm_340_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_341_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_341_layer, 341,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_341_chain,
  NULL, &nl_342_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_267_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output14),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_267_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_267_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_267_layer, 267,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_267_chain,
  NULL, &eltwise_341_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_340_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_339_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_340_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_340_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_340_layer, 340,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_340_chain,
  NULL, &gemm_267_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_339_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -124, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -102, -100, -97, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 97, 100, 102, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 124, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_339_nl_params, AI_ARRAY_FORMAT_S8,
    nl_339_nl_params_data, nl_339_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_339_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_338_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_339_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_339_layer, 339,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_339_chain,
  NULL, &gemm_340_layer, AI_STATIC, 
  .nl_params = &nl_339_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_338_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_266_output, &gemm_337_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_338_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_338_layer, 338,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_338_chain,
  NULL, &nl_339_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_266_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output13),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_266_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_266_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_266_layer, 266,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_266_chain,
  NULL, &eltwise_338_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_337_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_336_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_337_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_337_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_337_layer, 337,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_337_chain,
  NULL, &gemm_266_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_336_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -124, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -102, -100, -97, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 97, 100, 102, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 124, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_336_nl_params, AI_ARRAY_FORMAT_S8,
    nl_336_nl_params_data, nl_336_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_336_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_335_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_336_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_336_layer, 336,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_336_chain,
  NULL, &gemm_337_layer, AI_STATIC, 
  .nl_params = &nl_336_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_335_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_265_output, &gemm_334_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_335_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_335_layer, 335,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_335_chain,
  NULL, &nl_336_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_265_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output12),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_265_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_265_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_265_layer, 265,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_265_chain,
  NULL, &eltwise_335_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_334_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_333_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_334_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_334_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_334_layer, 334,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_334_chain,
  NULL, &gemm_265_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_333_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -102, -100, -97, -93, -90, -86, -81, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 81, 86, 90, 93, 97, 100, 102, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_333_nl_params, AI_ARRAY_FORMAT_S8,
    nl_333_nl_params_data, nl_333_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_333_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_332_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_333_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_333_layer, 333,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_333_chain,
  NULL, &gemm_334_layer, AI_STATIC, 
  .nl_params = &nl_333_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_332_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_264_output, &gemm_331_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_332_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_332_layer, 332,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_332_chain,
  NULL, &nl_333_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_264_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output11),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_264_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_264_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_264_layer, 264,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_264_chain,
  NULL, &eltwise_332_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_331_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_330_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_331_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_331_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_331_layer, 331,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_331_chain,
  NULL, &gemm_264_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_330_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -115, -113, -111, -109, -107, -105, -103, -100, -97, -93, -90, -86, -82, -77, -72, -67, -61, -55, -49, -43, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 43, 49, 55, 61, 67, 72, 77, 82, 86, 90, 93, 97, 100, 103, 105, 107, 109, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_330_nl_params, AI_ARRAY_FORMAT_S8,
    nl_330_nl_params_data, nl_330_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_330_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_329_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_330_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_330_layer, 330,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_330_chain,
  NULL, &gemm_331_layer, AI_STATIC, 
  .nl_params = &nl_330_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_329_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_263_output, &gemm_328_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_329_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_329_layer, 329,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_329_chain,
  NULL, &nl_330_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_263_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output10),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_263_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_263_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_263_layer, 263,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_263_chain,
  NULL, &eltwise_329_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_328_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_327_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_328_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_328_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_328_layer, 328,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_328_chain,
  NULL, &gemm_263_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_327_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -121, -120, -119, -118, -117, -115, -114, -112, -111, -109, -107, -104, -102, -99, -96, -92, -89, -85, -81, -76, -71, -66, -60, -55, -48, -42, -35, -29, -22, -15, -7, 0, 7, 15, 22, 29, 35, 42, 48, 55, 60, 66, 71, 76, 81, 85, 89, 92, 96, 99, 102, 104, 107, 109, 111, 112, 114, 115, 117, 118, 119, 120, 121, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_327_nl_params, AI_ARRAY_FORMAT_S8,
    nl_327_nl_params_data, nl_327_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_327_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_326_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_327_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_327_layer, 327,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_327_chain,
  NULL, &gemm_328_layer, AI_STATIC, 
  .nl_params = &nl_327_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_326_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_262_output, &gemm_325_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_326_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_326_layer, 326,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_326_chain,
  NULL, &nl_327_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_262_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output9),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_262_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_262_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_262_layer, 262,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_262_chain,
  NULL, &eltwise_326_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_325_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_324_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_325_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_325_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_325_layer, 325,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_325_chain,
  NULL, &gemm_262_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_324_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -105, -102, -99, -96, -93, -89, -85, -81, -77, -72, -66, -61, -55, -49, -42, -36, -29, -22, -15, -7, 0, 7, 15, 22, 29, 36, 42, 49, 55, 61, 66, 72, 77, 81, 85, 89, 93, 96, 99, 102, 105, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_324_nl_params, AI_ARRAY_FORMAT_S8,
    nl_324_nl_params_data, nl_324_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_324_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_323_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_324_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_324_layer, 324,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_324_chain,
  NULL, &gemm_325_layer, AI_STATIC, 
  .nl_params = &nl_324_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_323_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_261_output, &gemm_322_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_323_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_323_layer, 323,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_323_chain,
  NULL, &nl_324_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_261_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output8),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_261_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_261_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_261_layer, 261,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_261_chain,
  NULL, &eltwise_323_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_322_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_321_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_322_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_322_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_322_layer, 322,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_322_chain,
  NULL, &gemm_261_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_321_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -120, -119, -118, -117, -115, -114, -112, -110, -108, -106, -104, -101, -98, -94, -91, -87, -83, -78, -73, -68, -62, -56, -50, -44, -37, -30, -22, -15, -8, 0, 8, 15, 22, 30, 37, 44, 50, 56, 62, 68, 73, 78, 83, 87, 91, 94, 98, 101, 104, 106, 108, 110, 112, 114, 115, 117, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_321_nl_params, AI_ARRAY_FORMAT_S8,
    nl_321_nl_params_data, nl_321_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_321_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_320_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_321_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_321_layer, 321,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_321_chain,
  NULL, &gemm_322_layer, AI_STATIC, 
  .nl_params = &nl_321_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_320_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_260_output, &gemm_319_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_320_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_320_layer, 320,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_320_chain,
  NULL, &nl_321_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_260_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output7),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_260_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_260_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_260_layer, 260,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_260_chain,
  NULL, &eltwise_320_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_319_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_318_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_319_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_319_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_319_layer, 319,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_319_chain,
  NULL, &gemm_260_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_318_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -122, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -106, -104, -101, -98, -95, -91, -87, -83, -79, -74, -68, -63, -57, -51, -44, -37, -30, -23, -15, -8, 0, 8, 15, 23, 30, 37, 44, 51, 57, 63, 68, 74, 79, 83, 87, 91, 95, 98, 101, 104, 106, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_318_nl_params, AI_ARRAY_FORMAT_S8,
    nl_318_nl_params_data, nl_318_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_318_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_317_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_318_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_318_layer, 318,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_318_chain,
  NULL, &gemm_319_layer, AI_STATIC, 
  .nl_params = &nl_318_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_317_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_259_output, &gemm_316_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_317_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_317_layer, 317,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_317_chain,
  NULL, &nl_318_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_259_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output6),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_259_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_259_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_259_layer, 259,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_259_chain,
  NULL, &eltwise_317_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_316_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_315_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_316_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_316_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_316_layer, 316,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_316_chain,
  NULL, &gemm_259_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_315_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -122, -122, -121, -120, -119, -118, -116, -115, -113, -112, -110, -108, -105, -102, -100, -96, -93, -89, -85, -80, -75, -70, -64, -58, -52, -45, -38, -31, -23, -16, -8, 0, 8, 16, 23, 31, 38, 45, 52, 58, 64, 70, 75, 80, 85, 89, 93, 96, 100, 102, 105, 108, 110, 112, 113, 115, 116, 118, 119, 120, 121, 122, 122, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_315_nl_params, AI_ARRAY_FORMAT_S8,
    nl_315_nl_params_data, nl_315_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_315_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_314_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_315_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_315_layer, 315,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_315_chain,
  NULL, &gemm_316_layer, AI_STATIC, 
  .nl_params = &nl_315_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_314_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_258_output, &gemm_313_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_314_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_314_layer, 314,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_314_chain,
  NULL, &nl_315_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_258_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output5),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_258_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_258_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_258_layer, 258,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_258_chain,
  NULL, &eltwise_314_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_313_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_312_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_313_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_313_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_313_layer, 313,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_313_chain,
  NULL, &gemm_258_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_312_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -114, -112, -110, -108, -106, -103, -100, -97, -94, -90, -85, -81, -76, -71, -65, -59, -52, -46, -38, -31, -24, -16, -8, 0, 8, 16, 24, 31, 38, 46, 52, 59, 65, 71, 76, 81, 85, 90, 94, 97, 100, 103, 106, 108, 110, 112, 114, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_312_nl_params, AI_ARRAY_FORMAT_S8,
    nl_312_nl_params_data, nl_312_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_312_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_311_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_312_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_312_layer, 312,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_312_chain,
  NULL, &gemm_313_layer, AI_STATIC, 
  .nl_params = &nl_312_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_311_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_310_output, &gemm_299_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_311_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_311_layer, 311,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_311_chain,
  NULL, &nl_312_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_310_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output4),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_310_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_310_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_310_layer, 310,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_310_chain,
  NULL, &eltwise_311_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_299_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_298_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_299_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_299_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_299_layer, 299,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_299_chain,
  NULL, &gemm_310_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_298_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -121, -120, -119, -118, -117, -116, -114, -113, -111, -109, -107, -104, -101, -98, -95, -92, -88, -83, -79, -74, -69, -63, -57, -51, -44, -37, -30, -23, -15, -8, 0, 8, 15, 23, 30, 37, 44, 51, 57, 63, 69, 74, 79, 83, 88, 92, 95, 98, 101, 104, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_298_nl_params, AI_ARRAY_FORMAT_S8,
    nl_298_nl_params_data, nl_298_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_298_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_297_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_298_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_298_layer, 298,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_298_chain,
  NULL, &gemm_299_layer, AI_STATIC, 
  .nl_params = &nl_298_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_297_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_296_output, &gemm_285_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_297_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_297_layer, 297,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_297_chain,
  NULL, &nl_298_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_296_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output3),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_296_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_296_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_296_layer, 296,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_296_chain,
  NULL, &eltwise_297_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_285_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_284_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_285_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_285_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_285_layer, 285,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_285_chain,
  NULL, &gemm_296_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_284_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -98, -95, -92, -89, -86, -82, -78, -74, -69, -65, -60, -55, -49, -44, -38, -32, -26, -19, -13, -6, 0, 6, 13, 19, 26, 32, 38, 44, 49, 55, 60, 65, 69, 74, 78, 82, 86, 89, 92, 95, 98, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_284_nl_params, AI_ARRAY_FORMAT_S8,
    nl_284_nl_params_data, nl_284_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_284_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_283_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_284_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_284_layer, 284,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_284_chain,
  NULL, &gemm_285_layer, AI_STATIC, 
  .nl_params = &nl_284_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_283_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_282_output, &gemm_271_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_283_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_283_layer, 283,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_283_chain,
  NULL, &nl_284_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_282_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output2),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_282_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_282_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_282_layer, 282,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_282_chain,
  NULL, &eltwise_283_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_271_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_270_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_271_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_271_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_271_layer, 271,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_271_chain,
  NULL, &gemm_282_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_270_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -113, -112, -110, -108, -107, -105, -103, -101, -98, -96, -93, -90, -87, -84, -80, -77, -73, -69, -64, -60, -55, -50, -45, -40, -35, -29, -23, -18, -12, -6, 0, 6, 12, 18, 23, 29, 35, 40, 45, 50, 55, 60, 64, 69, 73, 77, 80, 84, 87, 90, 93, 96, 98, 101, 103, 105, 107, 108, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_270_nl_params, AI_ARRAY_FORMAT_S8,
    nl_270_nl_params_data, nl_270_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_270_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_269_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_270_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_270_layer, 270,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_270_chain,
  NULL, &gemm_271_layer, AI_STATIC, 
  .nl_params = &nl_270_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_269_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_268_output, &gemm_257_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_269_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_269_layer, 269,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_269_chain,
  NULL, &nl_270_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_268_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_268_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_268_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_268_layer, 268,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_268_chain,
  NULL, &eltwise_269_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_257_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_256_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_257_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_257_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_257_layer, 257,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_257_chain,
  NULL, &gemm_268_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_256_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -123, -122, -122, -121, -121, -120, -120, -119, -119, -118, -118, -117, -116, -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -105, -104, -103, -101, -99, -98, -96, -94, -92, -90, -88, -86, -83, -81, -78, -76, -73, -70, -67, -64, -61, -58, -54, -51, -47, -44, -40, -36, -32, -29, -25, -21, -16, -12, -8, -4, 0, 4, 8, 12, 16, 21, 25, 29, 32, 36, 40, 44, 47, 51, 54, 58, 61, 64, 67, 70, 73, 76, 78, 81, 83, 86, 88, 90, 92, 94, 96, 98, 99, 101, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 116, 117, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_256_nl_params, AI_ARRAY_FORMAT_S8,
    nl_256_nl_params_data, nl_256_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_256_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_255_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_256_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_256_layer, 256,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_256_chain,
  NULL, &gemm_257_layer, AI_STATIC, 
  .nl_params = &nl_256_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_255_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_output, &gemm_253_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_255_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_255_layer, 255,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_255_chain,
  NULL, &nl_256_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_254_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_252_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_254_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_254_weights, &gemm_254_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_254_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_254_layer, 254,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_254_chain,
  NULL, &eltwise_255_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  unpack_252_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pack_246_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 48, &unpack_252_output0, &unpack_252_output1, &unpack_252_output2, &unpack_252_output3, &unpack_252_output4, &unpack_252_output5, &unpack_252_output6, &unpack_252_output7, &unpack_252_output8, &unpack_252_output9, &unpack_252_output10, &unpack_252_output11, &unpack_252_output12, &unpack_252_output13, &unpack_252_output14, &unpack_252_output15, &unpack_252_output16, &unpack_252_output17, &unpack_252_output18, &unpack_252_output19, &unpack_252_output20, &unpack_252_output21, &unpack_252_output22, &unpack_252_output23, &unpack_252_output24, &unpack_252_output25, &unpack_252_output26, &unpack_252_output27, &unpack_252_output28, &unpack_252_output29, &unpack_252_output30, &unpack_252_output31, &unpack_252_output32, &unpack_252_output33, &unpack_252_output34, &unpack_252_output35, &unpack_252_output36, &unpack_252_output37, &unpack_252_output38, &unpack_252_output39, &unpack_252_output40, &unpack_252_output41, &unpack_252_output42, &unpack_252_output43, &unpack_252_output44, &unpack_252_output45, &unpack_252_output46, &unpack_252_output47),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  unpack_252_layer, 252,
  UNPACK_TYPE, 0x0, NULL,
  unpack, forward_unpack,
  &unpack_252_chain,
  NULL, &gemm_254_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pack_246_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 48, &conversion_10_output, &conversion_25_output, &conversion_40_output, &conversion_55_output, &conversion_70_output, &conversion_74_output, &conversion_78_output, &conversion_82_output, &conversion_86_output, &conversion_90_output, &conversion_94_output, &conversion_98_output, &conversion_102_output, &conversion_106_output, &conversion_110_output, &conversion_114_output, &conversion_118_output, &conversion_122_output, &conversion_126_output, &conversion_130_output, &conversion_134_output, &conversion_138_output, &conversion_142_output, &conversion_146_output, &conversion_150_output, &conversion_154_output, &conversion_158_output, &conversion_162_output, &conversion_166_output, &conversion_170_output, &conversion_174_output, &conversion_178_output, &conversion_182_output, &conversion_186_output, &conversion_190_output, &conversion_194_output, &conversion_198_output, &conversion_202_output, &conversion_206_output, &conversion_210_output, &conversion_214_output, &conversion_218_output, &conversion_222_output, &conversion_226_output, &conversion_230_output, &conversion_235_output, &conversion_240_output, &conversion_245_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pack_246_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pack_246_layer, 246,
  PACK_TYPE, 0x0, NULL,
  pack, forward_pack,
  &pack_246_chain,
  NULL, &unpack_252_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_245_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_244_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_245_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_245_layer, 245,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_245_chain,
  NULL, &pack_246_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_244_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_244_nl_params, AI_ARRAY_FORMAT_S8,
    nl_244_nl_params_data, nl_244_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_244_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_243_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_244_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_244_layer, 244,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_244_chain,
  NULL, &conversion_245_layer, AI_STATIC, 
  .nl_params = &nl_244_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_243_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_242_output, &gemm_241_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_243_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_243_layer, 243,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_243_chain,
  NULL, &nl_244_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_242_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output47),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_242_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_242_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_242_layer, 242,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_242_chain,
  NULL, &eltwise_243_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_241_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_240_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_241_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_241_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_241_layer, 241,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_241_chain,
  NULL, &gemm_242_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_240_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_239_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_240_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_240_layer, 240,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_240_chain,
  NULL, &gemm_241_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_239_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_239_nl_params, AI_ARRAY_FORMAT_S8,
    nl_239_nl_params_data, nl_239_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_239_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_238_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_239_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_239_layer, 239,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_239_chain,
  NULL, &conversion_240_layer, AI_STATIC, 
  .nl_params = &nl_239_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_238_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_237_output, &gemm_236_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_238_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_238_layer, 238,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_238_chain,
  NULL, &nl_239_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_237_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output46),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_237_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_237_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_237_layer, 237,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_237_chain,
  NULL, &eltwise_238_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_236_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_235_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_236_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_236_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_236_layer, 236,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_236_chain,
  NULL, &gemm_237_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_235_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_234_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_235_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_235_layer, 235,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_235_chain,
  NULL, &gemm_236_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_234_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_234_nl_params, AI_ARRAY_FORMAT_S8,
    nl_234_nl_params_data, nl_234_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_234_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_233_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_234_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_234_layer, 234,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_234_chain,
  NULL, &conversion_235_layer, AI_STATIC, 
  .nl_params = &nl_234_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_233_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_232_output, &gemm_231_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_233_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_233_layer, 233,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_233_chain,
  NULL, &nl_234_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_232_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output45),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_232_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_232_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_232_layer, 232,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_232_chain,
  NULL, &eltwise_233_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_231_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_230_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_231_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_231_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_231_layer, 231,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_231_chain,
  NULL, &gemm_232_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_230_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_229_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_230_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_230_layer, 230,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_230_chain,
  NULL, &gemm_231_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_229_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_229_nl_params, AI_ARRAY_FORMAT_S8,
    nl_229_nl_params_data, nl_229_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_229_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_228_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_229_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_229_layer, 229,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_229_chain,
  NULL, &conversion_230_layer, AI_STATIC, 
  .nl_params = &nl_229_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_228_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_66_output, &gemm_227_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_228_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_228_layer, 228,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_228_chain,
  NULL, &nl_229_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_66_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output44),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_66_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_66_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_66_layer, 66,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_66_chain,
  NULL, &eltwise_228_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_227_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_226_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_227_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_227_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_227_layer, 227,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_227_chain,
  NULL, &gemm_66_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_226_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_225_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_226_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_226_layer, 226,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_226_chain,
  NULL, &gemm_227_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_225_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_225_nl_params, AI_ARRAY_FORMAT_S8,
    nl_225_nl_params_data, nl_225_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_225_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_224_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_225_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_225_layer, 225,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_225_chain,
  NULL, &conversion_226_layer, AI_STATIC, 
  .nl_params = &nl_225_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_224_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_65_output, &gemm_223_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_224_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_224_layer, 224,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_224_chain,
  NULL, &nl_225_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output43),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_65_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_65_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_65_layer, 65,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_65_chain,
  NULL, &eltwise_224_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_223_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_222_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_223_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_223_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_223_layer, 223,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_223_chain,
  NULL, &gemm_65_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_222_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_221_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_222_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_222_layer, 222,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_222_chain,
  NULL, &gemm_223_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_221_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_221_nl_params, AI_ARRAY_FORMAT_S8,
    nl_221_nl_params_data, nl_221_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_221_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_220_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_221_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_221_layer, 221,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_221_chain,
  NULL, &conversion_222_layer, AI_STATIC, 
  .nl_params = &nl_221_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_220_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_64_output, &gemm_219_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_220_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_220_layer, 220,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_220_chain,
  NULL, &nl_221_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_64_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output42),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_64_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_64_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_64_layer, 64,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_64_chain,
  NULL, &eltwise_220_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_219_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_218_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_219_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_219_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_219_layer, 219,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_219_chain,
  NULL, &gemm_64_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_218_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_217_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_218_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_218_layer, 218,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_218_chain,
  NULL, &gemm_219_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_217_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_217_nl_params, AI_ARRAY_FORMAT_S8,
    nl_217_nl_params_data, nl_217_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_217_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_216_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_217_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_217_layer, 217,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_217_chain,
  NULL, &conversion_218_layer, AI_STATIC, 
  .nl_params = &nl_217_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_216_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_63_output, &gemm_215_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_216_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_216_layer, 216,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_216_chain,
  NULL, &nl_217_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_63_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output41),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_63_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_63_layer, 63,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_63_chain,
  NULL, &eltwise_216_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_215_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_214_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_215_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_215_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_215_layer, 215,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_215_chain,
  NULL, &gemm_63_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_214_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_213_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_214_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_214_layer, 214,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_214_chain,
  NULL, &gemm_215_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_213_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_213_nl_params, AI_ARRAY_FORMAT_S8,
    nl_213_nl_params_data, nl_213_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_213_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_212_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_213_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_213_layer, 213,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_213_chain,
  NULL, &conversion_214_layer, AI_STATIC, 
  .nl_params = &nl_213_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_212_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_62_output, &gemm_211_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_212_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_212_layer, 212,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_212_chain,
  NULL, &nl_213_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_62_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output40),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_62_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_62_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_62_layer, 62,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_62_chain,
  NULL, &eltwise_212_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_211_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_210_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_211_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_211_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_211_layer, 211,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_211_chain,
  NULL, &gemm_62_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_210_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_209_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_210_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_210_layer, 210,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_210_chain,
  NULL, &gemm_211_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_209_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_209_nl_params, AI_ARRAY_FORMAT_S8,
    nl_209_nl_params_data, nl_209_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_209_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_208_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_209_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_209_layer, 209,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_209_chain,
  NULL, &conversion_210_layer, AI_STATIC, 
  .nl_params = &nl_209_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_208_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_61_output, &gemm_207_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_208_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_208_layer, 208,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_208_chain,
  NULL, &nl_209_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_61_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output39),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_61_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_61_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_61_layer, 61,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_61_chain,
  NULL, &eltwise_208_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_207_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_206_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_207_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_207_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_207_layer, 207,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_207_chain,
  NULL, &gemm_61_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_206_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_205_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_206_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_206_layer, 206,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_206_chain,
  NULL, &gemm_207_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_205_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_205_nl_params, AI_ARRAY_FORMAT_S8,
    nl_205_nl_params_data, nl_205_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_205_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_204_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_205_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_205_layer, 205,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_205_chain,
  NULL, &conversion_206_layer, AI_STATIC, 
  .nl_params = &nl_205_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_204_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_60_output, &gemm_203_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_204_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_204_layer, 204,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_204_chain,
  NULL, &nl_205_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_60_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output38),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_60_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_60_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_60_layer, 60,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_60_chain,
  NULL, &eltwise_204_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_203_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_202_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_203_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_203_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_203_layer, 203,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_203_chain,
  NULL, &gemm_60_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_202_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_201_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_202_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_202_layer, 202,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_202_chain,
  NULL, &gemm_203_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_201_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_201_nl_params, AI_ARRAY_FORMAT_S8,
    nl_201_nl_params_data, nl_201_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_201_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_200_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_201_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_201_layer, 201,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_201_chain,
  NULL, &conversion_202_layer, AI_STATIC, 
  .nl_params = &nl_201_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_200_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_59_output, &gemm_199_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_200_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_200_layer, 200,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_200_chain,
  NULL, &nl_201_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_59_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output37),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_59_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_59_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_59_layer, 59,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_59_chain,
  NULL, &eltwise_200_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_199_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_198_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_199_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_199_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_199_layer, 199,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_199_chain,
  NULL, &gemm_59_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_198_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_197_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_198_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_198_layer, 198,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_198_chain,
  NULL, &gemm_199_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_197_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_197_nl_params, AI_ARRAY_FORMAT_S8,
    nl_197_nl_params_data, nl_197_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_197_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_196_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_197_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_197_layer, 197,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_197_chain,
  NULL, &conversion_198_layer, AI_STATIC, 
  .nl_params = &nl_197_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_196_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_58_output, &gemm_195_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_196_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_196_layer, 196,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_196_chain,
  NULL, &nl_197_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output36),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_58_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_58_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_58_layer, 58,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_58_chain,
  NULL, &eltwise_196_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_195_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_194_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_195_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_195_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_195_layer, 195,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_195_chain,
  NULL, &gemm_58_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_194_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_193_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_194_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_194_layer, 194,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_194_chain,
  NULL, &gemm_195_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_193_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_193_nl_params, AI_ARRAY_FORMAT_S8,
    nl_193_nl_params_data, nl_193_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_193_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_192_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_193_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_193_layer, 193,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_193_chain,
  NULL, &conversion_194_layer, AI_STATIC, 
  .nl_params = &nl_193_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_192_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_57_output, &gemm_191_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_192_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_192_layer, 192,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_192_chain,
  NULL, &nl_193_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_57_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output35),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_57_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_57_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_57_layer, 57,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_57_chain,
  NULL, &eltwise_192_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_191_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_190_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_191_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_191_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_191_layer, 191,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_191_chain,
  NULL, &gemm_57_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_190_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_189_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_190_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_190_layer, 190,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_190_chain,
  NULL, &gemm_191_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_189_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_189_nl_params, AI_ARRAY_FORMAT_S8,
    nl_189_nl_params_data, nl_189_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_189_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_188_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_189_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_189_layer, 189,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_189_chain,
  NULL, &conversion_190_layer, AI_STATIC, 
  .nl_params = &nl_189_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_188_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_51_output, &gemm_187_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_188_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_188_layer, 188,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_188_chain,
  NULL, &nl_189_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output34),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_51_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_51_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_51_layer, 51,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_51_chain,
  NULL, &eltwise_188_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_187_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_186_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_187_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_187_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_187_layer, 187,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_187_chain,
  NULL, &gemm_51_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_186_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_185_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_186_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_186_layer, 186,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_186_chain,
  NULL, &gemm_187_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_185_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -69, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 69, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_185_nl_params, AI_ARRAY_FORMAT_S8,
    nl_185_nl_params_data, nl_185_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_185_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_184_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_185_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_185_layer, 185,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_185_chain,
  NULL, &conversion_186_layer, AI_STATIC, 
  .nl_params = &nl_185_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_184_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_50_output, &gemm_183_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_184_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_184_layer, 184,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_184_chain,
  NULL, &nl_185_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output33),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_50_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_50_layer, 50,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_50_chain,
  NULL, &eltwise_184_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_183_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_182_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_183_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_183_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_183_layer, 183,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_183_chain,
  NULL, &gemm_50_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_182_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_181_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_182_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_182_layer, 182,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_182_chain,
  NULL, &gemm_183_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_181_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -69, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 69, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_181_nl_params, AI_ARRAY_FORMAT_S8,
    nl_181_nl_params_data, nl_181_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_181_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_180_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_181_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_181_layer, 181,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_181_chain,
  NULL, &conversion_182_layer, AI_STATIC, 
  .nl_params = &nl_181_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_180_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_49_output, &gemm_179_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_180_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_180_layer, 180,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_180_chain,
  NULL, &nl_181_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output32),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_49_layer, 49,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_49_chain,
  NULL, &eltwise_180_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_179_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_178_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_179_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_179_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_179_layer, 179,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_179_chain,
  NULL, &gemm_49_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_178_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_177_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_178_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_178_layer, 178,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_178_chain,
  NULL, &gemm_179_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_177_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_177_nl_params, AI_ARRAY_FORMAT_S8,
    nl_177_nl_params_data, nl_177_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_177_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_176_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_177_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_177_layer, 177,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_177_chain,
  NULL, &conversion_178_layer, AI_STATIC, 
  .nl_params = &nl_177_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_176_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_48_output, &gemm_175_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_176_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_176_layer, 176,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_176_chain,
  NULL, &nl_177_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output31),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_48_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_48_layer, 48,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_48_chain,
  NULL, &eltwise_176_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_175_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_174_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_175_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_175_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_175_layer, 175,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_175_chain,
  NULL, &gemm_48_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_174_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_173_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_174_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_174_layer, 174,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_174_chain,
  NULL, &gemm_175_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_173_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_173_nl_params, AI_ARRAY_FORMAT_S8,
    nl_173_nl_params_data, nl_173_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_173_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_172_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_173_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_173_layer, 173,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_173_chain,
  NULL, &conversion_174_layer, AI_STATIC, 
  .nl_params = &nl_173_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_172_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_47_output, &gemm_171_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_172_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_172_layer, 172,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_172_chain,
  NULL, &nl_173_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output30),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_47_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_47_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_47_layer, 47,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_47_chain,
  NULL, &eltwise_172_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_171_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_170_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_171_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_171_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_171_layer, 171,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_171_chain,
  NULL, &gemm_47_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_170_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_169_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_170_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_170_layer, 170,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_170_chain,
  NULL, &gemm_171_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_169_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -48, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 48, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_169_nl_params, AI_ARRAY_FORMAT_S8,
    nl_169_nl_params_data, nl_169_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_169_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_168_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_169_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_169_layer, 169,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_169_chain,
  NULL, &conversion_170_layer, AI_STATIC, 
  .nl_params = &nl_169_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_168_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_46_output, &gemm_167_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_168_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_168_layer, 168,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_168_chain,
  NULL, &nl_169_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output29),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_46_layer, 46,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_46_chain,
  NULL, &eltwise_168_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_167_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_166_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_167_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_167_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_167_layer, 167,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_167_chain,
  NULL, &gemm_46_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_166_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_165_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_166_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_166_layer, 166,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_166_chain,
  NULL, &gemm_167_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_165_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -48, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 48, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_165_nl_params, AI_ARRAY_FORMAT_S8,
    nl_165_nl_params_data, nl_165_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_165_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_164_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_165_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_165_layer, 165,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_165_chain,
  NULL, &conversion_166_layer, AI_STATIC, 
  .nl_params = &nl_165_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_164_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_45_output, &gemm_163_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_164_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_164_layer, 164,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_164_chain,
  NULL, &nl_165_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output28),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_45_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_45_layer, 45,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_45_chain,
  NULL, &eltwise_164_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_163_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_162_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_163_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_163_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_163_layer, 163,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_163_chain,
  NULL, &gemm_45_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_162_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_161_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_162_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_162_layer, 162,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_162_chain,
  NULL, &gemm_163_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_161_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_161_nl_params, AI_ARRAY_FORMAT_S8,
    nl_161_nl_params_data, nl_161_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_161_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_160_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_161_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_161_layer, 161,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_161_chain,
  NULL, &conversion_162_layer, AI_STATIC, 
  .nl_params = &nl_161_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_160_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_44_output, &gemm_159_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_160_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_160_layer, 160,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_160_chain,
  NULL, &nl_161_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_44_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output27),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_44_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_44_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_44_layer, 44,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_44_chain,
  NULL, &eltwise_160_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_159_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_158_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_159_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_159_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_159_layer, 159,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_159_chain,
  NULL, &gemm_44_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_158_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_157_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_158_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_158_layer, 158,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_158_chain,
  NULL, &gemm_159_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_157_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -100, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 100, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_157_nl_params, AI_ARRAY_FORMAT_S8,
    nl_157_nl_params_data, nl_157_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_157_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_156_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_157_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_157_layer, 157,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_157_chain,
  NULL, &conversion_158_layer, AI_STATIC, 
  .nl_params = &nl_157_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_156_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_43_output, &gemm_155_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_156_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_156_layer, 156,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_156_chain,
  NULL, &nl_157_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output26),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_43_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_43_layer, 43,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_43_chain,
  NULL, &eltwise_156_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_155_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_154_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_155_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_155_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_155_layer, 155,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_155_chain,
  NULL, &gemm_43_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_154_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_153_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_154_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_154_layer, 154,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_154_chain,
  NULL, &gemm_155_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_153_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -100, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 100, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_153_nl_params, AI_ARRAY_FORMAT_S8,
    nl_153_nl_params_data, nl_153_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_153_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_152_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_153_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_153_layer, 153,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_153_chain,
  NULL, &conversion_154_layer, AI_STATIC, 
  .nl_params = &nl_153_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_152_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_42_output, &gemm_151_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_152_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_152_layer, 152,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_152_chain,
  NULL, &nl_153_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output25),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_42_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_42_layer, 42,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_42_chain,
  NULL, &eltwise_152_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_151_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_150_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_151_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_151_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_151_layer, 151,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_151_chain,
  NULL, &gemm_42_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_150_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_149_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_150_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_150_layer, 150,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_150_chain,
  NULL, &gemm_151_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_149_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_149_nl_params, AI_ARRAY_FORMAT_S8,
    nl_149_nl_params_data, nl_149_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_149_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_148_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_149_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_149_layer, 149,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_149_chain,
  NULL, &conversion_150_layer, AI_STATIC, 
  .nl_params = &nl_149_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_148_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_36_output, &gemm_147_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_148_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_148_layer, 148,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_148_chain,
  NULL, &nl_149_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output24),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_36_layer, 36,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_36_chain,
  NULL, &eltwise_148_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_147_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_146_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_147_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_147_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_147_layer, 147,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_147_chain,
  NULL, &gemm_36_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_146_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_145_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_146_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_146_layer, 146,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_146_chain,
  NULL, &gemm_147_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_145_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_145_nl_params, AI_ARRAY_FORMAT_S8,
    nl_145_nl_params_data, nl_145_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_145_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_144_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_145_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_145_layer, 145,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_145_chain,
  NULL, &conversion_146_layer, AI_STATIC, 
  .nl_params = &nl_145_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_144_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_35_output, &gemm_143_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_144_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_144_layer, 144,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_144_chain,
  NULL, &nl_145_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output23),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_35_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_35_layer, 35,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_35_chain,
  NULL, &eltwise_144_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_143_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_142_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_143_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_143_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_143_layer, 143,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_143_chain,
  NULL, &gemm_35_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_142_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_141_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_142_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_142_layer, 142,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_142_chain,
  NULL, &gemm_143_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_141_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_141_nl_params, AI_ARRAY_FORMAT_S8,
    nl_141_nl_params_data, nl_141_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_141_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_140_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_141_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_141_layer, 141,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_141_chain,
  NULL, &conversion_142_layer, AI_STATIC, 
  .nl_params = &nl_141_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_140_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_34_output, &gemm_139_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_140_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_140_layer, 140,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_140_chain,
  NULL, &nl_141_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output22),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_34_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_34_layer, 34,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_34_chain,
  NULL, &eltwise_140_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_139_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_138_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_139_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_139_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_139_layer, 139,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_139_chain,
  NULL, &gemm_34_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_138_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_137_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_138_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_138_layer, 138,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_138_chain,
  NULL, &gemm_139_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_137_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_137_nl_params, AI_ARRAY_FORMAT_S8,
    nl_137_nl_params_data, nl_137_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_137_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_136_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_137_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_137_layer, 137,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_137_chain,
  NULL, &conversion_138_layer, AI_STATIC, 
  .nl_params = &nl_137_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_136_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_33_output, &gemm_135_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_136_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_136_layer, 136,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_136_chain,
  NULL, &nl_137_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output21),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_33_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_33_layer, 33,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_33_chain,
  NULL, &eltwise_136_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_135_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_134_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_135_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_135_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_135_layer, 135,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_135_chain,
  NULL, &gemm_33_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_134_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_133_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_134_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_134_layer, 134,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_134_chain,
  NULL, &gemm_135_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_133_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_133_nl_params, AI_ARRAY_FORMAT_S8,
    nl_133_nl_params_data, nl_133_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_133_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_133_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_133_layer, 133,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_133_chain,
  NULL, &conversion_134_layer, AI_STATIC, 
  .nl_params = &nl_133_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_132_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_32_output, &gemm_131_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_132_layer, 132,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_132_chain,
  NULL, &nl_133_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output20),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_32_layer, 32,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_32_chain,
  NULL, &eltwise_132_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_131_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_130_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_131_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_131_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_131_layer, 131,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_131_chain,
  NULL, &gemm_32_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_130_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_129_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_130_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_130_layer, 130,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_130_chain,
  NULL, &gemm_131_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_129_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -75, -72, -69, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 69, 72, 75, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_129_nl_params, AI_ARRAY_FORMAT_S8,
    nl_129_nl_params_data, nl_129_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_129_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_128_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_129_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_129_layer, 129,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_129_chain,
  NULL, &conversion_130_layer, AI_STATIC, 
  .nl_params = &nl_129_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_128_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_31_output, &gemm_127_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_128_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_128_layer, 128,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_128_chain,
  NULL, &nl_129_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output19),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_31_layer, 31,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_31_chain,
  NULL, &eltwise_128_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_127_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_126_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_127_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_127_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_127_layer, 127,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_127_chain,
  NULL, &gemm_31_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_126_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_125_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_126_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_126_layer, 126,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_126_chain,
  NULL, &gemm_127_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_125_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -122, -121, -120, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -87, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 87, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 120, 121, 122, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_125_nl_params, AI_ARRAY_FORMAT_S8,
    nl_125_nl_params_data, nl_125_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_125_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_124_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_125_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_125_layer, 125,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_125_chain,
  NULL, &conversion_126_layer, AI_STATIC, 
  .nl_params = &nl_125_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_124_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_30_output, &gemm_123_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_124_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_124_layer, 124,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_124_chain,
  NULL, &nl_125_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output18),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_30_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_30_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_30_layer, 30,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_30_chain,
  NULL, &eltwise_124_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_123_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_122_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_123_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_123_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_123_layer, 123,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_123_chain,
  NULL, &gemm_30_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_122_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_121_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_122_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_122_layer, 122,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_122_chain,
  NULL, &gemm_123_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_121_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -122, -121, -120, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -87, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 87, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 120, 121, 122, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_121_nl_params, AI_ARRAY_FORMAT_S8,
    nl_121_nl_params_data, nl_121_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_121_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_120_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_121_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_121_layer, 121,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_121_chain,
  NULL, &conversion_122_layer, AI_STATIC, 
  .nl_params = &nl_121_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_120_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_29_output, &gemm_119_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_120_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_120_layer, 120,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_120_chain,
  NULL, &nl_121_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output17),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_29_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_29_layer, 29,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_29_chain,
  NULL, &eltwise_120_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_119_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_118_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_119_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_119_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_119_layer, 119,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_119_chain,
  NULL, &gemm_29_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_118_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_117_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_118_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_118_layer, 118,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_118_chain,
  NULL, &gemm_119_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_117_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -87, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 87, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_117_nl_params, AI_ARRAY_FORMAT_S8,
    nl_117_nl_params_data, nl_117_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_117_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_116_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_117_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_117_layer, 117,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_117_chain,
  NULL, &conversion_118_layer, AI_STATIC, 
  .nl_params = &nl_117_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_116_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_28_output, &gemm_115_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_116_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_116_layer, 116,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_116_chain,
  NULL, &nl_117_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output16),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_28_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_28_layer, 28,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_28_chain,
  NULL, &eltwise_116_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_115_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_114_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_115_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_115_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_115_layer, 115,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_115_chain,
  NULL, &gemm_28_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_114_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_113_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_114_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_114_layer, 114,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_114_chain,
  NULL, &gemm_115_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_113_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -103, -101, -99, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 99, 101, 103, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_113_nl_params, AI_ARRAY_FORMAT_S8,
    nl_113_nl_params_data, nl_113_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_113_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_112_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_113_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_113_layer, 113,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_113_chain,
  NULL, &conversion_114_layer, AI_STATIC, 
  .nl_params = &nl_113_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_112_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_27_output, &gemm_111_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_112_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_112_layer, 112,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_112_chain,
  NULL, &nl_113_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output15),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_27_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_27_layer, 27,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_27_chain,
  NULL, &eltwise_112_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_111_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_110_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_111_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_111_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_111_layer, 111,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_111_chain,
  NULL, &gemm_27_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_110_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_109_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_110_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_110_layer, 110,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_110_chain,
  NULL, &gemm_111_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_109_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -102, -100, -98, -95, -93, -91, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 91, 93, 95, 98, 100, 102, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_109_nl_params, AI_ARRAY_FORMAT_S8,
    nl_109_nl_params_data, nl_109_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_109_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_108_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_109_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_109_layer, 109,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_109_chain,
  NULL, &conversion_110_layer, AI_STATIC, 
  .nl_params = &nl_109_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_108_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_21_output, &gemm_107_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_108_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_108_layer, 108,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_108_chain,
  NULL, &nl_109_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output14),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_21_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_21_layer, 21,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_21_chain,
  NULL, &eltwise_108_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_107_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_106_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_107_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_107_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_107_layer, 107,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_107_chain,
  NULL, &gemm_21_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_106_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_105_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_106_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_106_layer, 106,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_106_chain,
  NULL, &gemm_107_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_105_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -102, -100, -98, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 98, 100, 102, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_105_nl_params, AI_ARRAY_FORMAT_S8,
    nl_105_nl_params_data, nl_105_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_105_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_104_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_105_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_105_layer, 105,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_105_chain,
  NULL, &conversion_106_layer, AI_STATIC, 
  .nl_params = &nl_105_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_104_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_20_output, &gemm_103_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_104_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_104_layer, 104,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_104_chain,
  NULL, &nl_105_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output13),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_20_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_20_layer, 20,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_20_chain,
  NULL, &eltwise_104_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_103_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_102_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_103_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_103_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_103_layer, 103,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_103_chain,
  NULL, &gemm_20_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_102_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_101_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_102_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_102_layer, 102,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_102_chain,
  NULL, &gemm_103_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_101_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -108, -107, -105, -103, -101, -100, -97, -95, -93, -90, -88, -85, -82, -79, -76, -72, -69, -65, -61, -57, -53, -49, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 49, 53, 57, 61, 65, 69, 72, 76, 79, 82, 85, 88, 90, 93, 95, 97, 100, 101, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_101_nl_params, AI_ARRAY_FORMAT_S8,
    nl_101_nl_params_data, nl_101_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_101_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_100_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_101_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_101_layer, 101,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_101_chain,
  NULL, &conversion_102_layer, AI_STATIC, 
  .nl_params = &nl_101_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_100_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_19_output, &gemm_99_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_100_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_100_layer, 100,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_100_chain,
  NULL, &nl_101_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output12),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_19_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_19_layer, 19,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_19_chain,
  NULL, &eltwise_100_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_99_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_98_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_99_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_99_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_99_layer, 99,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_99_chain,
  NULL, &gemm_19_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_98_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_97_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_98_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_98_layer, 98,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_98_chain,
  NULL, &gemm_99_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_97_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -122, -122, -121, -121, -120, -120, -119, -118, -118, -117, -116, -115, -114, -113, -112, -110, -109, -108, -106, -105, -103, -101, -99, -97, -95, -92, -90, -87, -84, -81, -78, -75, -72, -68, -64, -61, -57, -52, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 52, 57, 61, 64, 68, 72, 75, 78, 81, 84, 87, 90, 92, 95, 97, 99, 101, 103, 105, 106, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 118, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_97_nl_params, AI_ARRAY_FORMAT_S8,
    nl_97_nl_params_data, nl_97_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_97_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_96_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_97_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_97_layer, 97,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_97_chain,
  NULL, &conversion_98_layer, AI_STATIC, 
  .nl_params = &nl_97_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_96_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_18_output, &gemm_95_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_96_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_96_layer, 96,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_96_chain,
  NULL, &nl_97_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output11),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_18_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_18_layer, 18,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_18_chain,
  NULL, &eltwise_96_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_95_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_94_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_95_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_95_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_95_layer, 95,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_95_chain,
  NULL, &gemm_18_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_94_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_94_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_94_layer, 94,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_94_chain,
  NULL, &gemm_95_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_93_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -122, -122, -121, -121, -120, -120, -119, -118, -118, -117, -116, -115, -114, -113, -112, -110, -109, -108, -106, -105, -103, -101, -99, -97, -95, -92, -90, -87, -84, -81, -78, -75, -72, -68, -64, -61, -57, -52, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 52, 57, 61, 64, 68, 72, 75, 78, 81, 84, 87, 90, 92, 95, 97, 99, 101, 103, 105, 106, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 118, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_93_nl_params, AI_ARRAY_FORMAT_S8,
    nl_93_nl_params_data, nl_93_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_93_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_92_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_93_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_93_layer, 93,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_93_chain,
  NULL, &conversion_94_layer, AI_STATIC, 
  .nl_params = &nl_93_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_92_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_17_output, &gemm_91_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_92_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_92_layer, 92,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_92_chain,
  NULL, &nl_93_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output10),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_17_layer, 17,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_17_chain,
  NULL, &eltwise_92_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_91_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_90_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_91_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_91_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_91_layer, 91,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_91_chain,
  NULL, &gemm_17_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_90_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_89_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_90_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_90_layer, 90,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_90_chain,
  NULL, &gemm_91_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_89_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -122, -121, -120, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -109, -108, -106, -105, -103, -101, -99, -97, -95, -93, -90, -87, -85, -82, -79, -75, -72, -68, -65, -61, -57, -53, -48, -44, -39, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 48, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 87, 90, 93, 95, 97, 99, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 120, 121, 122, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_89_nl_params, AI_ARRAY_FORMAT_S8,
    nl_89_nl_params_data, nl_89_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_89_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_89_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_89_layer, 89,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_89_chain,
  NULL, &conversion_90_layer, AI_STATIC, 
  .nl_params = &nl_89_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_88_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_16_output, &gemm_87_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_88_layer, 88,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_88_chain,
  NULL, &nl_89_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output9),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_16_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_16_chain,
  NULL, &eltwise_88_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_87_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_86_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_87_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_87_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_87_layer, 87,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_87_chain,
  NULL, &gemm_16_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_86_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_86_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_86_layer, 86,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_86_chain,
  NULL, &gemm_87_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_85_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -121, -120, -120, -119, -118, -118, -117, -116, -115, -114, -113, -111, -110, -109, -107, -106, -104, -102, -100, -98, -96, -94, -91, -88, -86, -83, -80, -76, -73, -69, -66, -62, -58, -53, -49, -45, -40, -35, -30, -26, -21, -15, -10, -5, 0, 5, 10, 15, 21, 26, 30, 35, 40, 45, 49, 53, 58, 62, 66, 69, 73, 76, 80, 83, 86, 88, 91, 94, 96, 98, 100, 102, 104, 106, 107, 109, 110, 111, 113, 114, 115, 116, 117, 118, 118, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_85_nl_params, AI_ARRAY_FORMAT_S8,
    nl_85_nl_params_data, nl_85_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_85_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_84_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_85_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_85_layer, 85,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_85_chain,
  NULL, &conversion_86_layer, AI_STATIC, 
  .nl_params = &nl_85_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_84_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_15_output, &gemm_83_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_84_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_84_layer, 84,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_84_chain,
  NULL, &nl_85_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output8),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_15_layer, 15,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_15_chain,
  NULL, &eltwise_84_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_82_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_83_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_83_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_83_layer, 83,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_83_chain,
  NULL, &gemm_15_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_82_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_82_layer, 82,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_82_chain,
  NULL, &gemm_83_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_81_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -120, -120, -119, -118, -118, -117, -116, -115, -114, -113, -111, -110, -109, -107, -105, -104, -102, -100, -97, -95, -93, -90, -87, -84, -81, -78, -75, -71, -67, -63, -59, -55, -50, -46, -41, -36, -31, -26, -21, -16, -11, -5, 0, 5, 11, 16, 21, 26, 31, 36, 41, 46, 50, 55, 59, 63, 67, 71, 75, 78, 81, 84, 87, 90, 93, 95, 97, 100, 102, 104, 105, 107, 109, 110, 111, 113, 114, 115, 116, 117, 118, 118, 119, 120, 120, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_81_nl_params, AI_ARRAY_FORMAT_S8,
    nl_81_nl_params_data, nl_81_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_81_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_81_layer, 81,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_81_chain,
  NULL, &conversion_82_layer, AI_STATIC, 
  .nl_params = &nl_81_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_80_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_14_output, &gemm_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_80_layer, 80,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_80_chain,
  NULL, &nl_81_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output7),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_14_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_14_chain,
  NULL, &eltwise_80_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_78_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_79_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_79_layer, 79,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_79_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_78_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_77_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_78_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_78_layer, 78,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_78_chain,
  NULL, &gemm_79_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_77_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -122, -121, -120, -120, -119, -118, -118, -117, -116, -115, -114, -112, -111, -110, -108, -107, -105, -103, -101, -99, -97, -95, -92, -90, -87, -84, -81, -78, -74, -71, -67, -63, -59, -55, -50, -46, -41, -36, -31, -26, -21, -16, -11, -5, 0, 5, 11, 16, 21, 26, 31, 36, 41, 46, 50, 55, 59, 63, 67, 71, 74, 78, 81, 84, 87, 90, 92, 95, 97, 99, 101, 103, 105, 107, 108, 110, 111, 112, 114, 115, 116, 117, 118, 118, 119, 120, 120, 121, 122, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_77_nl_params, AI_ARRAY_FORMAT_S8,
    nl_77_nl_params_data, nl_77_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_77_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_76_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_77_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_77_layer, 77,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_77_chain,
  NULL, &conversion_78_layer, AI_STATIC, 
  .nl_params = &nl_77_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_76_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_13_output, &gemm_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_76_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_76_layer, 76,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_76_chain,
  NULL, &nl_77_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output6),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_13_layer, 13,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_13_chain,
  NULL, &eltwise_76_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_75_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_74_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_75_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_75_layer, 75,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_75_chain,
  NULL, &gemm_13_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_74_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_73_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_74_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_74_layer, 74,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_74_chain,
  NULL, &gemm_75_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_73_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -120, -119, -118, -117, -117, -116, -115, -114, -113, -112, -110, -109, -108, -106, -104, -103, -101, -99, -97, -94, -92, -90, -87, -84, -81, -78, -75, -71, -68, -64, -60, -56, -52, -48, -44, -39, -34, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 34, 39, 44, 48, 52, 56, 60, 64, 68, 71, 75, 78, 81, 84, 87, 90, 92, 94, 97, 99, 101, 103, 104, 106, 108, 109, 110, 112, 113, 114, 115, 116, 117, 117, 118, 119, 120, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_73_nl_params, AI_ARRAY_FORMAT_S8,
    nl_73_nl_params_data, nl_73_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_73_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_72_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_73_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_73_layer, 73,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_73_chain,
  NULL, &conversion_74_layer, AI_STATIC, 
  .nl_params = &nl_73_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_72_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_12_output, &gemm_71_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_72_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_72_layer, 72,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_72_chain,
  NULL, &nl_73_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output5),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_12_layer, 12,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_12_chain,
  NULL, &eltwise_72_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_71_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_70_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_71_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_71_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_71_layer, 71,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_71_chain,
  NULL, &gemm_12_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_70_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_70_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_70_layer, 70,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_70_chain,
  NULL, &gemm_71_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_69_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -120, -119, -118, -117, -117, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -104, -102, -100, -98, -96, -94, -92, -89, -87, -84, -81, -78, -75, -72, -69, -65, -62, -58, -54, -50, -46, -42, -37, -33, -28, -24, -19, -14, -10, -5, 0, 5, 10, 14, 19, 24, 28, 33, 37, 42, 46, 50, 54, 58, 62, 65, 69, 72, 75, 78, 81, 84, 87, 89, 92, 94, 96, 98, 100, 102, 104, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 117, 118, 119, 120, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_69_nl_params, AI_ARRAY_FORMAT_S8,
    nl_69_nl_params_data, nl_69_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_69_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_68_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_69_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_69_layer, 69,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_69_chain,
  NULL, &conversion_70_layer, AI_STATIC, 
  .nl_params = &nl_69_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_68_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_67_output, &gemm_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_68_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_68_layer, 68,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_68_chain,
  NULL, &nl_69_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_67_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output4),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_67_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_67_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_67_layer, 67,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_67_chain,
  NULL, &eltwise_68_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_56_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_56_layer, 56,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_56_chain,
  NULL, &gemm_67_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_55_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_55_layer, 55,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_55_chain,
  NULL, &gemm_56_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_54_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -120, -119, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -109, -107, -106, -104, -102, -101, -99, -97, -95, -92, -90, -87, -85, -82, -79, -76, -73, -69, -66, -62, -58, -55, -50, -46, -42, -38, -33, -29, -24, -19, -14, -10, -5, 0, 5, 10, 14, 19, 24, 29, 33, 38, 42, 46, 50, 55, 58, 62, 66, 69, 73, 76, 79, 82, 85, 87, 90, 92, 95, 97, 99, 101, 102, 104, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_54_nl_params, AI_ARRAY_FORMAT_S8,
    nl_54_nl_params_data, nl_54_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_54_layer, 54,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_54_chain,
  NULL, &conversion_55_layer, AI_STATIC, 
  .nl_params = &nl_54_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_52_output, &gemm_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_53_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_53_layer, 53,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_53_chain,
  NULL, &nl_54_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output3),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_52_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_52_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_52_layer, 52,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_52_chain,
  NULL, &eltwise_53_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_41_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_40_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_41_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_41_layer, 41,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_41_chain,
  NULL, &gemm_52_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_40_layer, 40,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_40_chain,
  NULL, &gemm_41_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_39_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -120, -119, -119, -118, -117, -116, -116, -115, -114, -113, -111, -110, -109, -108, -106, -105, -103, -101, -100, -98, -96, -93, -91, -89, -86, -84, -81, -78, -75, -72, -68, -65, -61, -57, -54, -50, -45, -41, -37, -33, -28, -23, -19, -14, -9, -5, 0, 5, 9, 14, 19, 23, 28, 33, 37, 41, 45, 50, 54, 57, 61, 65, 68, 72, 75, 78, 81, 84, 86, 89, 91, 93, 96, 98, 100, 101, 103, 105, 106, 108, 109, 110, 111, 113, 114, 115, 116, 116, 117, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_39_nl_params, AI_ARRAY_FORMAT_S8,
    nl_39_nl_params_data, nl_39_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_39_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_39_layer, 39,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_39_chain,
  NULL, &conversion_40_layer, AI_STATIC, 
  .nl_params = &nl_39_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_37_output, &gemm_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_38_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_38_layer, 38,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_38_chain,
  NULL, &nl_39_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output2),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_37_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_37_layer, 37,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_37_chain,
  NULL, &eltwise_38_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_26_layer, 26,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_26_chain,
  NULL, &gemm_37_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_25_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_25_chain,
  NULL, &gemm_26_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_24_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -122, -122, -122, -121, -121, -120, -120, -119, -119, -118, -118, -117, -116, -115, -115, -114, -113, -112, -111, -110, -109, -108, -107, -105, -104, -103, -101, -100, -98, -96, -94, -93, -91, -89, -86, -84, -82, -79, -77, -74, -71, -69, -66, -63, -60, -56, -53, -50, -46, -43, -39, -35, -32, -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32, 35, 39, 43, 46, 50, 53, 56, 60, 63, 66, 69, 71, 74, 77, 79, 82, 84, 86, 89, 91, 93, 94, 96, 98, 100, 101, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 115, 116, 117, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_24_nl_params, AI_ARRAY_FORMAT_S8,
    nl_24_nl_params_data, nl_24_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_24_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_24_chain,
  NULL, &conversion_25_layer, AI_STATIC, 
  .nl_params = &nl_24_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_22_output, &gemm_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_23_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_23_layer, 23,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_23_chain,
  NULL, &nl_24_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_22_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_22_layer, 22,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_22_chain,
  NULL, &eltwise_23_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_11_layer, 11,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_11_chain,
  NULL, &gemm_22_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_10_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_10_chain,
  NULL, &gemm_11_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_9_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -123, -122, -122, -122, -121, -121, -121, -120, -120, -120, -119, -119, -118, -118, -117, -116, -116, -115, -115, -114, -113, -112, -112, -111, -110, -109, -108, -107, -106, -105, -104, -102, -101, -100, -98, -97, -96, -94, -92, -91, -89, -87, -85, -83, -81, -79, -77, -75, -73, -70, -68, -65, -63, -60, -57, -55, -52, -49, -46, -43, -40, -37, -33, -30, -27, -24, -20, -17, -14, -10, -7, -3, 0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 33, 37, 40, 43, 46, 49, 52, 55, 57, 60, 63, 65, 68, 70, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 92, 94, 96, 97, 98, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 112, 113, 114, 115, 115, 116, 116, 117, 118, 118, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_9_nl_params, AI_ARRAY_FORMAT_S8,
    nl_9_nl_params_data, nl_9_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_9_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_9_chain,
  NULL, &conversion_10_layer, AI_STATIC, 
  .nl_params = &nl_9_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_output, &gemm_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_8_layer, 8,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_8_chain,
  NULL, &nl_9_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_3_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_7_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_7_layer, 7,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_7_chain,
  NULL, &eltwise_8_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  unpack_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 48, &unpack_3_output0, &unpack_3_output1, &unpack_3_output2, &unpack_3_output3, &unpack_3_output4, &unpack_3_output5, &unpack_3_output6, &unpack_3_output7, &unpack_3_output8, &unpack_3_output9, &unpack_3_output10, &unpack_3_output11, &unpack_3_output12, &unpack_3_output13, &unpack_3_output14, &unpack_3_output15, &unpack_3_output16, &unpack_3_output17, &unpack_3_output18, &unpack_3_output19, &unpack_3_output20, &unpack_3_output21, &unpack_3_output22, &unpack_3_output23, &unpack_3_output24, &unpack_3_output25, &unpack_3_output26, &unpack_3_output27, &unpack_3_output28, &unpack_3_output29, &unpack_3_output30, &unpack_3_output31, &unpack_3_output32, &unpack_3_output33, &unpack_3_output34, &unpack_3_output35, &unpack_3_output36, &unpack_3_output37, &unpack_3_output38, &unpack_3_output39, &unpack_3_output40, &unpack_3_output41, &unpack_3_output42, &unpack_3_output43, &unpack_3_output44, &unpack_3_output45, &unpack_3_output46, &unpack_3_output47),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  unpack_3_layer, 3,
  UNPACK_TYPE, 0x0, NULL,
  unpack, forward_unpack,
  &unpack_3_chain,
  NULL, &gemm_7_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_keras_tensor_60_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_2_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_2_chain,
  NULL, &unpack_3_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_253_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &constantofshape_251_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_253_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_253_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_253_layer, 253,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_253_chain,
  NULL, &transpose_2_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &constantofshape_5_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_6_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_6_layer, 6,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_6_chain,
  NULL, &gemm_253_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 20064, 1, 1),
    20064, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 6304, 1, 1),
    6304, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_keras_tensor_60_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &gemm_446_output),
  &gemm_6_layer, 0x99b4361d, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 20064, 1, 1),
      20064, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 6304, 1, 1),
      6304, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_keras_tensor_60_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &gemm_446_output),
  &gemm_6_layer, 0x99b4361d, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool forecast_temp_ml_model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_forecast_temp_ml_model_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_keras_tensor_60_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    serving_default_keras_tensor_60_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_6_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    gemm_6_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    gemm_6_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_6_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_253_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    gemm_253_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    gemm_253_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1504);
    gemm_253_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1504);
    transpose_2_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    transpose_2_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    unpack_3_output0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    unpack_3_output0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    unpack_3_output1_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    unpack_3_output1_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    unpack_3_output2_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    unpack_3_output2_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    unpack_3_output3_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 48);
    unpack_3_output3_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 48);
    unpack_3_output4_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    unpack_3_output4_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    unpack_3_output5_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 80);
    unpack_3_output5_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 80);
    unpack_3_output6_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 96);
    unpack_3_output6_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 96);
    unpack_3_output7_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 112);
    unpack_3_output7_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 112);
    unpack_3_output8_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    unpack_3_output8_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    unpack_3_output9_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 144);
    unpack_3_output9_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 144);
    unpack_3_output10_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 160);
    unpack_3_output10_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 160);
    unpack_3_output11_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 176);
    unpack_3_output11_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 176);
    unpack_3_output12_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    unpack_3_output12_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    unpack_3_output13_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    unpack_3_output13_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    unpack_3_output14_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    unpack_3_output14_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    unpack_3_output15_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 240);
    unpack_3_output15_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 240);
    unpack_3_output16_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 256);
    unpack_3_output16_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 256);
    unpack_3_output17_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 272);
    unpack_3_output17_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 272);
    unpack_3_output18_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    unpack_3_output18_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    unpack_3_output19_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 304);
    unpack_3_output19_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 304);
    unpack_3_output20_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 320);
    unpack_3_output20_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 320);
    unpack_3_output21_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 336);
    unpack_3_output21_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 336);
    unpack_3_output22_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 352);
    unpack_3_output22_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 352);
    unpack_3_output23_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 368);
    unpack_3_output23_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 368);
    unpack_3_output24_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 384);
    unpack_3_output24_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 384);
    unpack_3_output25_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 400);
    unpack_3_output25_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 400);
    unpack_3_output26_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 416);
    unpack_3_output26_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 416);
    unpack_3_output27_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 432);
    unpack_3_output27_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 432);
    unpack_3_output28_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 448);
    unpack_3_output28_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 448);
    unpack_3_output29_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 464);
    unpack_3_output29_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 464);
    unpack_3_output30_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 480);
    unpack_3_output30_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 480);
    unpack_3_output31_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 496);
    unpack_3_output31_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 496);
    unpack_3_output32_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 512);
    unpack_3_output32_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 512);
    unpack_3_output33_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 528);
    unpack_3_output33_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 528);
    unpack_3_output34_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 544);
    unpack_3_output34_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 544);
    unpack_3_output35_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 560);
    unpack_3_output35_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 560);
    unpack_3_output36_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 576);
    unpack_3_output36_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 576);
    unpack_3_output37_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 592);
    unpack_3_output37_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 592);
    unpack_3_output38_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 608);
    unpack_3_output38_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 608);
    unpack_3_output39_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 624);
    unpack_3_output39_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 624);
    unpack_3_output40_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 640);
    unpack_3_output40_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 640);
    unpack_3_output41_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 656);
    unpack_3_output41_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 656);
    unpack_3_output42_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    unpack_3_output42_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    unpack_3_output43_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1360);
    unpack_3_output43_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1360);
    unpack_3_output44_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1376);
    unpack_3_output44_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1376);
    unpack_3_output45_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    unpack_3_output45_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    unpack_3_output46_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1408);
    unpack_3_output46_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1408);
    unpack_3_output47_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1424);
    unpack_3_output47_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1424);
    gemm_7_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    gemm_7_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    gemm_7_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_7_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    eltwise_8_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    eltwise_8_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    nl_9_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_9_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_10_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    conversion_10_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 672);
    gemm_11_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_11_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_11_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_11_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_22_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_22_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_22_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 800);
    gemm_22_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 800);
    eltwise_23_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 864);
    eltwise_23_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 864);
    nl_24_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_24_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_25_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 800);
    conversion_25_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 800);
    gemm_26_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_26_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_26_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_26_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_37_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_37_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_37_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 864);
    gemm_37_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 864);
    eltwise_38_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 928);
    eltwise_38_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 928);
    nl_39_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_39_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_40_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 864);
    conversion_40_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 864);
    gemm_41_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_41_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_41_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_41_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_52_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_52_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_52_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 928);
    gemm_52_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 928);
    eltwise_53_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_53_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_54_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_54_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_55_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conversion_55_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_56_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_56_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_56_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_56_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_67_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_67_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_67_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 928);
    gemm_67_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 928);
    eltwise_68_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 992);
    eltwise_68_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 992);
    nl_69_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_69_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_70_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 928);
    conversion_70_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 928);
    gemm_71_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_71_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_71_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_71_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_12_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_12_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_12_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 992);
    gemm_12_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 992);
    eltwise_72_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1056);
    eltwise_72_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1056);
    nl_73_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_73_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_74_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 992);
    conversion_74_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 992);
    gemm_75_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_75_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_75_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_75_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_13_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_13_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_13_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1056);
    gemm_13_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1056);
    eltwise_76_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1120);
    eltwise_76_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1120);
    nl_77_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_77_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_78_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1056);
    conversion_78_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1056);
    gemm_79_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_79_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_79_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_79_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_14_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_14_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_14_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1120);
    gemm_14_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1120);
    eltwise_80_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_80_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_81_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_81_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_82_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    conversion_82_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_83_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_83_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_83_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_83_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_15_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_15_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_15_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1120);
    gemm_15_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1120);
    eltwise_84_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1184);
    eltwise_84_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1184);
    nl_85_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_85_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_86_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1120);
    conversion_86_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1120);
    gemm_87_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_87_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_87_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_87_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_16_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_16_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_16_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1184);
    gemm_16_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1184);
    eltwise_88_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    eltwise_88_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    nl_89_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_89_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_90_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1184);
    conversion_90_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1184);
    gemm_91_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_91_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_91_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_91_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_17_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_17_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_17_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_17_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    eltwise_92_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    eltwise_92_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    nl_93_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_93_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_94_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    conversion_94_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_95_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_95_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_95_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_95_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_18_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_18_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_18_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_18_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    eltwise_96_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_96_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_97_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_97_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_98_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    conversion_98_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_99_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_99_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_99_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_99_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_19_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_19_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_19_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_19_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    eltwise_100_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    eltwise_100_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    nl_101_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_101_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_102_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    conversion_102_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_103_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_103_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_103_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_103_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_20_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_20_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_20_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2236);
    gemm_20_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2236);
    eltwise_104_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    eltwise_104_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    nl_105_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_105_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_106_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    conversion_106_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    gemm_107_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    gemm_107_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    gemm_107_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_107_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_21_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    gemm_21_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    gemm_21_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2300);
    gemm_21_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2300);
    eltwise_108_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    eltwise_108_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    nl_109_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_109_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_110_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    conversion_110_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    gemm_111_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_111_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_111_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_111_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_27_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_27_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_27_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2364);
    gemm_27_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2364);
    eltwise_112_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    eltwise_112_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    nl_113_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_113_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_114_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    conversion_114_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    gemm_115_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_115_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_115_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_115_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_28_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_28_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_28_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2364);
    gemm_28_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2364);
    eltwise_116_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    eltwise_116_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    nl_117_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_117_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_118_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    conversion_118_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    gemm_119_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    gemm_119_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    gemm_119_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_119_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_29_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    gemm_29_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    gemm_29_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2428);
    gemm_29_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2428);
    eltwise_120_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    eltwise_120_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    nl_121_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_121_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_122_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    conversion_122_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    gemm_123_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    gemm_123_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    gemm_123_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_123_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_30_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    gemm_30_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    gemm_30_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2492);
    gemm_30_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2492);
    eltwise_124_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    eltwise_124_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    nl_125_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_125_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_126_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    conversion_126_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    gemm_127_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_127_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_127_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_127_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_31_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_31_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_31_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2556);
    gemm_31_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2556);
    eltwise_128_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 256);
    eltwise_128_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 256);
    nl_129_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_129_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_130_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 256);
    conversion_130_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 256);
    gemm_131_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_131_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_131_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_131_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_32_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_32_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_32_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2556);
    gemm_32_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2556);
    eltwise_132_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    eltwise_132_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    nl_133_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_133_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_134_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    conversion_134_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    gemm_135_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    gemm_135_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    gemm_135_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_135_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_33_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    gemm_33_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    gemm_33_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2620);
    gemm_33_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2620);
    eltwise_136_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    eltwise_136_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    nl_137_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_137_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_138_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    conversion_138_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    gemm_139_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    gemm_139_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    gemm_139_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_139_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_34_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    gemm_34_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    gemm_34_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2684);
    gemm_34_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2684);
    eltwise_140_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    eltwise_140_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    nl_141_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_141_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_142_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    conversion_142_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    gemm_143_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_143_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_143_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_143_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_35_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_35_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_35_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    gemm_35_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    eltwise_144_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 320);
    eltwise_144_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 320);
    nl_145_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_145_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_146_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 320);
    conversion_146_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 320);
    gemm_147_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_147_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_147_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_147_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_36_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_36_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_36_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    gemm_36_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    eltwise_148_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    eltwise_148_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    nl_149_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_149_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_150_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    conversion_150_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    gemm_151_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    gemm_151_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    gemm_151_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_151_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_42_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    gemm_42_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    gemm_42_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2812);
    gemm_42_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2812);
    eltwise_152_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    eltwise_152_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    nl_153_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_153_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_154_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    conversion_154_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    gemm_155_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    gemm_155_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    gemm_155_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_155_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_43_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    gemm_43_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    gemm_43_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2876);
    gemm_43_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2876);
    eltwise_156_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    eltwise_156_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    nl_157_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_157_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_158_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    conversion_158_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    gemm_159_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_159_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_159_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_159_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_44_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_44_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_44_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2940);
    gemm_44_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2940);
    eltwise_160_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 384);
    eltwise_160_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 384);
    nl_161_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_161_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_162_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 384);
    conversion_162_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 384);
    gemm_163_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_163_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_163_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_163_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_45_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_45_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_45_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2940);
    gemm_45_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2940);
    eltwise_164_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    eltwise_164_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    nl_165_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_165_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_166_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    conversion_166_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    gemm_167_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    gemm_167_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    gemm_167_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_167_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_46_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    gemm_46_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    gemm_46_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3004);
    gemm_46_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3004);
    eltwise_168_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    eltwise_168_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    nl_169_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_169_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_170_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    conversion_170_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    gemm_171_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    gemm_171_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    gemm_171_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_171_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_47_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    gemm_47_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    gemm_47_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3068);
    gemm_47_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3068);
    eltwise_172_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    eltwise_172_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    nl_173_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_173_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_174_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    conversion_174_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    gemm_175_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_175_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_175_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_175_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_48_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_48_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_48_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3132);
    gemm_48_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3132);
    eltwise_176_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 448);
    eltwise_176_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 448);
    nl_177_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_177_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_178_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 448);
    conversion_178_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 448);
    gemm_179_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_179_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_179_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_179_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_49_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_49_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_49_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3132);
    gemm_49_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3132);
    eltwise_180_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    eltwise_180_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    nl_181_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_181_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_182_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    conversion_182_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    gemm_183_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    gemm_183_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    gemm_183_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_183_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_50_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    gemm_50_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    gemm_50_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3196);
    gemm_50_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3196);
    eltwise_184_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    eltwise_184_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    nl_185_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_185_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_186_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    conversion_186_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    gemm_187_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    gemm_187_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    gemm_187_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_187_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_51_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    gemm_51_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    gemm_51_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3260);
    gemm_51_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3260);
    eltwise_188_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    eltwise_188_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    nl_189_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_189_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_190_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    conversion_190_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    gemm_191_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_191_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_191_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_191_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_57_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_57_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_57_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3324);
    gemm_57_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3324);
    eltwise_192_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 512);
    eltwise_192_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 512);
    nl_193_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_193_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_194_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 512);
    conversion_194_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 512);
    gemm_195_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_195_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_195_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_195_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_58_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_58_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_58_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3324);
    gemm_58_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3324);
    eltwise_196_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    eltwise_196_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    nl_197_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_197_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_198_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    conversion_198_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    gemm_199_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    gemm_199_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    gemm_199_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_199_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_59_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    gemm_59_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    gemm_59_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3388);
    gemm_59_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3388);
    eltwise_200_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    eltwise_200_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    nl_201_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_201_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_202_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    conversion_202_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    gemm_203_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    gemm_203_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    gemm_203_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_203_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_60_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    gemm_60_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    gemm_60_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3452);
    gemm_60_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3452);
    eltwise_204_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    eltwise_204_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    nl_205_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_205_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_206_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    conversion_206_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    gemm_207_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_207_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_207_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_207_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_61_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_61_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_61_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3516);
    gemm_61_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3516);
    eltwise_208_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 576);
    eltwise_208_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 576);
    nl_209_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_209_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_210_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 576);
    conversion_210_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 576);
    gemm_211_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_211_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_211_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_211_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_62_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_62_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_62_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3516);
    gemm_62_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3516);
    eltwise_212_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    eltwise_212_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    nl_213_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_213_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_214_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    conversion_214_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_215_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    gemm_215_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    gemm_215_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_215_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_63_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    gemm_63_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    gemm_63_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3580);
    gemm_63_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3580);
    eltwise_216_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    eltwise_216_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    nl_217_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_217_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_218_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    conversion_218_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    gemm_219_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    gemm_219_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    gemm_219_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_219_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_64_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    gemm_64_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    gemm_64_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3644);
    gemm_64_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3644);
    eltwise_220_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    eltwise_220_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    nl_221_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_221_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_222_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    conversion_222_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    gemm_223_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_223_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_223_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_223_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_65_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_65_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_65_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3708);
    gemm_65_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3708);
    eltwise_224_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1312);
    eltwise_224_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1312);
    nl_225_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_225_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_226_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1312);
    conversion_226_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1312);
    gemm_227_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_227_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_227_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_227_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_66_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_66_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_66_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3708);
    gemm_66_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3708);
    eltwise_228_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    eltwise_228_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    nl_229_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_229_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_230_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    conversion_230_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    gemm_231_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    gemm_231_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    gemm_231_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_231_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_232_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    gemm_232_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    gemm_232_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3772);
    gemm_232_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3772);
    eltwise_233_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    eltwise_233_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    nl_234_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_234_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_235_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    conversion_235_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    gemm_236_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_236_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_236_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_236_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_237_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_237_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_237_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3836);
    gemm_237_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3836);
    eltwise_238_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    eltwise_238_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    nl_239_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_239_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_240_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    conversion_240_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_241_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3232);
    gemm_241_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3232);
    gemm_241_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_241_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    gemm_242_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3232);
    gemm_242_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3232);
    gemm_242_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3900);
    gemm_242_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3900);
    eltwise_243_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1376);
    eltwise_243_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1376);
    nl_244_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    nl_244_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 736);
    conversion_245_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1376);
    conversion_245_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1376);
    pack_246_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3232);
    pack_246_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3232);
    unpack_252_output0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    unpack_252_output0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    unpack_252_output1_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    unpack_252_output1_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    unpack_252_output2_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    unpack_252_output2_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    unpack_252_output3_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    unpack_252_output3_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    unpack_252_output4_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 256);
    unpack_252_output4_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 256);
    unpack_252_output5_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 320);
    unpack_252_output5_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 320);
    unpack_252_output6_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 384);
    unpack_252_output6_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 384);
    unpack_252_output7_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 448);
    unpack_252_output7_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 448);
    unpack_252_output8_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 512);
    unpack_252_output8_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 512);
    unpack_252_output9_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 576);
    unpack_252_output9_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 576);
    unpack_252_output10_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 640);
    unpack_252_output10_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 640);
    unpack_252_output11_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 704);
    unpack_252_output11_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 704);
    unpack_252_output12_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    unpack_252_output12_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    unpack_252_output13_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    unpack_252_output13_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    unpack_252_output14_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 896);
    unpack_252_output14_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 896);
    unpack_252_output15_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 960);
    unpack_252_output15_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 960);
    unpack_252_output16_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1024);
    unpack_252_output16_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1024);
    unpack_252_output17_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1088);
    unpack_252_output17_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1088);
    unpack_252_output18_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    unpack_252_output18_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    unpack_252_output19_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1216);
    unpack_252_output19_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1216);
    unpack_252_output20_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1280);
    unpack_252_output20_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1280);
    unpack_252_output21_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    unpack_252_output21_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    unpack_252_output22_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1408);
    unpack_252_output22_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1408);
    unpack_252_output23_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    unpack_252_output23_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1568);
    unpack_252_output24_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    unpack_252_output24_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    unpack_252_output25_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    unpack_252_output25_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1696);
    unpack_252_output26_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    unpack_252_output26_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1760);
    unpack_252_output27_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    unpack_252_output27_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    unpack_252_output28_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    unpack_252_output28_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1888);
    unpack_252_output29_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    unpack_252_output29_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1952);
    unpack_252_output30_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    unpack_252_output30_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    unpack_252_output31_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    unpack_252_output31_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2080);
    unpack_252_output32_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    unpack_252_output32_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    unpack_252_output33_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    unpack_252_output33_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    unpack_252_output34_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    unpack_252_output34_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2272);
    unpack_252_output35_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    unpack_252_output35_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2336);
    unpack_252_output36_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    unpack_252_output36_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2400);
    unpack_252_output37_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    unpack_252_output37_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2464);
    unpack_252_output38_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    unpack_252_output38_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2528);
    unpack_252_output39_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    unpack_252_output39_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2592);
    unpack_252_output40_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    unpack_252_output40_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2656);
    unpack_252_output41_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    unpack_252_output41_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2720);
    unpack_252_output42_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    unpack_252_output42_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2784);
    unpack_252_output43_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    unpack_252_output43_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    unpack_252_output44_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    unpack_252_output44_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2912);
    unpack_252_output45_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    unpack_252_output45_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2976);
    unpack_252_output46_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    unpack_252_output46_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    unpack_252_output47_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    unpack_252_output47_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3104);
    gemm_254_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_254_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_254_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3936);
    gemm_254_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3936);
    eltwise_255_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_255_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_256_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1472);
    nl_256_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1472);
    gemm_257_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_257_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_257_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_257_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_268_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_268_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_268_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1472);
    gemm_268_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1472);
    eltwise_269_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_269_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_270_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_270_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_271_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_271_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_271_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_271_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_282_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_282_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_282_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_282_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_283_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_283_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_284_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_284_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_285_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_285_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_285_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_285_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_296_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_296_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_296_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_296_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_297_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_297_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_298_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_298_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_299_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_299_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_299_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_299_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_310_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_310_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_310_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_310_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_311_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_311_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_312_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_312_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_313_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_313_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_313_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_313_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_258_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_258_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_258_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_258_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_314_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_314_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_315_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_315_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_316_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_316_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_316_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_316_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_259_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_259_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_259_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_259_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_317_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_317_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_318_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_318_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_319_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_319_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_319_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_319_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_260_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_260_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_260_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_260_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_320_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_320_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_321_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_321_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_322_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_322_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_322_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_322_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_261_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_261_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_261_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_261_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_323_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_323_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_324_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_324_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_325_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_325_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_325_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_325_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_262_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_262_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_262_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_262_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_326_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_326_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_327_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_327_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_328_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_328_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_328_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_328_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_263_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_263_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_263_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_263_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_329_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_329_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_330_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_330_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_331_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_331_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_331_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_331_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_264_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_264_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_264_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_264_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_332_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_332_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_333_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_333_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_334_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_334_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_334_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_334_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_265_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_265_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3168);
    gemm_265_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_265_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_335_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    eltwise_335_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    nl_336_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_336_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_337_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_337_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_337_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1472);
    gemm_337_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1472);
    gemm_266_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_266_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_266_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_266_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_338_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_338_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_339_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_339_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_340_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_340_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_340_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_340_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_267_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_267_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_267_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_267_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_341_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_341_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_342_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_342_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_343_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_343_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_343_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_343_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_272_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_272_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_272_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_272_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_344_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_344_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_345_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_345_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_346_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_346_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_346_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_346_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_273_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_273_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_273_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_273_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_347_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_347_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_348_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_348_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_349_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_349_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_349_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_349_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_274_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_274_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_274_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_274_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_350_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_350_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_351_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_351_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_352_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_352_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_352_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_352_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_275_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_275_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_275_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_275_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_353_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_353_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_354_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_354_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_355_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_355_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_355_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_355_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_276_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_276_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_276_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_276_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_356_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_356_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_357_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_357_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_358_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_358_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_358_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_358_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_277_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_277_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_277_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_277_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_359_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_359_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_360_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_360_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_361_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_361_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_361_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_361_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_278_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_278_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_278_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_278_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_362_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_362_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_363_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_363_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_364_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_364_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_364_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_364_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_279_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_279_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_279_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_279_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_365_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_365_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_366_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_366_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_367_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_367_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_367_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_367_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_280_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_280_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_280_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_280_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_368_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_368_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_369_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_369_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_370_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_370_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_370_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_370_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_281_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_281_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_281_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_281_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_371_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_371_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_372_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_372_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_373_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_373_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_373_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_373_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_286_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_286_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_286_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_286_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_374_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_374_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_375_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_375_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_376_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_376_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_376_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_376_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_287_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_287_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_287_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_287_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_377_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_377_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_378_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_378_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_379_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_379_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_379_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_379_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_288_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_288_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_288_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_288_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_380_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_380_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_381_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_381_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_382_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_382_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_382_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_382_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_289_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_289_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_289_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_289_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_383_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_383_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_384_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_384_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_385_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_385_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_385_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_385_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_290_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_290_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_290_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_290_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_386_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_386_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_387_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_387_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_388_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_388_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_388_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_388_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_291_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_291_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_291_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_291_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_389_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_389_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_390_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_390_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_391_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_391_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_391_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_391_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_292_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_292_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_292_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_292_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_392_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_392_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_393_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_393_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_394_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_394_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_394_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_394_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_293_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_293_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_293_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_293_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_395_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_395_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_396_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_396_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_397_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_397_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_397_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_397_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_294_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_294_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_294_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_294_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_398_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_398_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_399_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_399_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_400_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_400_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_400_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_400_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_295_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_295_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_295_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_295_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_401_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_401_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_402_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_402_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_403_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_403_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_403_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_403_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_300_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_300_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_300_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_300_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_404_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_404_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_405_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_405_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_406_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_406_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_406_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_406_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_301_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_301_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_301_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_301_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_407_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_407_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_408_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_408_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_409_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_409_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_409_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_409_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_302_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_302_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_302_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_302_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_410_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_410_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_411_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_411_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_412_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_412_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_412_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_412_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_303_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_303_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_303_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_303_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_413_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_413_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_414_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_414_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_415_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_415_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_415_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_415_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_304_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_304_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_304_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_304_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_416_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_416_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_417_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_417_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_418_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_418_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_418_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_418_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_305_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_305_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_305_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_305_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_419_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_419_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_420_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_420_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_421_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_421_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_421_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_421_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_306_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_306_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_306_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_306_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_422_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_422_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_423_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_423_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_424_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_424_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_424_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_424_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_307_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_307_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_307_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_307_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_425_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_425_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_426_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_426_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_427_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_427_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_427_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_427_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_308_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_308_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_308_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_308_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_428_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_428_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_429_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_429_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_430_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_430_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_430_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_430_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_309_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_309_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_309_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_309_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_431_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_431_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_432_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_432_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_433_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_433_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_433_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_433_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_434_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_434_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_434_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_434_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_435_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_435_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_436_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_436_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_437_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_437_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_437_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_437_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_438_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_438_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_438_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_438_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    eltwise_439_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    eltwise_439_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_440_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_440_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_441_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_441_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_441_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_441_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_442_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_442_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_442_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_442_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    eltwise_443_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_443_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_444_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    nl_444_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_445_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_445_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 128);
    gemm_445_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_445_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_446_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_446_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_446_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 432);
    gemm_446_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 432);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool forecast_temp_ml_model_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_forecast_temp_ml_model_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    constantofshape_5_const_array.format |= AI_FMT_FLAG_CONST;
    constantofshape_5_const_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 0);
    constantofshape_5_const_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 0);
    gemm_6_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_6_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 64);
    gemm_6_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 64);
    gemm_6_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_6_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 4160);
    gemm_6_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 4160);
    constantofshape_251_const_array.format |= AI_FMT_FLAG_CONST;
    constantofshape_251_const_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 4416);
    constantofshape_251_const_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 4416);
    gemm_253_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_253_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 4480);
    gemm_253_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 4480);
    gemm_7_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_7_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 8576);
    gemm_7_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 8576);
    gemm_7_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_7_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 9472);
    gemm_7_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 9472);
    gemm_254_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_254_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 9728);
    gemm_254_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 9728);
    gemm_254_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_254_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 13824);
    gemm_254_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 13824);
    gemm_445_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_445_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 14080);
    gemm_445_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 14080);
    gemm_445_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_445_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 18176);
    gemm_445_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 18176);
    gemm_446_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_446_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 18432);
    gemm_446_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 18432);
    gemm_446_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_446_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 19968);
    gemm_446_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 19968);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_forecast_temp_ml_model_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_FORECAST_TEMP_ML_MODEL_MODEL_NAME,
      .model_signature   = AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 669608,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x99b4361d,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_forecast_temp_ml_model_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_FORECAST_TEMP_ML_MODEL_MODEL_NAME,
      .model_signature   = AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 669608,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x99b4361d,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_forecast_temp_ml_model_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_forecast_temp_ml_model_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_forecast_temp_ml_model_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_forecast_temp_ml_model_create(network, AI_FORECAST_TEMP_ML_MODEL_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_forecast_temp_ml_model_data_params_get(&params) != true) {
    err = ai_forecast_temp_ml_model_get_error(*network);
    return err;
  }
#if defined(AI_FORECAST_TEMP_ML_MODEL_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_FORECAST_TEMP_ML_MODEL_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_forecast_temp_ml_model_init(*network, &params) != true) {
    err = ai_forecast_temp_ml_model_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_forecast_temp_ml_model_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_forecast_temp_ml_model_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_forecast_temp_ml_model_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_forecast_temp_ml_model_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= forecast_temp_ml_model_configure_weights(net_ctx, params);
  ok &= forecast_temp_ml_model_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_forecast_temp_ml_model_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_forecast_temp_ml_model_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

