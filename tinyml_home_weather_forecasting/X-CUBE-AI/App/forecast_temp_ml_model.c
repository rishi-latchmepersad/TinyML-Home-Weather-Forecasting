/**
  ******************************************************************************
  * @file    forecast_temp_ml_model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-28T16:06:24-0400
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
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
#define AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE     "0x57394cb92bd8ef801469af0172a4bbeb"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-03-28T16:06:24-0400"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_FORECAST_TEMP_ML_MODEL_N_BATCHES
#define AI_FORECAST_TEMP_ML_MODEL_N_BATCHES         (1)

static ai_ptr g_forecast_temp_ml_model_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_forecast_temp_ml_model_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  gemm_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gemm_253_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_sensor_history0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 432, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  transpose_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 432, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output2_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output3_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output4_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output5_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output6_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output7_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output8_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output9_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output10_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output11_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output12_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output13_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output14_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output15_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output16_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output17_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output18_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output19_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output20_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output21_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output22_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output23_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output24_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output25_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output26_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output27_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output28_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output29_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output30_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output31_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output32_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output33_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output34_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output35_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output36_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output37_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output38_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output39_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output40_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output41_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output42_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output43_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output44_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output45_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output46_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  unpack_7_output47_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  gemm_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  nl_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conversion_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  gemm_23_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  nl_25_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  conversion_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  gemm_27_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  gemm_38_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_39_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  nl_40_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  conversion_41_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  gemm_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  nl_55_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  conversion_56_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  gemm_57_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  gemm_68_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_69_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  nl_70_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  conversion_71_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  gemm_72_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_73_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  nl_74_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  conversion_75_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  gemm_76_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_77_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  nl_78_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  conversion_79_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  gemm_80_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_81_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  nl_82_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  conversion_83_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  gemm_84_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_85_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  nl_86_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  conversion_87_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  gemm_88_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_89_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  nl_90_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  conversion_91_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  gemm_92_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  gemm_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_93_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  nl_94_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  conversion_95_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  gemm_96_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_97_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  nl_98_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  conversion_99_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  gemm_100_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  gemm_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_101_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  nl_102_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  conversion_103_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  gemm_104_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  gemm_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_105_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  nl_106_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  conversion_107_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  gemm_108_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  gemm_22_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_109_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  nl_110_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  conversion_111_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  gemm_112_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  gemm_28_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_113_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  nl_114_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  conversion_115_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  gemm_116_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  gemm_29_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_117_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  nl_118_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  conversion_119_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  gemm_120_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  gemm_30_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_121_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  nl_122_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  conversion_123_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  gemm_124_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  gemm_31_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_125_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  nl_126_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  conversion_127_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  gemm_128_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  gemm_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_129_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  nl_130_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  conversion_131_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  gemm_132_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  gemm_33_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_133_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  nl_134_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  conversion_135_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  gemm_136_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  gemm_34_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_137_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  nl_138_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  conversion_139_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  gemm_140_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  gemm_35_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_141_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  nl_142_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  conversion_143_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  gemm_144_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_145_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  nl_146_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  conversion_147_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  gemm_148_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  gemm_37_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_149_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  nl_150_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  conversion_151_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  gemm_152_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  gemm_43_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_153_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  nl_154_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  conversion_155_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  gemm_156_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  gemm_44_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_157_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  nl_158_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  conversion_159_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  gemm_160_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  gemm_45_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_161_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  nl_162_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  conversion_163_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  gemm_164_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_165_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  nl_166_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  conversion_167_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  gemm_168_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  gemm_47_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_169_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  nl_170_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  conversion_171_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  gemm_172_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_173_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  nl_174_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  conversion_175_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  gemm_176_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_177_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  nl_178_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  conversion_179_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  gemm_180_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#212 */
AI_ARRAY_OBJ_DECLARE(
  gemm_50_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#213 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_181_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#214 */
AI_ARRAY_OBJ_DECLARE(
  nl_182_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#215 */
AI_ARRAY_OBJ_DECLARE(
  conversion_183_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#216 */
AI_ARRAY_OBJ_DECLARE(
  gemm_184_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#217 */
AI_ARRAY_OBJ_DECLARE(
  gemm_51_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#218 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_185_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#219 */
AI_ARRAY_OBJ_DECLARE(
  nl_186_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#220 */
AI_ARRAY_OBJ_DECLARE(
  conversion_187_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#221 */
AI_ARRAY_OBJ_DECLARE(
  gemm_188_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#222 */
AI_ARRAY_OBJ_DECLARE(
  gemm_52_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#223 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_189_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#224 */
AI_ARRAY_OBJ_DECLARE(
  nl_190_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#225 */
AI_ARRAY_OBJ_DECLARE(
  conversion_191_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#226 */
AI_ARRAY_OBJ_DECLARE(
  gemm_192_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#227 */
AI_ARRAY_OBJ_DECLARE(
  gemm_58_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#228 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_193_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#229 */
AI_ARRAY_OBJ_DECLARE(
  nl_194_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#230 */
AI_ARRAY_OBJ_DECLARE(
  conversion_195_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#231 */
AI_ARRAY_OBJ_DECLARE(
  gemm_196_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#232 */
AI_ARRAY_OBJ_DECLARE(
  gemm_59_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#233 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_197_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#234 */
AI_ARRAY_OBJ_DECLARE(
  nl_198_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#235 */
AI_ARRAY_OBJ_DECLARE(
  conversion_199_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#236 */
AI_ARRAY_OBJ_DECLARE(
  gemm_200_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#237 */
AI_ARRAY_OBJ_DECLARE(
  gemm_60_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#238 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_201_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#239 */
AI_ARRAY_OBJ_DECLARE(
  nl_202_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#240 */
AI_ARRAY_OBJ_DECLARE(
  conversion_203_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#241 */
AI_ARRAY_OBJ_DECLARE(
  gemm_204_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#242 */
AI_ARRAY_OBJ_DECLARE(
  gemm_61_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#243 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_205_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#244 */
AI_ARRAY_OBJ_DECLARE(
  nl_206_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#245 */
AI_ARRAY_OBJ_DECLARE(
  conversion_207_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#246 */
AI_ARRAY_OBJ_DECLARE(
  gemm_208_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#247 */
AI_ARRAY_OBJ_DECLARE(
  gemm_62_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#248 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_209_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#249 */
AI_ARRAY_OBJ_DECLARE(
  nl_210_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#250 */
AI_ARRAY_OBJ_DECLARE(
  conversion_211_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#251 */
AI_ARRAY_OBJ_DECLARE(
  gemm_212_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#252 */
AI_ARRAY_OBJ_DECLARE(
  gemm_63_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#253 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_213_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#254 */
AI_ARRAY_OBJ_DECLARE(
  nl_214_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#255 */
AI_ARRAY_OBJ_DECLARE(
  conversion_215_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#256 */
AI_ARRAY_OBJ_DECLARE(
  gemm_216_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#257 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#258 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_217_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#259 */
AI_ARRAY_OBJ_DECLARE(
  nl_218_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#260 */
AI_ARRAY_OBJ_DECLARE(
  conversion_219_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#261 */
AI_ARRAY_OBJ_DECLARE(
  gemm_220_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#262 */
AI_ARRAY_OBJ_DECLARE(
  gemm_65_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#263 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_221_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#264 */
AI_ARRAY_OBJ_DECLARE(
  nl_222_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#265 */
AI_ARRAY_OBJ_DECLARE(
  conversion_223_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#266 */
AI_ARRAY_OBJ_DECLARE(
  gemm_224_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#267 */
AI_ARRAY_OBJ_DECLARE(
  gemm_66_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#268 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_225_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#269 */
AI_ARRAY_OBJ_DECLARE(
  nl_226_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#270 */
AI_ARRAY_OBJ_DECLARE(
  conversion_227_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#271 */
AI_ARRAY_OBJ_DECLARE(
  gemm_228_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#272 */
AI_ARRAY_OBJ_DECLARE(
  gemm_67_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#273 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_229_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#274 */
AI_ARRAY_OBJ_DECLARE(
  nl_230_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#275 */
AI_ARRAY_OBJ_DECLARE(
  conversion_231_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#276 */
AI_ARRAY_OBJ_DECLARE(
  gemm_232_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#277 */
AI_ARRAY_OBJ_DECLARE(
  gemm_233_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#278 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_234_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#279 */
AI_ARRAY_OBJ_DECLARE(
  nl_235_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#280 */
AI_ARRAY_OBJ_DECLARE(
  conversion_236_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#281 */
AI_ARRAY_OBJ_DECLARE(
  gemm_237_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#282 */
AI_ARRAY_OBJ_DECLARE(
  gemm_238_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#283 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_239_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#284 */
AI_ARRAY_OBJ_DECLARE(
  nl_240_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#285 */
AI_ARRAY_OBJ_DECLARE(
  conversion_241_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#286 */
AI_ARRAY_OBJ_DECLARE(
  gemm_242_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#287 */
AI_ARRAY_OBJ_DECLARE(
  gemm_243_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#288 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_244_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#289 */
AI_ARRAY_OBJ_DECLARE(
  nl_245_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#290 */
AI_ARRAY_OBJ_DECLARE(
  conversion_246_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#291 */
AI_ARRAY_OBJ_DECLARE(
  pack_247_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#292 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#293 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#294 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output2_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#295 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output3_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#296 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output4_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#297 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output5_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#298 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output6_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#299 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output7_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#300 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output8_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#301 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output9_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#302 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output10_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#303 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output11_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#304 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output12_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#305 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output13_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#306 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output14_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#307 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output15_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#308 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output16_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#309 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output17_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#310 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output18_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#311 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output19_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#312 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output20_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#313 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output21_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#314 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output22_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#315 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output23_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#316 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output24_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#317 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output25_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#318 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output26_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#319 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output27_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#320 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output28_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#321 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output29_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#322 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output30_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#323 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output31_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#324 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output32_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#325 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output33_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#326 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output34_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#327 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output35_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#328 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output36_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#329 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output37_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#330 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output38_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#331 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output39_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#332 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output40_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#333 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output41_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#334 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output42_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#335 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output43_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#336 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output44_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#337 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output45_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#338 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output46_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#339 */
AI_ARRAY_OBJ_DECLARE(
  unpack_254_output47_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#340 */
AI_ARRAY_OBJ_DECLARE(
  gemm_255_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#341 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_256_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#342 */
AI_ARRAY_OBJ_DECLARE(
  nl_257_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#343 */
AI_ARRAY_OBJ_DECLARE(
  gemm_258_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#344 */
AI_ARRAY_OBJ_DECLARE(
  gemm_269_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#345 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_270_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#346 */
AI_ARRAY_OBJ_DECLARE(
  nl_271_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#347 */
AI_ARRAY_OBJ_DECLARE(
  gemm_272_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#348 */
AI_ARRAY_OBJ_DECLARE(
  gemm_283_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#349 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_284_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#350 */
AI_ARRAY_OBJ_DECLARE(
  nl_285_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#351 */
AI_ARRAY_OBJ_DECLARE(
  gemm_286_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#352 */
AI_ARRAY_OBJ_DECLARE(
  gemm_297_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#353 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_298_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#354 */
AI_ARRAY_OBJ_DECLARE(
  nl_299_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#355 */
AI_ARRAY_OBJ_DECLARE(
  gemm_300_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#356 */
AI_ARRAY_OBJ_DECLARE(
  gemm_311_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#357 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_312_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#358 */
AI_ARRAY_OBJ_DECLARE(
  nl_313_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#359 */
AI_ARRAY_OBJ_DECLARE(
  gemm_314_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#360 */
AI_ARRAY_OBJ_DECLARE(
  gemm_259_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#361 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_315_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#362 */
AI_ARRAY_OBJ_DECLARE(
  nl_316_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#363 */
AI_ARRAY_OBJ_DECLARE(
  gemm_317_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#364 */
AI_ARRAY_OBJ_DECLARE(
  gemm_260_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#365 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_318_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#366 */
AI_ARRAY_OBJ_DECLARE(
  nl_319_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#367 */
AI_ARRAY_OBJ_DECLARE(
  gemm_320_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#368 */
AI_ARRAY_OBJ_DECLARE(
  gemm_261_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#369 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_321_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#370 */
AI_ARRAY_OBJ_DECLARE(
  nl_322_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#371 */
AI_ARRAY_OBJ_DECLARE(
  gemm_323_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#372 */
AI_ARRAY_OBJ_DECLARE(
  gemm_262_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#373 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_324_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#374 */
AI_ARRAY_OBJ_DECLARE(
  nl_325_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#375 */
AI_ARRAY_OBJ_DECLARE(
  gemm_326_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#376 */
AI_ARRAY_OBJ_DECLARE(
  gemm_263_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#377 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_327_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#378 */
AI_ARRAY_OBJ_DECLARE(
  nl_328_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#379 */
AI_ARRAY_OBJ_DECLARE(
  gemm_329_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#380 */
AI_ARRAY_OBJ_DECLARE(
  gemm_264_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#381 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_330_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#382 */
AI_ARRAY_OBJ_DECLARE(
  nl_331_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#383 */
AI_ARRAY_OBJ_DECLARE(
  gemm_332_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#384 */
AI_ARRAY_OBJ_DECLARE(
  gemm_265_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#385 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_333_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#386 */
AI_ARRAY_OBJ_DECLARE(
  nl_334_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#387 */
AI_ARRAY_OBJ_DECLARE(
  gemm_335_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#388 */
AI_ARRAY_OBJ_DECLARE(
  gemm_266_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#389 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_336_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#390 */
AI_ARRAY_OBJ_DECLARE(
  nl_337_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#391 */
AI_ARRAY_OBJ_DECLARE(
  gemm_338_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#392 */
AI_ARRAY_OBJ_DECLARE(
  gemm_267_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#393 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_339_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#394 */
AI_ARRAY_OBJ_DECLARE(
  nl_340_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#395 */
AI_ARRAY_OBJ_DECLARE(
  gemm_341_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#396 */
AI_ARRAY_OBJ_DECLARE(
  gemm_268_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#397 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_342_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#398 */
AI_ARRAY_OBJ_DECLARE(
  nl_343_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#399 */
AI_ARRAY_OBJ_DECLARE(
  gemm_344_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#400 */
AI_ARRAY_OBJ_DECLARE(
  gemm_273_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#401 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_345_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#402 */
AI_ARRAY_OBJ_DECLARE(
  nl_346_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#403 */
AI_ARRAY_OBJ_DECLARE(
  gemm_347_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#404 */
AI_ARRAY_OBJ_DECLARE(
  gemm_274_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#405 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_348_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#406 */
AI_ARRAY_OBJ_DECLARE(
  nl_349_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#407 */
AI_ARRAY_OBJ_DECLARE(
  gemm_350_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#408 */
AI_ARRAY_OBJ_DECLARE(
  gemm_275_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#409 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_351_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#410 */
AI_ARRAY_OBJ_DECLARE(
  nl_352_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#411 */
AI_ARRAY_OBJ_DECLARE(
  gemm_353_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#412 */
AI_ARRAY_OBJ_DECLARE(
  gemm_276_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#413 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_354_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#414 */
AI_ARRAY_OBJ_DECLARE(
  nl_355_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#415 */
AI_ARRAY_OBJ_DECLARE(
  gemm_356_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#416 */
AI_ARRAY_OBJ_DECLARE(
  gemm_277_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#417 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_357_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#418 */
AI_ARRAY_OBJ_DECLARE(
  nl_358_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#419 */
AI_ARRAY_OBJ_DECLARE(
  gemm_359_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#420 */
AI_ARRAY_OBJ_DECLARE(
  gemm_278_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#421 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_360_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#422 */
AI_ARRAY_OBJ_DECLARE(
  nl_361_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#423 */
AI_ARRAY_OBJ_DECLARE(
  gemm_362_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#424 */
AI_ARRAY_OBJ_DECLARE(
  gemm_279_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#425 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_363_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#426 */
AI_ARRAY_OBJ_DECLARE(
  nl_364_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#427 */
AI_ARRAY_OBJ_DECLARE(
  gemm_365_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#428 */
AI_ARRAY_OBJ_DECLARE(
  gemm_280_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#429 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_366_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#430 */
AI_ARRAY_OBJ_DECLARE(
  nl_367_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#431 */
AI_ARRAY_OBJ_DECLARE(
  gemm_368_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#432 */
AI_ARRAY_OBJ_DECLARE(
  gemm_281_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#433 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_369_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#434 */
AI_ARRAY_OBJ_DECLARE(
  nl_370_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#435 */
AI_ARRAY_OBJ_DECLARE(
  gemm_371_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#436 */
AI_ARRAY_OBJ_DECLARE(
  gemm_282_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#437 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_372_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#438 */
AI_ARRAY_OBJ_DECLARE(
  nl_373_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#439 */
AI_ARRAY_OBJ_DECLARE(
  gemm_374_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#440 */
AI_ARRAY_OBJ_DECLARE(
  gemm_287_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#441 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_375_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#442 */
AI_ARRAY_OBJ_DECLARE(
  nl_376_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#443 */
AI_ARRAY_OBJ_DECLARE(
  gemm_377_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#444 */
AI_ARRAY_OBJ_DECLARE(
  gemm_288_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#445 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_378_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#446 */
AI_ARRAY_OBJ_DECLARE(
  nl_379_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#447 */
AI_ARRAY_OBJ_DECLARE(
  gemm_380_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#448 */
AI_ARRAY_OBJ_DECLARE(
  gemm_289_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#449 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_381_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#450 */
AI_ARRAY_OBJ_DECLARE(
  nl_382_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#451 */
AI_ARRAY_OBJ_DECLARE(
  gemm_383_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#452 */
AI_ARRAY_OBJ_DECLARE(
  gemm_290_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#453 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_384_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#454 */
AI_ARRAY_OBJ_DECLARE(
  nl_385_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#455 */
AI_ARRAY_OBJ_DECLARE(
  gemm_386_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#456 */
AI_ARRAY_OBJ_DECLARE(
  gemm_291_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#457 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_387_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#458 */
AI_ARRAY_OBJ_DECLARE(
  nl_388_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#459 */
AI_ARRAY_OBJ_DECLARE(
  gemm_389_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#460 */
AI_ARRAY_OBJ_DECLARE(
  gemm_292_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#461 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_390_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#462 */
AI_ARRAY_OBJ_DECLARE(
  nl_391_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#463 */
AI_ARRAY_OBJ_DECLARE(
  gemm_392_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#464 */
AI_ARRAY_OBJ_DECLARE(
  gemm_293_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#465 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_393_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#466 */
AI_ARRAY_OBJ_DECLARE(
  nl_394_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#467 */
AI_ARRAY_OBJ_DECLARE(
  gemm_395_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#468 */
AI_ARRAY_OBJ_DECLARE(
  gemm_294_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#469 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_396_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#470 */
AI_ARRAY_OBJ_DECLARE(
  nl_397_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#471 */
AI_ARRAY_OBJ_DECLARE(
  gemm_398_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#472 */
AI_ARRAY_OBJ_DECLARE(
  gemm_295_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#473 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_399_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#474 */
AI_ARRAY_OBJ_DECLARE(
  nl_400_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#475 */
AI_ARRAY_OBJ_DECLARE(
  gemm_401_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#476 */
AI_ARRAY_OBJ_DECLARE(
  gemm_296_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#477 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_402_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#478 */
AI_ARRAY_OBJ_DECLARE(
  nl_403_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#479 */
AI_ARRAY_OBJ_DECLARE(
  gemm_404_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#480 */
AI_ARRAY_OBJ_DECLARE(
  gemm_301_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#481 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_405_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#482 */
AI_ARRAY_OBJ_DECLARE(
  nl_406_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#483 */
AI_ARRAY_OBJ_DECLARE(
  gemm_407_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#484 */
AI_ARRAY_OBJ_DECLARE(
  gemm_302_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#485 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_408_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#486 */
AI_ARRAY_OBJ_DECLARE(
  nl_409_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#487 */
AI_ARRAY_OBJ_DECLARE(
  gemm_410_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#488 */
AI_ARRAY_OBJ_DECLARE(
  gemm_303_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#489 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_411_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#490 */
AI_ARRAY_OBJ_DECLARE(
  nl_412_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#491 */
AI_ARRAY_OBJ_DECLARE(
  gemm_413_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#492 */
AI_ARRAY_OBJ_DECLARE(
  gemm_304_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#493 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_414_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#494 */
AI_ARRAY_OBJ_DECLARE(
  nl_415_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#495 */
AI_ARRAY_OBJ_DECLARE(
  gemm_416_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#496 */
AI_ARRAY_OBJ_DECLARE(
  gemm_305_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#497 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_417_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#498 */
AI_ARRAY_OBJ_DECLARE(
  nl_418_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#499 */
AI_ARRAY_OBJ_DECLARE(
  gemm_419_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#500 */
AI_ARRAY_OBJ_DECLARE(
  gemm_306_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#501 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_420_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#502 */
AI_ARRAY_OBJ_DECLARE(
  nl_421_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#503 */
AI_ARRAY_OBJ_DECLARE(
  gemm_422_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#504 */
AI_ARRAY_OBJ_DECLARE(
  gemm_307_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#505 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_423_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#506 */
AI_ARRAY_OBJ_DECLARE(
  nl_424_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#507 */
AI_ARRAY_OBJ_DECLARE(
  gemm_425_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#508 */
AI_ARRAY_OBJ_DECLARE(
  gemm_308_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#509 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_426_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#510 */
AI_ARRAY_OBJ_DECLARE(
  nl_427_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#511 */
AI_ARRAY_OBJ_DECLARE(
  gemm_428_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#512 */
AI_ARRAY_OBJ_DECLARE(
  gemm_309_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#513 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_429_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#514 */
AI_ARRAY_OBJ_DECLARE(
  nl_430_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#515 */
AI_ARRAY_OBJ_DECLARE(
  gemm_431_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#516 */
AI_ARRAY_OBJ_DECLARE(
  gemm_310_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#517 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_432_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#518 */
AI_ARRAY_OBJ_DECLARE(
  nl_433_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#519 */
AI_ARRAY_OBJ_DECLARE(
  gemm_434_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#520 */
AI_ARRAY_OBJ_DECLARE(
  gemm_435_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#521 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_436_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#522 */
AI_ARRAY_OBJ_DECLARE(
  nl_437_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#523 */
AI_ARRAY_OBJ_DECLARE(
  gemm_438_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#524 */
AI_ARRAY_OBJ_DECLARE(
  gemm_439_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#525 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_440_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#526 */
AI_ARRAY_OBJ_DECLARE(
  nl_441_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#527 */
AI_ARRAY_OBJ_DECLARE(
  gemm_442_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#528 */
AI_ARRAY_OBJ_DECLARE(
  gemm_443_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#529 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_444_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#530 */
AI_ARRAY_OBJ_DECLARE(
  nl_445_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#531 */
AI_ARRAY_OBJ_DECLARE(
  gemm_446_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#532 */
AI_ARRAY_OBJ_DECLARE(
  gemm_447_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#533 */
AI_ARRAY_OBJ_DECLARE(
  slice_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#534 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_448_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 24, AI_STATIC)

/* Array#535 */
AI_ARRAY_OBJ_DECLARE(
  constantofshape_4_const_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24, AI_STATIC)

/* Array#536 */
AI_ARRAY_OBJ_DECLARE(
  gemm_5_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#537 */
AI_ARRAY_OBJ_DECLARE(
  gemm_5_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#538 */
AI_ARRAY_OBJ_DECLARE(
  constantofshape_252_const_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#539 */
AI_ARRAY_OBJ_DECLARE(
  gemm_253_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#540 */
AI_ARRAY_OBJ_DECLARE(
  gemm_253_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#541 */
AI_ARRAY_OBJ_DECLARE(
  gemm_8_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 216, AI_STATIC)

/* Array#542 */
AI_ARRAY_OBJ_DECLARE(
  gemm_8_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#543 */
AI_ARRAY_OBJ_DECLARE(
  gemm_255_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#544 */
AI_ARRAY_OBJ_DECLARE(
  gemm_255_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#545 */
AI_ARRAY_OBJ_DECLARE(
  gemm_446_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#546 */
AI_ARRAY_OBJ_DECLARE(
  gemm_446_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#547 */
AI_ARRAY_OBJ_DECLARE(
  gemm_447_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#548 */
AI_ARRAY_OBJ_DECLARE(
  gemm_447_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#549 */
AI_ARRAY_OBJ_DECLARE(
  gemm_5_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#550 */
AI_ARRAY_OBJ_DECLARE(
  gemm_253_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#551 */
AI_ARRAY_OBJ_DECLARE(
  gemm_8_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#552 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#553 */
AI_ARRAY_OBJ_DECLARE(
  gemm_23_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#554 */
AI_ARRAY_OBJ_DECLARE(
  gemm_27_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#555 */
AI_ARRAY_OBJ_DECLARE(
  gemm_38_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#556 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#557 */
AI_ARRAY_OBJ_DECLARE(
  gemm_53_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#558 */
AI_ARRAY_OBJ_DECLARE(
  gemm_57_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#559 */
AI_ARRAY_OBJ_DECLARE(
  gemm_68_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#560 */
AI_ARRAY_OBJ_DECLARE(
  gemm_72_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#561 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#562 */
AI_ARRAY_OBJ_DECLARE(
  gemm_76_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#563 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#564 */
AI_ARRAY_OBJ_DECLARE(
  gemm_80_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#565 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#566 */
AI_ARRAY_OBJ_DECLARE(
  gemm_84_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#567 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#568 */
AI_ARRAY_OBJ_DECLARE(
  gemm_88_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#569 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#570 */
AI_ARRAY_OBJ_DECLARE(
  gemm_92_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#571 */
AI_ARRAY_OBJ_DECLARE(
  gemm_18_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#572 */
AI_ARRAY_OBJ_DECLARE(
  gemm_96_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#573 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#574 */
AI_ARRAY_OBJ_DECLARE(
  gemm_100_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#575 */
AI_ARRAY_OBJ_DECLARE(
  gemm_20_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#576 */
AI_ARRAY_OBJ_DECLARE(
  gemm_104_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#577 */
AI_ARRAY_OBJ_DECLARE(
  gemm_21_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#578 */
AI_ARRAY_OBJ_DECLARE(
  gemm_108_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#579 */
AI_ARRAY_OBJ_DECLARE(
  gemm_22_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#580 */
AI_ARRAY_OBJ_DECLARE(
  gemm_112_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#581 */
AI_ARRAY_OBJ_DECLARE(
  gemm_28_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#582 */
AI_ARRAY_OBJ_DECLARE(
  gemm_116_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#583 */
AI_ARRAY_OBJ_DECLARE(
  gemm_29_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#584 */
AI_ARRAY_OBJ_DECLARE(
  gemm_120_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#585 */
AI_ARRAY_OBJ_DECLARE(
  gemm_30_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#586 */
AI_ARRAY_OBJ_DECLARE(
  gemm_124_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#587 */
AI_ARRAY_OBJ_DECLARE(
  gemm_31_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#588 */
AI_ARRAY_OBJ_DECLARE(
  gemm_128_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#589 */
AI_ARRAY_OBJ_DECLARE(
  gemm_32_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#590 */
AI_ARRAY_OBJ_DECLARE(
  gemm_132_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#591 */
AI_ARRAY_OBJ_DECLARE(
  gemm_33_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#592 */
AI_ARRAY_OBJ_DECLARE(
  gemm_136_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#593 */
AI_ARRAY_OBJ_DECLARE(
  gemm_34_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#594 */
AI_ARRAY_OBJ_DECLARE(
  gemm_140_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#595 */
AI_ARRAY_OBJ_DECLARE(
  gemm_35_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#596 */
AI_ARRAY_OBJ_DECLARE(
  gemm_144_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#597 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#598 */
AI_ARRAY_OBJ_DECLARE(
  gemm_148_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#599 */
AI_ARRAY_OBJ_DECLARE(
  gemm_37_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#600 */
AI_ARRAY_OBJ_DECLARE(
  gemm_152_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#601 */
AI_ARRAY_OBJ_DECLARE(
  gemm_43_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#602 */
AI_ARRAY_OBJ_DECLARE(
  gemm_156_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#603 */
AI_ARRAY_OBJ_DECLARE(
  gemm_44_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#604 */
AI_ARRAY_OBJ_DECLARE(
  gemm_160_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#605 */
AI_ARRAY_OBJ_DECLARE(
  gemm_45_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#606 */
AI_ARRAY_OBJ_DECLARE(
  gemm_164_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#607 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#608 */
AI_ARRAY_OBJ_DECLARE(
  gemm_168_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#609 */
AI_ARRAY_OBJ_DECLARE(
  gemm_47_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#610 */
AI_ARRAY_OBJ_DECLARE(
  gemm_172_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#611 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#612 */
AI_ARRAY_OBJ_DECLARE(
  gemm_176_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#613 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#614 */
AI_ARRAY_OBJ_DECLARE(
  gemm_180_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#615 */
AI_ARRAY_OBJ_DECLARE(
  gemm_50_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#616 */
AI_ARRAY_OBJ_DECLARE(
  gemm_184_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#617 */
AI_ARRAY_OBJ_DECLARE(
  gemm_51_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#618 */
AI_ARRAY_OBJ_DECLARE(
  gemm_188_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#619 */
AI_ARRAY_OBJ_DECLARE(
  gemm_52_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#620 */
AI_ARRAY_OBJ_DECLARE(
  gemm_192_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#621 */
AI_ARRAY_OBJ_DECLARE(
  gemm_58_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#622 */
AI_ARRAY_OBJ_DECLARE(
  gemm_196_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#623 */
AI_ARRAY_OBJ_DECLARE(
  gemm_59_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#624 */
AI_ARRAY_OBJ_DECLARE(
  gemm_200_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#625 */
AI_ARRAY_OBJ_DECLARE(
  gemm_60_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#626 */
AI_ARRAY_OBJ_DECLARE(
  gemm_204_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#627 */
AI_ARRAY_OBJ_DECLARE(
  gemm_61_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#628 */
AI_ARRAY_OBJ_DECLARE(
  gemm_208_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#629 */
AI_ARRAY_OBJ_DECLARE(
  gemm_62_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#630 */
AI_ARRAY_OBJ_DECLARE(
  gemm_212_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#631 */
AI_ARRAY_OBJ_DECLARE(
  gemm_63_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#632 */
AI_ARRAY_OBJ_DECLARE(
  gemm_216_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#633 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#634 */
AI_ARRAY_OBJ_DECLARE(
  gemm_220_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#635 */
AI_ARRAY_OBJ_DECLARE(
  gemm_65_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#636 */
AI_ARRAY_OBJ_DECLARE(
  gemm_224_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#637 */
AI_ARRAY_OBJ_DECLARE(
  gemm_66_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#638 */
AI_ARRAY_OBJ_DECLARE(
  gemm_228_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#639 */
AI_ARRAY_OBJ_DECLARE(
  gemm_67_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#640 */
AI_ARRAY_OBJ_DECLARE(
  gemm_232_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#641 */
AI_ARRAY_OBJ_DECLARE(
  gemm_233_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#642 */
AI_ARRAY_OBJ_DECLARE(
  gemm_237_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#643 */
AI_ARRAY_OBJ_DECLARE(
  gemm_238_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#644 */
AI_ARRAY_OBJ_DECLARE(
  gemm_242_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#645 */
AI_ARRAY_OBJ_DECLARE(
  gemm_243_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 129, AI_STATIC)

/* Array#646 */
AI_ARRAY_OBJ_DECLARE(
  gemm_255_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#647 */
AI_ARRAY_OBJ_DECLARE(
  gemm_258_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#648 */
AI_ARRAY_OBJ_DECLARE(
  gemm_269_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#649 */
AI_ARRAY_OBJ_DECLARE(
  gemm_272_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#650 */
AI_ARRAY_OBJ_DECLARE(
  gemm_283_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#651 */
AI_ARRAY_OBJ_DECLARE(
  gemm_286_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#652 */
AI_ARRAY_OBJ_DECLARE(
  gemm_297_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#653 */
AI_ARRAY_OBJ_DECLARE(
  gemm_300_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#654 */
AI_ARRAY_OBJ_DECLARE(
  gemm_311_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#655 */
AI_ARRAY_OBJ_DECLARE(
  gemm_314_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#656 */
AI_ARRAY_OBJ_DECLARE(
  gemm_259_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#657 */
AI_ARRAY_OBJ_DECLARE(
  gemm_317_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#658 */
AI_ARRAY_OBJ_DECLARE(
  gemm_260_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#659 */
AI_ARRAY_OBJ_DECLARE(
  gemm_320_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#660 */
AI_ARRAY_OBJ_DECLARE(
  gemm_261_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#661 */
AI_ARRAY_OBJ_DECLARE(
  gemm_323_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#662 */
AI_ARRAY_OBJ_DECLARE(
  gemm_262_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#663 */
AI_ARRAY_OBJ_DECLARE(
  gemm_326_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#664 */
AI_ARRAY_OBJ_DECLARE(
  gemm_263_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#665 */
AI_ARRAY_OBJ_DECLARE(
  gemm_329_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#666 */
AI_ARRAY_OBJ_DECLARE(
  gemm_264_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#667 */
AI_ARRAY_OBJ_DECLARE(
  gemm_332_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#668 */
AI_ARRAY_OBJ_DECLARE(
  gemm_265_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#669 */
AI_ARRAY_OBJ_DECLARE(
  gemm_335_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#670 */
AI_ARRAY_OBJ_DECLARE(
  gemm_266_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#671 */
AI_ARRAY_OBJ_DECLARE(
  gemm_338_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#672 */
AI_ARRAY_OBJ_DECLARE(
  gemm_267_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#673 */
AI_ARRAY_OBJ_DECLARE(
  gemm_341_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#674 */
AI_ARRAY_OBJ_DECLARE(
  gemm_268_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#675 */
AI_ARRAY_OBJ_DECLARE(
  gemm_344_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#676 */
AI_ARRAY_OBJ_DECLARE(
  gemm_273_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#677 */
AI_ARRAY_OBJ_DECLARE(
  gemm_347_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#678 */
AI_ARRAY_OBJ_DECLARE(
  gemm_274_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#679 */
AI_ARRAY_OBJ_DECLARE(
  gemm_350_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#680 */
AI_ARRAY_OBJ_DECLARE(
  gemm_275_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#681 */
AI_ARRAY_OBJ_DECLARE(
  gemm_353_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#682 */
AI_ARRAY_OBJ_DECLARE(
  gemm_276_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#683 */
AI_ARRAY_OBJ_DECLARE(
  gemm_356_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#684 */
AI_ARRAY_OBJ_DECLARE(
  gemm_277_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#685 */
AI_ARRAY_OBJ_DECLARE(
  gemm_359_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#686 */
AI_ARRAY_OBJ_DECLARE(
  gemm_278_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#687 */
AI_ARRAY_OBJ_DECLARE(
  gemm_362_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#688 */
AI_ARRAY_OBJ_DECLARE(
  gemm_279_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#689 */
AI_ARRAY_OBJ_DECLARE(
  gemm_365_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#690 */
AI_ARRAY_OBJ_DECLARE(
  gemm_280_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#691 */
AI_ARRAY_OBJ_DECLARE(
  gemm_368_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#692 */
AI_ARRAY_OBJ_DECLARE(
  gemm_281_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#693 */
AI_ARRAY_OBJ_DECLARE(
  gemm_371_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#694 */
AI_ARRAY_OBJ_DECLARE(
  gemm_282_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#695 */
AI_ARRAY_OBJ_DECLARE(
  gemm_374_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#696 */
AI_ARRAY_OBJ_DECLARE(
  gemm_287_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#697 */
AI_ARRAY_OBJ_DECLARE(
  gemm_377_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#698 */
AI_ARRAY_OBJ_DECLARE(
  gemm_288_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#699 */
AI_ARRAY_OBJ_DECLARE(
  gemm_380_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#700 */
AI_ARRAY_OBJ_DECLARE(
  gemm_289_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#701 */
AI_ARRAY_OBJ_DECLARE(
  gemm_383_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#702 */
AI_ARRAY_OBJ_DECLARE(
  gemm_290_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#703 */
AI_ARRAY_OBJ_DECLARE(
  gemm_386_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#704 */
AI_ARRAY_OBJ_DECLARE(
  gemm_291_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#705 */
AI_ARRAY_OBJ_DECLARE(
  gemm_389_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#706 */
AI_ARRAY_OBJ_DECLARE(
  gemm_292_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#707 */
AI_ARRAY_OBJ_DECLARE(
  gemm_392_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#708 */
AI_ARRAY_OBJ_DECLARE(
  gemm_293_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#709 */
AI_ARRAY_OBJ_DECLARE(
  gemm_395_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#710 */
AI_ARRAY_OBJ_DECLARE(
  gemm_294_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#711 */
AI_ARRAY_OBJ_DECLARE(
  gemm_398_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#712 */
AI_ARRAY_OBJ_DECLARE(
  gemm_295_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#713 */
AI_ARRAY_OBJ_DECLARE(
  gemm_401_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#714 */
AI_ARRAY_OBJ_DECLARE(
  gemm_296_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#715 */
AI_ARRAY_OBJ_DECLARE(
  gemm_404_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#716 */
AI_ARRAY_OBJ_DECLARE(
  gemm_301_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#717 */
AI_ARRAY_OBJ_DECLARE(
  gemm_407_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#718 */
AI_ARRAY_OBJ_DECLARE(
  gemm_302_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#719 */
AI_ARRAY_OBJ_DECLARE(
  gemm_410_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#720 */
AI_ARRAY_OBJ_DECLARE(
  gemm_303_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#721 */
AI_ARRAY_OBJ_DECLARE(
  gemm_413_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#722 */
AI_ARRAY_OBJ_DECLARE(
  gemm_304_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#723 */
AI_ARRAY_OBJ_DECLARE(
  gemm_416_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#724 */
AI_ARRAY_OBJ_DECLARE(
  gemm_305_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#725 */
AI_ARRAY_OBJ_DECLARE(
  gemm_419_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#726 */
AI_ARRAY_OBJ_DECLARE(
  gemm_306_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#727 */
AI_ARRAY_OBJ_DECLARE(
  gemm_422_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#728 */
AI_ARRAY_OBJ_DECLARE(
  gemm_307_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#729 */
AI_ARRAY_OBJ_DECLARE(
  gemm_425_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#730 */
AI_ARRAY_OBJ_DECLARE(
  gemm_308_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#731 */
AI_ARRAY_OBJ_DECLARE(
  gemm_428_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#732 */
AI_ARRAY_OBJ_DECLARE(
  gemm_309_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#733 */
AI_ARRAY_OBJ_DECLARE(
  gemm_431_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#734 */
AI_ARRAY_OBJ_DECLARE(
  gemm_310_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#735 */
AI_ARRAY_OBJ_DECLARE(
  gemm_434_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#736 */
AI_ARRAY_OBJ_DECLARE(
  gemm_435_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#737 */
AI_ARRAY_OBJ_DECLARE(
  gemm_438_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#738 */
AI_ARRAY_OBJ_DECLARE(
  gemm_439_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#739 */
AI_ARRAY_OBJ_DECLARE(
  gemm_442_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)

/* Array#740 */
AI_ARRAY_OBJ_DECLARE(
  gemm_443_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 104, AI_STATIC)

/* Array#741 */
AI_ARRAY_OBJ_DECLARE(
  gemm_446_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 136, AI_STATIC)

/* Array#742 */
AI_ARRAY_OBJ_DECLARE(
  gemm_447_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(constantofshape_252_const_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.84313680668447e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(constantofshape_4_const_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.84313680668447e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_103_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_107_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_111_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_115_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_119_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_123_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_127_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_131_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_135_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_139_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_143_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_147_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_151_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_155_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_159_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_163_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_167_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_171_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_175_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_179_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_183_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_187_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_191_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_195_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_199_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_203_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_207_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_211_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_215_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_219_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_223_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_227_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_231_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_236_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_241_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_246_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_26_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_41_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_56_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_71_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_75_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_79_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_83_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_87_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_91_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_95_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_99_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_101_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1459224373102188f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_105_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14640586078166962f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_109_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14542637765407562f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_113_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14511644840240479f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_117_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14324449002742767f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_121_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.145821213722229f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_125_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14769351482391357f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_129_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1384948492050171f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_133_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13382822275161743f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_137_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13660591840744019f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_141_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14210382103919983f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_145_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1394105702638626f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_149_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14449942111968994f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_153_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14013196527957916f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_157_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14050161838531494f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_161_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13503959774971008f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_165_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13770999014377594f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_169_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14209139347076416f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_173_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1442280262708664f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_177_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14471471309661865f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_181_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15314535796642303f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_185_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15448066592216492f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_189_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14416749775409698f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_193_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14594726264476776f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_197_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14765433967113495f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_201_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14986319839954376f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_205_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1464150995016098f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_209_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1483529806137085f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_213_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14720308780670166f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_217_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1469298005104065f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_221_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15314535796642303f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_225_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15448066592216492f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_229_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14416749775409698f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_234_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14449942111968994f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_239_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14160016179084778f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_244_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13954630494117737f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_24_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14403875172138214f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_256_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033157795667648315f),
    AI_PACK_INTQ_ZP(42)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_270_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04673503711819649f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_284_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049388810992240906f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_298_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0496283695101738f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_312_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04882626235485077f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #92 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_315_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04945547878742218f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #93 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_318_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0485810860991478f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #94 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_321_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04876519739627838f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #95 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_324_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04866844043135643f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #96 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_327_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049234405159950256f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #97 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_330_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05022359639406204f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #98 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_333_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04894876107573509f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #99 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_336_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04939884692430496f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #100 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_339_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04929705709218979f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #101 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_342_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04896190017461777f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #102 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_345_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04922289401292801f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #103 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_348_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04944216459989548f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #104 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_351_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04846888408064842f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #105 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_354_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928065091371536f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #106 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_357_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04885736107826233f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #107 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_360_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048645466566085815f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #108 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_363_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048294275999069214f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #109 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_366_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04873383417725563f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #110 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_369_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04866742715239525f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #111 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_372_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04845776408910751f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #112 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_375_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04878458380699158f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #113 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_378_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04956322908401489f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #114 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_381_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04892447963356972f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #115 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_384_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049033597111701965f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #116 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_387_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048594389110803604f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #117 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_390_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04864315688610077f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #118 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_393_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04853310063481331f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #119 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_396_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04812590777873993f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #120 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_399_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04903566092252731f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #121 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_39_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14398184418678284f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #122 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_402_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04841282591223717f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #123 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_405_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04863939434289932f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #124 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_408_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0486653633415699f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #125 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_411_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.050248365849256516f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #126 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_414_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04980278015136719f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #127 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_417_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04891526326537132f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #128 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_420_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04942074045538902f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #129 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_423_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04958129674196243f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #130 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_426_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04920598119497299f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #131 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_429_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049246422946453094f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #132 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_432_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0486920066177845f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #133 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_436_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04869818314909935f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #134 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_440_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048510726541280746f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #135 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_444_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04895433411002159f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #136 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_448_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.16967834532260895f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #137 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_54_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1420409083366394f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #138 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_69_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14170116186141968f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #139 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_73_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14497870206832886f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #140 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_77_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14772561192512512f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #141 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_81_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1436968296766281f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #142 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_85_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14791090786457062f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #143 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_89_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1495790034532547f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #144 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_93_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14794833958148956f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #145 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_97_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14038364589214325f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #146 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_9_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13466480374336243f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #147 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_100_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033351290971040726f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #148 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_104_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033868856728076935f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #149 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_108_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03369446471333504f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #150 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_112_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03362813591957092f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #151 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_116_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03317222744226456f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #152 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_120_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03257344663143158f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #153 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_124_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03252048045396805f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #154 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_128_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03259489685297012f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #155 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_12_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03179554641246796f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #156 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_132_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03294853866100311f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #157 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_136_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033414460718631744f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #158 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14497870206832886f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #159 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_140_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.032618749886751175f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #160 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_144_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03299355134367943f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #161 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_148_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0336167998611927f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #162 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14772561192512512f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #163 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_152_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033192213624715805f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #164 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_156_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03353745862841606f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #165 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_15_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1436968296766281f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #166 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_160_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03363793343305588f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #167 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_164_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033224064856767654f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #168 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_168_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03440818935632706f),
    AI_PACK_INTQ_ZP(11)))

/* Int quant #169 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_16_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14791090786457062f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #170 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_172_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033807992935180664f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #171 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_176_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03375629708170891f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #172 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_17_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1495790034532547f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #173 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_180_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03343703970313072f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #174 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_184_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03348264470696449f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #175 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_188_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03327428922057152f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #176 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14905594289302826f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #177 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_192_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03364004194736481f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #178 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_196_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03351854905486107f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #179 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_19_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14038364589214325f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #180 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_200_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033527061343193054f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #181 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_204_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03320958837866783f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #182 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_208_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03380759060382843f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #183 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_20_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1459224373102188f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #184 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_212_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03302001953125f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #185 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_216_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03317752107977867f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #186 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_21_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14640586078166962f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #187 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_220_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03373948857188225f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #188 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_224_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0335383377969265f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #189 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_228_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03395580127835274f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #190 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_22_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14542637765407562f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #191 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_232_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0338544137775898f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #192 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_233_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14449942111968994f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #193 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_237_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033665210008621216f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #194 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_238_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14160016179084778f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #195 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_23_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14403875172138214f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #196 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_242_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03348797932267189f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #197 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_243_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13954630494117737f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #198 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_253_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.84313680668447e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #199 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_253_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003794273128733039f, 0.003279490629211068f, 0.0037987257819622755f, 0.0037911252584308386f, 0.004012898541986942f, 0.004236295353621244f, 0.003037783782929182f, 0.0040718852542340755f, 0.003254667855799198f, 0.004586542956531048f, 0.0035959293600171804f, 0.00459314277395606f, 0.004171476699411869f, 0.0040742200799286366f, 0.004442964214831591f, 0.004439712502062321f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #200 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_255_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033157795667648315f),
    AI_PACK_INTQ_ZP(42)))

/* Int quant #201 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_255_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0032345058862119913f, 0.0059101334773004055f, 0.004959672223776579f, 0.0027206626255065203f, 0.002917695790529251f, 0.003927197773009539f, 0.004650527611374855f, 0.0032837630715221167f, 0.004761279094964266f, 0.00351216783747077f, 0.005114229861646891f, 0.0034428024664521217f, 0.004543997347354889f, 0.0033459067344665527f, 0.003787273308262229f, 0.0035953784827142954f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #202 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_258_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015924278646707535f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #203 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_259_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04945547878742218f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #204 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_260_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0485810860991478f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #205 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_261_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04876519739627838f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #206 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_262_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04866844043135643f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #207 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_263_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049234405159950256f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #208 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_264_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05022359639406204f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #209 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_265_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04894876107573509f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #210 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_266_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04939884692430496f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #211 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_267_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04929705709218979f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #212 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_268_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04896190017461777f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #213 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_269_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04673503711819649f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #214 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_272_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020147167146205902f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #215 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_273_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04922289401292801f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #216 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_274_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04944216459989548f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #217 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_275_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04846888408064842f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #218 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_276_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04928065091371536f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #219 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_277_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04885736107826233f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #220 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_278_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048645466566085815f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #221 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_279_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048294275999069214f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #222 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_27_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03329376503825188f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #223 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_280_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04873383417725563f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #224 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_281_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04866742715239525f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #225 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_282_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04845776408910751f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #226 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_283_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049388810992240906f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #227 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_286_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020471280440688133f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #228 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_287_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04878458380699158f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #229 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_288_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04956322908401489f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #230 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_289_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04892447963356972f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #231 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_28_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14511644840240479f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #232 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_290_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049033597111701965f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #233 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_291_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048594389110803604f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #234 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_292_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04864315688610077f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #235 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_293_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04853310063481331f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #236 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_294_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04812590777873993f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #237 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_295_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04903566092252731f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #238 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_296_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04841282591223717f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #239 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_297_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0496283695101738f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #240 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_29_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14324449002742767f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #241 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_300_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02059045061469078f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #242 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_301_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04863939434289932f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #243 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_302_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0486653633415699f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #244 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_303_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.050248365849256516f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #245 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_304_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04980278015136719f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #246 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_305_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04891526326537132f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #247 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_306_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04942074045538902f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #248 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_307_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04958129674196243f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #249 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_308_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04920598119497299f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #250 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_309_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049246422946453094f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #251 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_30_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.145821213722229f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #252 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_310_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0486920066177845f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #253 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_311_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04882626235485077f),
    AI_PACK_INTQ_ZP(21)))

/* Int quant #254 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_314_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.019933296367526054f),
    AI_PACK_INTQ_ZP(11)))

/* Int quant #255 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_317_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020773431286215782f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #256 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_31_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14769351482391357f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #257 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_320_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020697161555290222f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #258 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_323_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020199691876769066f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #259 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_326_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020645368844270706f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #260 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_329_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020590588450431824f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #261 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_32_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1384948492050171f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #262 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_332_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020935511216521263f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #263 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_335_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020580176264047623f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #264 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_338_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021018750965595245f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #265 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_33_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13382822275161743f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #266 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_341_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021327408030629158f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #267 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_344_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02087034285068512f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #268 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_347_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02082826942205429f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #269 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_34_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13660591840744019f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #270 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_350_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02106602117419243f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #271 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_353_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021236378699541092f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #272 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_356_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021227646619081497f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #273 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_359_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02147875539958477f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #274 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_35_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14210382103919983f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #275 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_362_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021166255697607994f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #276 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_365_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020222174003720284f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #277 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_368_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02062174305319786f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #278 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_36_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1394105702638626f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #279 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_371_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02072114683687687f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #280 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_374_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02078719064593315f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #281 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_377_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0211210697889328f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #282 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_37_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14449942111968994f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #283 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_380_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021272962912917137f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #284 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_383_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02133076637983322f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #285 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_386_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021214237436652184f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #286 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_389_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02104266546666622f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #287 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_38_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14398184418678284f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #288 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_392_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02103136107325554f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #289 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_395_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02127796784043312f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #290 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_398_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021179573610424995f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #291 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_401_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02108260802924633f),
    AI_PACK_INTQ_ZP(11)))

/* Int quant #292 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_404_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02072712406516075f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #293 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_407_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021143119782209396f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #294 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_410_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02160804159939289f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #295 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_413_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021872423589229584f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #296 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_416_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021877501159906387f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #297 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_419_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021580494940280914f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #298 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_422_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021045291796326637f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #299 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_425_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021038061007857323f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #300 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_428_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021040767431259155f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #301 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_42_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03299371153116226f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #302 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_431_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020771102979779243f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #303 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_434_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020592069253325462f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #304 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_435_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04869818314909935f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #305 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_438_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021042663604021072f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #306 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_439_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.048510726541280746f),
    AI_PACK_INTQ_ZP(22)))

/* Int quant #307 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_43_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14013196527957916f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #308 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_442_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02073649875819683f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #309 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_443_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04895433411002159f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #310 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_446_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014086093753576279f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #311 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_446_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002680764999240637f, 0.002962835831567645f, 0.0028999035712331533f, 0.004030217416584492f, 0.002725389087572694f, 0.004215550143271685f, 0.002720894291996956f, 0.0035466267727315426f, 0.004344843327999115f, 0.004093167372047901f, 0.0027702697552740574f, 0.0029316304717212915f, 0.002696215407922864f, 0.0030404389835894108f, 0.0032770861871540546f, 0.003376424079760909f, 0.0032204699236899614f, 0.0038065684493631124f, 0.004281200002878904f, 0.003128824522718787f, 0.003442191518843174f, 0.002310307929292321f, 0.002815174637362361f, 0.00469117471948266f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #312 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_447_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04078159108757973f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #313 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_447_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0054322839714586735f, 0.006037342362105846f, 0.004473595879971981f, 0.004981787875294685f, 0.00439083157107234f, 0.003839573124423623f, 0.003957822453230619f, 0.004163348115980625f, 0.003947146702557802f, 0.00317748193629086f, 0.0036274478770792484f, 0.0038265110924839973f, 0.002671311842277646f, 0.0036248101387172937f, 0.002913307398557663f, 0.004244198556989431f, 0.0033317122142761946f, 0.003740095067769289f, 0.0035288496874272823f, 0.0032735136337578297f, 0.003210596274584532f, 0.0029317194130271673f, 0.004733918234705925f, 0.003379435045644641f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #314 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_44_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14050161838531494f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #315 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_45_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13503959774971008f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #316 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_46_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13770999014377594f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #317 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_47_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14209139347076416f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #318 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_48_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1442280262708664f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #319 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_49_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14471471309661865f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #320 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_50_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15314535796642303f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #321 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_51_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15448066592216492f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #322 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_52_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14416749775409698f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #323 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_53_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1420409083366394f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #324 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_57_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033666905015707016f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #325 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_58_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14594726264476776f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #326 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_59_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14765433967113495f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #327 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_5_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.84313680668447e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #328 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_5_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0033205922227352858f, 0.005689613055437803f, 0.0040229568257927895f, 0.004418355878442526f, 0.004911307245492935f, 0.00324625289067626f, 0.003242935286834836f, 0.004242394585162401f, 0.004595539532601833f, 0.0044973138719797134f, 0.004394478164613247f, 0.004254731349647045f, 0.004767614882439375f, 0.0035326748620718718f, 0.0035000997595489025f, 0.004415825009346008f, 0.0033076214604079723f, 0.0041897972114384174f, 0.004358816891908646f, 0.004435154143720865f, 0.003699507797136903f, 0.0051195379346609116f, 0.005581617821007967f, 0.004093503579497337f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #329 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_60_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14986319839954376f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #330 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_61_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1464150995016098f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #331 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_62_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1483529806137085f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #332 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_63_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14720308780670166f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #333 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_64_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1469298005104065f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #334 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_65_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15314535796642303f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #335 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_66_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15448066592216492f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #336 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_67_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14416749775409698f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #337 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_68_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14170116186141968f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #338 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_72_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03337807208299637f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #339 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_76_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033825427293777466f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #340 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_80_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03341631963849068f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #341 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_84_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03374583646655083f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #342 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_88_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03387478366494179f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #343 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_8_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13466480374336243f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #344 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_8_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004125533159822226f, 0.0038368920795619488f, 0.005083360709249973f, 0.005824336316436529f, 0.004348632413893938f, 0.0032804342918097973f, 0.003162828739732504f, 0.004831521771848202f, 0.003480293555185199f, 0.0045130266807973385f, 0.003162180073559284f, 0.007346419617533684f, 0.0036417157389223576f, 0.0031533834990113974f, 0.0028604634571820498f, 0.003323380369693041f, 0.004513463005423546f, 0.003326460486277938f, 0.004944044630974531f, 0.0057174814864993095f, 0.0029635438695549965f, 0.0050498261116445065f, 0.004391995724290609f, 0.005709606222808361f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #345 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_92_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03396844118833542f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #346 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_96_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033817801624536514f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #347 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_102_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #348 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_106_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #349 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #350 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_110_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #351 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_114_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #352 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_118_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #353 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_122_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #354 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_126_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #355 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_130_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #356 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_134_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #357 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_138_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #358 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_142_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #359 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_146_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #360 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_150_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #361 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_154_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #362 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_158_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #363 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_162_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #364 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_166_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #365 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_170_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #366 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_174_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #367 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_178_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #368 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_182_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #369 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_186_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #370 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_190_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #371 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_194_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #372 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_198_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #373 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_202_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #374 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_206_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #375 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_210_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #376 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_214_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #377 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_218_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #378 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_222_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #379 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_226_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #380 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_230_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #381 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_235_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #382 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_240_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #383 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_245_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #384 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_257_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #385 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_25_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #386 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_271_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #387 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_285_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #388 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_299_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #389 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_313_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #390 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_316_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #391 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_319_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #392 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_322_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #393 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_325_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #394 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_328_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #395 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_331_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #396 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_334_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #397 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_337_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #398 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_340_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #399 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_343_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #400 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_346_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #401 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_349_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #402 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_352_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #403 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_355_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #404 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_358_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #405 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_361_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #406 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_364_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #407 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_367_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #408 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_370_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #409 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_373_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #410 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_376_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #411 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_379_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #412 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_382_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #413 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_385_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #414 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_388_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #415 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_391_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #416 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_394_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #417 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_397_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #418 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_400_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #419 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_403_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #420 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_406_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #421 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_409_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #422 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_40_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #423 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_412_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #424 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_415_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #425 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_418_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #426 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_421_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #427 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_424_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #428 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_427_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #429 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_430_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #430 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_433_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #431 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_437_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #432 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_441_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #433 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_445_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #434 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_55_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #435 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_70_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #436 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_74_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #437 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_78_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #438 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_82_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #439 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_86_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #440 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_90_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #441 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_94_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #442 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_98_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #443 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pack_247_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #444 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_sensor_history0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #445 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #446 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_6_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #447 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output0_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #448 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #449 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output10_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #450 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output11_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #451 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output12_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #452 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output13_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #453 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output14_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #454 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output15_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #455 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output16_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #456 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output17_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #457 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output18_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #458 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output19_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #459 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output2_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #460 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output20_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #461 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output21_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #462 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output22_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #463 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output23_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #464 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output24_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #465 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output25_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #466 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output26_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #467 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output27_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #468 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output28_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #469 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output29_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #470 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output3_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #471 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output30_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #472 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output31_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #473 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output32_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #474 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output33_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #475 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output34_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #476 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output35_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #477 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output36_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #478 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output37_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #479 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output38_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #480 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output39_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #481 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output4_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #482 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output40_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #483 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output41_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #484 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output42_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #485 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output43_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #486 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output44_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #487 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output45_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #488 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output46_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #489 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output47_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #490 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output5_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #491 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output6_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #492 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output7_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #493 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output8_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #494 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_254_output9_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #495 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output0_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #496 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #497 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output10_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #498 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output11_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #499 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output12_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #500 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output13_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #501 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output14_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #502 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output15_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #503 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output16_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #504 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output17_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #505 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output18_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #506 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output19_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #507 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output2_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #508 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output20_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #509 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output21_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #510 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output22_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #511 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output23_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #512 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output24_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #513 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output25_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #514 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output26_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #515 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output27_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #516 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output28_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #517 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output29_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #518 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output3_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #519 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output30_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #520 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output31_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #521 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output32_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #522 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output33_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #523 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output34_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #524 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output35_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #525 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output36_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #526 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output37_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #527 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output38_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #528 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output39_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #529 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output4_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #530 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output40_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #531 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output41_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #532 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output42_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #533 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output43_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #534 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output44_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #535 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output45_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #536 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output46_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #537 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output47_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #538 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output5_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #539 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output6_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #540 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output7_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #541 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output8_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/* Int quant #542 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unpack_7_output9_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19839172065258026f),
    AI_PACK_INTQ_ZP(-92)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  constantofshape_252_const, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &constantofshape_252_const_array, &constantofshape_252_const_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  constantofshape_4_const, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &constantofshape_4_const_array, &constantofshape_4_const_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conversion_103_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_103_output_array, &conversion_103_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conversion_107_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_107_output_array, &conversion_107_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conversion_111_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_111_output_array, &conversion_111_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conversion_115_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_115_output_array, &conversion_115_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conversion_119_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_119_output_array, &conversion_119_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conversion_11_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_11_output_array, &conversion_11_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conversion_123_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_123_output_array, &conversion_123_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conversion_127_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_127_output_array, &conversion_127_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conversion_131_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_131_output_array, &conversion_131_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conversion_135_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_135_output_array, &conversion_135_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conversion_139_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_139_output_array, &conversion_139_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conversion_143_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_143_output_array, &conversion_143_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conversion_147_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_147_output_array, &conversion_147_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conversion_151_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_151_output_array, &conversion_151_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conversion_155_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_155_output_array, &conversion_155_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conversion_159_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_159_output_array, &conversion_159_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conversion_163_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_163_output_array, &conversion_163_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conversion_167_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_167_output_array, &conversion_167_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conversion_171_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_171_output_array, &conversion_171_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conversion_175_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_175_output_array, &conversion_175_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conversion_179_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_179_output_array, &conversion_179_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conversion_183_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_183_output_array, &conversion_183_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conversion_187_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_187_output_array, &conversion_187_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conversion_191_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_191_output_array, &conversion_191_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conversion_195_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_195_output_array, &conversion_195_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conversion_199_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_199_output_array, &conversion_199_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conversion_203_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_203_output_array, &conversion_203_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conversion_207_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_207_output_array, &conversion_207_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conversion_211_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_211_output_array, &conversion_211_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conversion_215_output, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_215_output_array, &conversion_215_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conversion_219_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_219_output_array, &conversion_219_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conversion_223_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_223_output_array, &conversion_223_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conversion_227_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_227_output_array, &conversion_227_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conversion_231_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_231_output_array, &conversion_231_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conversion_236_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_236_output_array, &conversion_236_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conversion_241_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_241_output_array, &conversion_241_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conversion_246_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_246_output_array, &conversion_246_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conversion_26_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_26_output_array, &conversion_26_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conversion_41_output, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_41_output_array, &conversion_41_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conversion_56_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_56_output_array, &conversion_56_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conversion_71_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_71_output_array, &conversion_71_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conversion_75_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_75_output_array, &conversion_75_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conversion_79_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_79_output_array, &conversion_79_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conversion_83_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_83_output_array, &conversion_83_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conversion_87_output, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_87_output_array, &conversion_87_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conversion_91_output, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_91_output_array, &conversion_91_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conversion_95_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_95_output_array, &conversion_95_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conversion_99_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &conversion_99_output_array, &conversion_99_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_101_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_101_output_array, &eltwise_101_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_105_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_105_output_array, &eltwise_105_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_109_output, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_109_output_array, &eltwise_109_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_113_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_113_output_array, &eltwise_113_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_117_output, AI_STATIC,
  54, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_117_output_array, &eltwise_117_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_121_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_121_output_array, &eltwise_121_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_125_output, AI_STATIC,
  56, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_125_output_array, &eltwise_125_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_129_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_129_output_array, &eltwise_129_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_133_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_133_output_array, &eltwise_133_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_137_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_137_output_array, &eltwise_137_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_141_output, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_141_output_array, &eltwise_141_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_145_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_145_output_array, &eltwise_145_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_149_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_149_output_array, &eltwise_149_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_153_output, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_153_output_array, &eltwise_153_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_157_output, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_157_output_array, &eltwise_157_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_161_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_161_output_array, &eltwise_161_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_165_output, AI_STATIC,
  66, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_165_output_array, &eltwise_165_output_array_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_169_output, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_169_output_array, &eltwise_169_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_173_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_173_output_array, &eltwise_173_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_177_output, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_177_output_array, &eltwise_177_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_181_output, AI_STATIC,
  70, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_181_output_array, &eltwise_181_output_array_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_185_output, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_185_output_array, &eltwise_185_output_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_189_output, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_189_output_array, &eltwise_189_output_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_193_output, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_193_output_array, &eltwise_193_output_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_197_output, AI_STATIC,
  74, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_197_output_array, &eltwise_197_output_array_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_201_output, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_201_output_array, &eltwise_201_output_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_205_output, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_205_output_array, &eltwise_205_output_array_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_209_output, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_209_output_array, &eltwise_209_output_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_213_output, AI_STATIC,
  78, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_213_output_array, &eltwise_213_output_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_217_output, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_217_output_array, &eltwise_217_output_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_221_output, AI_STATIC,
  80, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_221_output_array, &eltwise_221_output_array_intq)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_225_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_225_output_array, &eltwise_225_output_array_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_229_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_229_output_array, &eltwise_229_output_array_intq)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_234_output, AI_STATIC,
  83, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_234_output_array, &eltwise_234_output_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_239_output, AI_STATIC,
  84, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_239_output_array, &eltwise_239_output_array_intq)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_244_output, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_244_output_array, &eltwise_244_output_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_24_output, AI_STATIC,
  86, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_24_output_array, &eltwise_24_output_array_intq)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_256_output, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_256_output_array, &eltwise_256_output_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_270_output, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_270_output_array, &eltwise_270_output_array_intq)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_284_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_284_output_array, &eltwise_284_output_array_intq)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_298_output, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_298_output_array, &eltwise_298_output_array_intq)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_312_output, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_312_output_array, &eltwise_312_output_array_intq)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_315_output, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_315_output_array, &eltwise_315_output_array_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_318_output, AI_STATIC,
  93, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_318_output_array, &eltwise_318_output_array_intq)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_321_output, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_321_output_array, &eltwise_321_output_array_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_324_output, AI_STATIC,
  95, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_324_output_array, &eltwise_324_output_array_intq)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_327_output, AI_STATIC,
  96, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_327_output_array, &eltwise_327_output_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_330_output, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_330_output_array, &eltwise_330_output_array_intq)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_333_output, AI_STATIC,
  98, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_333_output_array, &eltwise_333_output_array_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_336_output, AI_STATIC,
  99, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_336_output_array, &eltwise_336_output_array_intq)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_339_output, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_339_output_array, &eltwise_339_output_array_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_342_output, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_342_output_array, &eltwise_342_output_array_intq)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_345_output, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_345_output_array, &eltwise_345_output_array_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_348_output, AI_STATIC,
  103, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_348_output_array, &eltwise_348_output_array_intq)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_351_output, AI_STATIC,
  104, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_351_output_array, &eltwise_351_output_array_intq)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_354_output, AI_STATIC,
  105, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_354_output_array, &eltwise_354_output_array_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_357_output, AI_STATIC,
  106, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_357_output_array, &eltwise_357_output_array_intq)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_360_output, AI_STATIC,
  107, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_360_output_array, &eltwise_360_output_array_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_363_output, AI_STATIC,
  108, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_363_output_array, &eltwise_363_output_array_intq)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_366_output, AI_STATIC,
  109, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_366_output_array, &eltwise_366_output_array_intq)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_369_output, AI_STATIC,
  110, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_369_output_array, &eltwise_369_output_array_intq)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_372_output, AI_STATIC,
  111, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_372_output_array, &eltwise_372_output_array_intq)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_375_output, AI_STATIC,
  112, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_375_output_array, &eltwise_375_output_array_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_378_output, AI_STATIC,
  113, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_378_output_array, &eltwise_378_output_array_intq)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_381_output, AI_STATIC,
  114, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_381_output_array, &eltwise_381_output_array_intq)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_384_output, AI_STATIC,
  115, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_384_output_array, &eltwise_384_output_array_intq)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_387_output, AI_STATIC,
  116, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_387_output_array, &eltwise_387_output_array_intq)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_390_output, AI_STATIC,
  117, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_390_output_array, &eltwise_390_output_array_intq)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_393_output, AI_STATIC,
  118, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_393_output_array, &eltwise_393_output_array_intq)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_396_output, AI_STATIC,
  119, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_396_output_array, &eltwise_396_output_array_intq)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_399_output, AI_STATIC,
  120, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_399_output_array, &eltwise_399_output_array_intq)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_39_output, AI_STATIC,
  121, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_39_output_array, &eltwise_39_output_array_intq)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_402_output, AI_STATIC,
  122, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_402_output_array, &eltwise_402_output_array_intq)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_405_output, AI_STATIC,
  123, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_405_output_array, &eltwise_405_output_array_intq)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_408_output, AI_STATIC,
  124, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_408_output_array, &eltwise_408_output_array_intq)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_411_output, AI_STATIC,
  125, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_411_output_array, &eltwise_411_output_array_intq)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_414_output, AI_STATIC,
  126, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_414_output_array, &eltwise_414_output_array_intq)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_417_output, AI_STATIC,
  127, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_417_output_array, &eltwise_417_output_array_intq)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_420_output, AI_STATIC,
  128, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_420_output_array, &eltwise_420_output_array_intq)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_423_output, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_423_output_array, &eltwise_423_output_array_intq)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_426_output, AI_STATIC,
  130, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_426_output_array, &eltwise_426_output_array_intq)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_429_output, AI_STATIC,
  131, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_429_output_array, &eltwise_429_output_array_intq)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_432_output, AI_STATIC,
  132, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_432_output_array, &eltwise_432_output_array_intq)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_436_output, AI_STATIC,
  133, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_436_output_array, &eltwise_436_output_array_intq)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_440_output, AI_STATIC,
  134, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_440_output_array, &eltwise_440_output_array_intq)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_444_output, AI_STATIC,
  135, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &eltwise_444_output_array, &eltwise_444_output_array_intq)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_448_output, AI_STATIC,
  136, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_448_output_array, &eltwise_448_output_array_intq)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_54_output, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_54_output_array, &eltwise_54_output_array_intq)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_69_output, AI_STATIC,
  138, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_69_output_array, &eltwise_69_output_array_intq)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_73_output, AI_STATIC,
  139, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_73_output_array, &eltwise_73_output_array_intq)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_77_output, AI_STATIC,
  140, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_77_output_array, &eltwise_77_output_array_intq)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_81_output, AI_STATIC,
  141, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_81_output_array, &eltwise_81_output_array_intq)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_85_output, AI_STATIC,
  142, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_85_output_array, &eltwise_85_output_array_intq)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_89_output, AI_STATIC,
  143, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_89_output_array, &eltwise_89_output_array_intq)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_93_output, AI_STATIC,
  144, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_93_output_array, &eltwise_93_output_array_intq)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_97_output, AI_STATIC,
  145, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_97_output_array, &eltwise_97_output_array_intq)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_9_output, AI_STATIC,
  146, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &eltwise_9_output_array, &eltwise_9_output_array_intq)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  gemm_100_output, AI_STATIC,
  147, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_100_output_array, &gemm_100_output_array_intq)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  gemm_100_scratch0, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_100_scratch0_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  gemm_104_output, AI_STATIC,
  149, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_104_output_array, &gemm_104_output_array_intq)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  gemm_104_scratch0, AI_STATIC,
  150, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_104_scratch0_array, NULL)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  gemm_108_output, AI_STATIC,
  151, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_108_output_array, &gemm_108_output_array_intq)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  gemm_108_scratch0, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_108_scratch0_array, NULL)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  gemm_112_output, AI_STATIC,
  153, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_112_output_array, &gemm_112_output_array_intq)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  gemm_112_scratch0, AI_STATIC,
  154, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_112_scratch0_array, NULL)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  gemm_116_output, AI_STATIC,
  155, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_116_output_array, &gemm_116_output_array_intq)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  gemm_116_scratch0, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_116_scratch0_array, NULL)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  gemm_120_output, AI_STATIC,
  157, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_120_output_array, &gemm_120_output_array_intq)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  gemm_120_scratch0, AI_STATIC,
  158, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_120_scratch0_array, NULL)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  gemm_124_output, AI_STATIC,
  159, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_124_output_array, &gemm_124_output_array_intq)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  gemm_124_scratch0, AI_STATIC,
  160, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_124_scratch0_array, NULL)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  gemm_128_output, AI_STATIC,
  161, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_128_output_array, &gemm_128_output_array_intq)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  gemm_128_scratch0, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_128_scratch0_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_output, AI_STATIC,
  163, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_12_output_array, &gemm_12_output_array_intq)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_scratch0, AI_STATIC,
  164, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_12_scratch0_array, NULL)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  gemm_132_output, AI_STATIC,
  165, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_132_output_array, &gemm_132_output_array_intq)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  gemm_132_scratch0, AI_STATIC,
  166, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_132_scratch0_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  gemm_136_output, AI_STATIC,
  167, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_136_output_array, &gemm_136_output_array_intq)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  gemm_136_scratch0, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_136_scratch0_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_output, AI_STATIC,
  169, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_13_output_array, &gemm_13_output_array_intq)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_scratch0, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_13_scratch0_array, NULL)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  gemm_140_output, AI_STATIC,
  171, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_140_output_array, &gemm_140_output_array_intq)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  gemm_140_scratch0, AI_STATIC,
  172, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_140_scratch0_array, NULL)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  gemm_144_output, AI_STATIC,
  173, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_144_output_array, &gemm_144_output_array_intq)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  gemm_144_scratch0, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_144_scratch0_array, NULL)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  gemm_148_output, AI_STATIC,
  175, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_148_output_array, &gemm_148_output_array_intq)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  gemm_148_scratch0, AI_STATIC,
  176, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_148_scratch0_array, NULL)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output, AI_STATIC,
  177, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_14_output_array, &gemm_14_output_array_intq)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_scratch0, AI_STATIC,
  178, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_14_scratch0_array, NULL)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  gemm_152_output, AI_STATIC,
  179, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_152_output_array, &gemm_152_output_array_intq)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  gemm_152_scratch0, AI_STATIC,
  180, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_152_scratch0_array, NULL)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  gemm_156_output, AI_STATIC,
  181, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_156_output_array, &gemm_156_output_array_intq)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  gemm_156_scratch0, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_156_scratch0_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_output, AI_STATIC,
  183, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_15_output_array, &gemm_15_output_array_intq)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_scratch0, AI_STATIC,
  184, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_15_scratch0_array, NULL)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  gemm_160_output, AI_STATIC,
  185, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_160_output_array, &gemm_160_output_array_intq)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  gemm_160_scratch0, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_160_scratch0_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  gemm_164_output, AI_STATIC,
  187, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_164_output_array, &gemm_164_output_array_intq)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  gemm_164_scratch0, AI_STATIC,
  188, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_164_scratch0_array, NULL)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  gemm_168_output, AI_STATIC,
  189, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_168_output_array, &gemm_168_output_array_intq)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  gemm_168_scratch0, AI_STATIC,
  190, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_168_scratch0_array, NULL)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_output, AI_STATIC,
  191, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_16_output_array, &gemm_16_output_array_intq)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_scratch0, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_16_scratch0_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  gemm_172_output, AI_STATIC,
  193, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_172_output_array, &gemm_172_output_array_intq)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  gemm_172_scratch0, AI_STATIC,
  194, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_172_scratch0_array, NULL)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  gemm_176_output, AI_STATIC,
  195, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_176_output_array, &gemm_176_output_array_intq)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  gemm_176_scratch0, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_176_scratch0_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_output, AI_STATIC,
  197, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_17_output_array, &gemm_17_output_array_intq)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_scratch0, AI_STATIC,
  198, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_17_scratch0_array, NULL)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  gemm_180_output, AI_STATIC,
  199, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_180_output_array, &gemm_180_output_array_intq)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  gemm_180_scratch0, AI_STATIC,
  200, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_180_scratch0_array, NULL)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  gemm_184_output, AI_STATIC,
  201, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_184_output_array, &gemm_184_output_array_intq)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  gemm_184_scratch0, AI_STATIC,
  202, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_184_scratch0_array, NULL)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  gemm_188_output, AI_STATIC,
  203, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_188_output_array, &gemm_188_output_array_intq)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  gemm_188_scratch0, AI_STATIC,
  204, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_188_scratch0_array, NULL)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  gemm_18_output, AI_STATIC,
  205, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_18_output_array, &gemm_18_output_array_intq)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  gemm_18_scratch0, AI_STATIC,
  206, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_18_scratch0_array, NULL)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  gemm_192_output, AI_STATIC,
  207, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_192_output_array, &gemm_192_output_array_intq)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  gemm_192_scratch0, AI_STATIC,
  208, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_192_scratch0_array, NULL)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  gemm_196_output, AI_STATIC,
  209, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_196_output_array, &gemm_196_output_array_intq)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  gemm_196_scratch0, AI_STATIC,
  210, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_196_scratch0_array, NULL)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_output, AI_STATIC,
  211, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_19_output_array, &gemm_19_output_array_intq)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_scratch0, AI_STATIC,
  212, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_19_scratch0_array, NULL)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  gemm_200_output, AI_STATIC,
  213, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_200_output_array, &gemm_200_output_array_intq)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  gemm_200_scratch0, AI_STATIC,
  214, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_200_scratch0_array, NULL)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  gemm_204_output, AI_STATIC,
  215, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_204_output_array, &gemm_204_output_array_intq)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  gemm_204_scratch0, AI_STATIC,
  216, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_204_scratch0_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  gemm_208_output, AI_STATIC,
  217, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_208_output_array, &gemm_208_output_array_intq)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  gemm_208_scratch0, AI_STATIC,
  218, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_208_scratch0_array, NULL)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  gemm_20_output, AI_STATIC,
  219, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_20_output_array, &gemm_20_output_array_intq)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  gemm_20_scratch0, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_20_scratch0_array, NULL)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  gemm_212_output, AI_STATIC,
  221, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_212_output_array, &gemm_212_output_array_intq)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  gemm_212_scratch0, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_212_scratch0_array, NULL)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  gemm_216_output, AI_STATIC,
  223, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_216_output_array, &gemm_216_output_array_intq)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  gemm_216_scratch0, AI_STATIC,
  224, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_216_scratch0_array, NULL)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  gemm_21_output, AI_STATIC,
  225, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_21_output_array, &gemm_21_output_array_intq)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  gemm_21_scratch0, AI_STATIC,
  226, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_21_scratch0_array, NULL)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  gemm_220_output, AI_STATIC,
  227, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_220_output_array, &gemm_220_output_array_intq)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  gemm_220_scratch0, AI_STATIC,
  228, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_220_scratch0_array, NULL)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  gemm_224_output, AI_STATIC,
  229, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_224_output_array, &gemm_224_output_array_intq)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  gemm_224_scratch0, AI_STATIC,
  230, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_224_scratch0_array, NULL)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  gemm_228_output, AI_STATIC,
  231, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_228_output_array, &gemm_228_output_array_intq)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  gemm_228_scratch0, AI_STATIC,
  232, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_228_scratch0_array, NULL)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  gemm_22_output, AI_STATIC,
  233, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_22_output_array, &gemm_22_output_array_intq)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  gemm_22_scratch0, AI_STATIC,
  234, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_22_scratch0_array, NULL)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  gemm_232_output, AI_STATIC,
  235, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_232_output_array, &gemm_232_output_array_intq)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  gemm_232_scratch0, AI_STATIC,
  236, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_232_scratch0_array, NULL)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  gemm_233_output, AI_STATIC,
  237, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_233_output_array, &gemm_233_output_array_intq)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  gemm_233_scratch0, AI_STATIC,
  238, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_233_scratch0_array, NULL)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  gemm_237_output, AI_STATIC,
  239, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_237_output_array, &gemm_237_output_array_intq)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  gemm_237_scratch0, AI_STATIC,
  240, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_237_scratch0_array, NULL)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  gemm_238_output, AI_STATIC,
  241, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_238_output_array, &gemm_238_output_array_intq)

/* Tensor #242 */
AI_TENSOR_OBJ_DECLARE(
  gemm_238_scratch0, AI_STATIC,
  242, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_238_scratch0_array, NULL)

/* Tensor #243 */
AI_TENSOR_OBJ_DECLARE(
  gemm_23_output, AI_STATIC,
  243, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_23_output_array, &gemm_23_output_array_intq)

/* Tensor #244 */
AI_TENSOR_OBJ_DECLARE(
  gemm_23_scratch0, AI_STATIC,
  244, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_23_scratch0_array, NULL)

/* Tensor #245 */
AI_TENSOR_OBJ_DECLARE(
  gemm_242_output, AI_STATIC,
  245, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_242_output_array, &gemm_242_output_array_intq)

/* Tensor #246 */
AI_TENSOR_OBJ_DECLARE(
  gemm_242_scratch0, AI_STATIC,
  246, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_242_scratch0_array, NULL)

/* Tensor #247 */
AI_TENSOR_OBJ_DECLARE(
  gemm_243_output, AI_STATIC,
  247, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_243_output_array, &gemm_243_output_array_intq)

/* Tensor #248 */
AI_TENSOR_OBJ_DECLARE(
  gemm_243_scratch0, AI_STATIC,
  248, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_243_scratch0_array, NULL)

/* Tensor #249 */
AI_TENSOR_OBJ_DECLARE(
  gemm_253_bias, AI_STATIC,
  249, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_253_bias_array, NULL)

/* Tensor #250 */
AI_TENSOR_OBJ_DECLARE(
  gemm_253_output, AI_STATIC,
  250, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_253_output_array, &gemm_253_output_array_intq)

/* Tensor #251 */
AI_TENSOR_OBJ_DECLARE(
  gemm_253_scratch0, AI_STATIC,
  251, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_253_scratch0_array, NULL)

/* Tensor #252 */
AI_TENSOR_OBJ_DECLARE(
  gemm_253_weights, AI_STATIC,
  252, 0x1,
  AI_SHAPE_INIT(4, 16, 16, 1, 1), AI_STRIDE_INIT(4, 1, 16, 256, 256),
  1, &gemm_253_weights_array, &gemm_253_weights_array_intq)

/* Tensor #253 */
AI_TENSOR_OBJ_DECLARE(
  gemm_255_bias, AI_STATIC,
  253, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_255_bias_array, NULL)

/* Tensor #254 */
AI_TENSOR_OBJ_DECLARE(
  gemm_255_output, AI_STATIC,
  254, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_255_output_array, &gemm_255_output_array_intq)

/* Tensor #255 */
AI_TENSOR_OBJ_DECLARE(
  gemm_255_scratch0, AI_STATIC,
  255, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_255_scratch0_array, NULL)

/* Tensor #256 */
AI_TENSOR_OBJ_DECLARE(
  gemm_255_weights, AI_STATIC,
  256, 0x1,
  AI_SHAPE_INIT(4, 24, 16, 1, 1), AI_STRIDE_INIT(4, 1, 24, 384, 384),
  1, &gemm_255_weights_array, &gemm_255_weights_array_intq)

/* Tensor #257 */
AI_TENSOR_OBJ_DECLARE(
  gemm_258_output, AI_STATIC,
  257, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_258_output_array, &gemm_258_output_array_intq)

/* Tensor #258 */
AI_TENSOR_OBJ_DECLARE(
  gemm_258_scratch0, AI_STATIC,
  258, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_258_scratch0_array, NULL)

/* Tensor #259 */
AI_TENSOR_OBJ_DECLARE(
  gemm_259_output, AI_STATIC,
  259, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_259_output_array, &gemm_259_output_array_intq)

/* Tensor #260 */
AI_TENSOR_OBJ_DECLARE(
  gemm_259_scratch0, AI_STATIC,
  260, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_259_scratch0_array, NULL)

/* Tensor #261 */
AI_TENSOR_OBJ_DECLARE(
  gemm_260_output, AI_STATIC,
  261, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_260_output_array, &gemm_260_output_array_intq)

/* Tensor #262 */
AI_TENSOR_OBJ_DECLARE(
  gemm_260_scratch0, AI_STATIC,
  262, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_260_scratch0_array, NULL)

/* Tensor #263 */
AI_TENSOR_OBJ_DECLARE(
  gemm_261_output, AI_STATIC,
  263, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_261_output_array, &gemm_261_output_array_intq)

/* Tensor #264 */
AI_TENSOR_OBJ_DECLARE(
  gemm_261_scratch0, AI_STATIC,
  264, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_261_scratch0_array, NULL)

/* Tensor #265 */
AI_TENSOR_OBJ_DECLARE(
  gemm_262_output, AI_STATIC,
  265, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_262_output_array, &gemm_262_output_array_intq)

/* Tensor #266 */
AI_TENSOR_OBJ_DECLARE(
  gemm_262_scratch0, AI_STATIC,
  266, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_262_scratch0_array, NULL)

/* Tensor #267 */
AI_TENSOR_OBJ_DECLARE(
  gemm_263_output, AI_STATIC,
  267, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_263_output_array, &gemm_263_output_array_intq)

/* Tensor #268 */
AI_TENSOR_OBJ_DECLARE(
  gemm_263_scratch0, AI_STATIC,
  268, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_263_scratch0_array, NULL)

/* Tensor #269 */
AI_TENSOR_OBJ_DECLARE(
  gemm_264_output, AI_STATIC,
  269, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_264_output_array, &gemm_264_output_array_intq)

/* Tensor #270 */
AI_TENSOR_OBJ_DECLARE(
  gemm_264_scratch0, AI_STATIC,
  270, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_264_scratch0_array, NULL)

/* Tensor #271 */
AI_TENSOR_OBJ_DECLARE(
  gemm_265_output, AI_STATIC,
  271, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_265_output_array, &gemm_265_output_array_intq)

/* Tensor #272 */
AI_TENSOR_OBJ_DECLARE(
  gemm_265_scratch0, AI_STATIC,
  272, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_265_scratch0_array, NULL)

/* Tensor #273 */
AI_TENSOR_OBJ_DECLARE(
  gemm_266_output, AI_STATIC,
  273, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_266_output_array, &gemm_266_output_array_intq)

/* Tensor #274 */
AI_TENSOR_OBJ_DECLARE(
  gemm_266_scratch0, AI_STATIC,
  274, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_266_scratch0_array, NULL)

/* Tensor #275 */
AI_TENSOR_OBJ_DECLARE(
  gemm_267_output, AI_STATIC,
  275, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_267_output_array, &gemm_267_output_array_intq)

/* Tensor #276 */
AI_TENSOR_OBJ_DECLARE(
  gemm_267_scratch0, AI_STATIC,
  276, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_267_scratch0_array, NULL)

/* Tensor #277 */
AI_TENSOR_OBJ_DECLARE(
  gemm_268_output, AI_STATIC,
  277, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_268_output_array, &gemm_268_output_array_intq)

/* Tensor #278 */
AI_TENSOR_OBJ_DECLARE(
  gemm_268_scratch0, AI_STATIC,
  278, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_268_scratch0_array, NULL)

/* Tensor #279 */
AI_TENSOR_OBJ_DECLARE(
  gemm_269_output, AI_STATIC,
  279, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_269_output_array, &gemm_269_output_array_intq)

/* Tensor #280 */
AI_TENSOR_OBJ_DECLARE(
  gemm_269_scratch0, AI_STATIC,
  280, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_269_scratch0_array, NULL)

/* Tensor #281 */
AI_TENSOR_OBJ_DECLARE(
  gemm_272_output, AI_STATIC,
  281, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_272_output_array, &gemm_272_output_array_intq)

/* Tensor #282 */
AI_TENSOR_OBJ_DECLARE(
  gemm_272_scratch0, AI_STATIC,
  282, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_272_scratch0_array, NULL)

/* Tensor #283 */
AI_TENSOR_OBJ_DECLARE(
  gemm_273_output, AI_STATIC,
  283, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_273_output_array, &gemm_273_output_array_intq)

/* Tensor #284 */
AI_TENSOR_OBJ_DECLARE(
  gemm_273_scratch0, AI_STATIC,
  284, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_273_scratch0_array, NULL)

/* Tensor #285 */
AI_TENSOR_OBJ_DECLARE(
  gemm_274_output, AI_STATIC,
  285, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_274_output_array, &gemm_274_output_array_intq)

/* Tensor #286 */
AI_TENSOR_OBJ_DECLARE(
  gemm_274_scratch0, AI_STATIC,
  286, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_274_scratch0_array, NULL)

/* Tensor #287 */
AI_TENSOR_OBJ_DECLARE(
  gemm_275_output, AI_STATIC,
  287, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_275_output_array, &gemm_275_output_array_intq)

/* Tensor #288 */
AI_TENSOR_OBJ_DECLARE(
  gemm_275_scratch0, AI_STATIC,
  288, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_275_scratch0_array, NULL)

/* Tensor #289 */
AI_TENSOR_OBJ_DECLARE(
  gemm_276_output, AI_STATIC,
  289, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_276_output_array, &gemm_276_output_array_intq)

/* Tensor #290 */
AI_TENSOR_OBJ_DECLARE(
  gemm_276_scratch0, AI_STATIC,
  290, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_276_scratch0_array, NULL)

/* Tensor #291 */
AI_TENSOR_OBJ_DECLARE(
  gemm_277_output, AI_STATIC,
  291, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_277_output_array, &gemm_277_output_array_intq)

/* Tensor #292 */
AI_TENSOR_OBJ_DECLARE(
  gemm_277_scratch0, AI_STATIC,
  292, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_277_scratch0_array, NULL)

/* Tensor #293 */
AI_TENSOR_OBJ_DECLARE(
  gemm_278_output, AI_STATIC,
  293, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_278_output_array, &gemm_278_output_array_intq)

/* Tensor #294 */
AI_TENSOR_OBJ_DECLARE(
  gemm_278_scratch0, AI_STATIC,
  294, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_278_scratch0_array, NULL)

/* Tensor #295 */
AI_TENSOR_OBJ_DECLARE(
  gemm_279_output, AI_STATIC,
  295, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_279_output_array, &gemm_279_output_array_intq)

/* Tensor #296 */
AI_TENSOR_OBJ_DECLARE(
  gemm_279_scratch0, AI_STATIC,
  296, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_279_scratch0_array, NULL)

/* Tensor #297 */
AI_TENSOR_OBJ_DECLARE(
  gemm_27_output, AI_STATIC,
  297, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_27_output_array, &gemm_27_output_array_intq)

/* Tensor #298 */
AI_TENSOR_OBJ_DECLARE(
  gemm_27_scratch0, AI_STATIC,
  298, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_27_scratch0_array, NULL)

/* Tensor #299 */
AI_TENSOR_OBJ_DECLARE(
  gemm_280_output, AI_STATIC,
  299, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_280_output_array, &gemm_280_output_array_intq)

/* Tensor #300 */
AI_TENSOR_OBJ_DECLARE(
  gemm_280_scratch0, AI_STATIC,
  300, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_280_scratch0_array, NULL)

/* Tensor #301 */
AI_TENSOR_OBJ_DECLARE(
  gemm_281_output, AI_STATIC,
  301, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_281_output_array, &gemm_281_output_array_intq)

/* Tensor #302 */
AI_TENSOR_OBJ_DECLARE(
  gemm_281_scratch0, AI_STATIC,
  302, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_281_scratch0_array, NULL)

/* Tensor #303 */
AI_TENSOR_OBJ_DECLARE(
  gemm_282_output, AI_STATIC,
  303, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_282_output_array, &gemm_282_output_array_intq)

/* Tensor #304 */
AI_TENSOR_OBJ_DECLARE(
  gemm_282_scratch0, AI_STATIC,
  304, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_282_scratch0_array, NULL)

/* Tensor #305 */
AI_TENSOR_OBJ_DECLARE(
  gemm_283_output, AI_STATIC,
  305, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_283_output_array, &gemm_283_output_array_intq)

/* Tensor #306 */
AI_TENSOR_OBJ_DECLARE(
  gemm_283_scratch0, AI_STATIC,
  306, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_283_scratch0_array, NULL)

/* Tensor #307 */
AI_TENSOR_OBJ_DECLARE(
  gemm_286_output, AI_STATIC,
  307, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_286_output_array, &gemm_286_output_array_intq)

/* Tensor #308 */
AI_TENSOR_OBJ_DECLARE(
  gemm_286_scratch0, AI_STATIC,
  308, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_286_scratch0_array, NULL)

/* Tensor #309 */
AI_TENSOR_OBJ_DECLARE(
  gemm_287_output, AI_STATIC,
  309, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_287_output_array, &gemm_287_output_array_intq)

/* Tensor #310 */
AI_TENSOR_OBJ_DECLARE(
  gemm_287_scratch0, AI_STATIC,
  310, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_287_scratch0_array, NULL)

/* Tensor #311 */
AI_TENSOR_OBJ_DECLARE(
  gemm_288_output, AI_STATIC,
  311, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_288_output_array, &gemm_288_output_array_intq)

/* Tensor #312 */
AI_TENSOR_OBJ_DECLARE(
  gemm_288_scratch0, AI_STATIC,
  312, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_288_scratch0_array, NULL)

/* Tensor #313 */
AI_TENSOR_OBJ_DECLARE(
  gemm_289_output, AI_STATIC,
  313, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_289_output_array, &gemm_289_output_array_intq)

/* Tensor #314 */
AI_TENSOR_OBJ_DECLARE(
  gemm_289_scratch0, AI_STATIC,
  314, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_289_scratch0_array, NULL)

/* Tensor #315 */
AI_TENSOR_OBJ_DECLARE(
  gemm_28_output, AI_STATIC,
  315, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_28_output_array, &gemm_28_output_array_intq)

/* Tensor #316 */
AI_TENSOR_OBJ_DECLARE(
  gemm_28_scratch0, AI_STATIC,
  316, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_28_scratch0_array, NULL)

/* Tensor #317 */
AI_TENSOR_OBJ_DECLARE(
  gemm_290_output, AI_STATIC,
  317, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_290_output_array, &gemm_290_output_array_intq)

/* Tensor #318 */
AI_TENSOR_OBJ_DECLARE(
  gemm_290_scratch0, AI_STATIC,
  318, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_290_scratch0_array, NULL)

/* Tensor #319 */
AI_TENSOR_OBJ_DECLARE(
  gemm_291_output, AI_STATIC,
  319, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_291_output_array, &gemm_291_output_array_intq)

/* Tensor #320 */
AI_TENSOR_OBJ_DECLARE(
  gemm_291_scratch0, AI_STATIC,
  320, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_291_scratch0_array, NULL)

/* Tensor #321 */
AI_TENSOR_OBJ_DECLARE(
  gemm_292_output, AI_STATIC,
  321, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_292_output_array, &gemm_292_output_array_intq)

/* Tensor #322 */
AI_TENSOR_OBJ_DECLARE(
  gemm_292_scratch0, AI_STATIC,
  322, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_292_scratch0_array, NULL)

/* Tensor #323 */
AI_TENSOR_OBJ_DECLARE(
  gemm_293_output, AI_STATIC,
  323, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_293_output_array, &gemm_293_output_array_intq)

/* Tensor #324 */
AI_TENSOR_OBJ_DECLARE(
  gemm_293_scratch0, AI_STATIC,
  324, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_293_scratch0_array, NULL)

/* Tensor #325 */
AI_TENSOR_OBJ_DECLARE(
  gemm_294_output, AI_STATIC,
  325, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_294_output_array, &gemm_294_output_array_intq)

/* Tensor #326 */
AI_TENSOR_OBJ_DECLARE(
  gemm_294_scratch0, AI_STATIC,
  326, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_294_scratch0_array, NULL)

/* Tensor #327 */
AI_TENSOR_OBJ_DECLARE(
  gemm_295_output, AI_STATIC,
  327, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_295_output_array, &gemm_295_output_array_intq)

/* Tensor #328 */
AI_TENSOR_OBJ_DECLARE(
  gemm_295_scratch0, AI_STATIC,
  328, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_295_scratch0_array, NULL)

/* Tensor #329 */
AI_TENSOR_OBJ_DECLARE(
  gemm_296_output, AI_STATIC,
  329, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_296_output_array, &gemm_296_output_array_intq)

/* Tensor #330 */
AI_TENSOR_OBJ_DECLARE(
  gemm_296_scratch0, AI_STATIC,
  330, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_296_scratch0_array, NULL)

/* Tensor #331 */
AI_TENSOR_OBJ_DECLARE(
  gemm_297_output, AI_STATIC,
  331, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_297_output_array, &gemm_297_output_array_intq)

/* Tensor #332 */
AI_TENSOR_OBJ_DECLARE(
  gemm_297_scratch0, AI_STATIC,
  332, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_297_scratch0_array, NULL)

/* Tensor #333 */
AI_TENSOR_OBJ_DECLARE(
  gemm_29_output, AI_STATIC,
  333, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_29_output_array, &gemm_29_output_array_intq)

/* Tensor #334 */
AI_TENSOR_OBJ_DECLARE(
  gemm_29_scratch0, AI_STATIC,
  334, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_29_scratch0_array, NULL)

/* Tensor #335 */
AI_TENSOR_OBJ_DECLARE(
  gemm_300_output, AI_STATIC,
  335, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_300_output_array, &gemm_300_output_array_intq)

/* Tensor #336 */
AI_TENSOR_OBJ_DECLARE(
  gemm_300_scratch0, AI_STATIC,
  336, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_300_scratch0_array, NULL)

/* Tensor #337 */
AI_TENSOR_OBJ_DECLARE(
  gemm_301_output, AI_STATIC,
  337, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_301_output_array, &gemm_301_output_array_intq)

/* Tensor #338 */
AI_TENSOR_OBJ_DECLARE(
  gemm_301_scratch0, AI_STATIC,
  338, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_301_scratch0_array, NULL)

/* Tensor #339 */
AI_TENSOR_OBJ_DECLARE(
  gemm_302_output, AI_STATIC,
  339, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_302_output_array, &gemm_302_output_array_intq)

/* Tensor #340 */
AI_TENSOR_OBJ_DECLARE(
  gemm_302_scratch0, AI_STATIC,
  340, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_302_scratch0_array, NULL)

/* Tensor #341 */
AI_TENSOR_OBJ_DECLARE(
  gemm_303_output, AI_STATIC,
  341, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_303_output_array, &gemm_303_output_array_intq)

/* Tensor #342 */
AI_TENSOR_OBJ_DECLARE(
  gemm_303_scratch0, AI_STATIC,
  342, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_303_scratch0_array, NULL)

/* Tensor #343 */
AI_TENSOR_OBJ_DECLARE(
  gemm_304_output, AI_STATIC,
  343, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_304_output_array, &gemm_304_output_array_intq)

/* Tensor #344 */
AI_TENSOR_OBJ_DECLARE(
  gemm_304_scratch0, AI_STATIC,
  344, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_304_scratch0_array, NULL)

/* Tensor #345 */
AI_TENSOR_OBJ_DECLARE(
  gemm_305_output, AI_STATIC,
  345, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_305_output_array, &gemm_305_output_array_intq)

/* Tensor #346 */
AI_TENSOR_OBJ_DECLARE(
  gemm_305_scratch0, AI_STATIC,
  346, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_305_scratch0_array, NULL)

/* Tensor #347 */
AI_TENSOR_OBJ_DECLARE(
  gemm_306_output, AI_STATIC,
  347, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_306_output_array, &gemm_306_output_array_intq)

/* Tensor #348 */
AI_TENSOR_OBJ_DECLARE(
  gemm_306_scratch0, AI_STATIC,
  348, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_306_scratch0_array, NULL)

/* Tensor #349 */
AI_TENSOR_OBJ_DECLARE(
  gemm_307_output, AI_STATIC,
  349, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_307_output_array, &gemm_307_output_array_intq)

/* Tensor #350 */
AI_TENSOR_OBJ_DECLARE(
  gemm_307_scratch0, AI_STATIC,
  350, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_307_scratch0_array, NULL)

/* Tensor #351 */
AI_TENSOR_OBJ_DECLARE(
  gemm_308_output, AI_STATIC,
  351, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_308_output_array, &gemm_308_output_array_intq)

/* Tensor #352 */
AI_TENSOR_OBJ_DECLARE(
  gemm_308_scratch0, AI_STATIC,
  352, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_308_scratch0_array, NULL)

/* Tensor #353 */
AI_TENSOR_OBJ_DECLARE(
  gemm_309_output, AI_STATIC,
  353, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_309_output_array, &gemm_309_output_array_intq)

/* Tensor #354 */
AI_TENSOR_OBJ_DECLARE(
  gemm_309_scratch0, AI_STATIC,
  354, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_309_scratch0_array, NULL)

/* Tensor #355 */
AI_TENSOR_OBJ_DECLARE(
  gemm_30_output, AI_STATIC,
  355, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_30_output_array, &gemm_30_output_array_intq)

/* Tensor #356 */
AI_TENSOR_OBJ_DECLARE(
  gemm_30_scratch0, AI_STATIC,
  356, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_30_scratch0_array, NULL)

/* Tensor #357 */
AI_TENSOR_OBJ_DECLARE(
  gemm_310_output, AI_STATIC,
  357, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_310_output_array, &gemm_310_output_array_intq)

/* Tensor #358 */
AI_TENSOR_OBJ_DECLARE(
  gemm_310_scratch0, AI_STATIC,
  358, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_310_scratch0_array, NULL)

/* Tensor #359 */
AI_TENSOR_OBJ_DECLARE(
  gemm_311_output, AI_STATIC,
  359, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_311_output_array, &gemm_311_output_array_intq)

/* Tensor #360 */
AI_TENSOR_OBJ_DECLARE(
  gemm_311_scratch0, AI_STATIC,
  360, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_311_scratch0_array, NULL)

/* Tensor #361 */
AI_TENSOR_OBJ_DECLARE(
  gemm_314_output, AI_STATIC,
  361, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_314_output_array, &gemm_314_output_array_intq)

/* Tensor #362 */
AI_TENSOR_OBJ_DECLARE(
  gemm_314_scratch0, AI_STATIC,
  362, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_314_scratch0_array, NULL)

/* Tensor #363 */
AI_TENSOR_OBJ_DECLARE(
  gemm_317_output, AI_STATIC,
  363, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_317_output_array, &gemm_317_output_array_intq)

/* Tensor #364 */
AI_TENSOR_OBJ_DECLARE(
  gemm_317_scratch0, AI_STATIC,
  364, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_317_scratch0_array, NULL)

/* Tensor #365 */
AI_TENSOR_OBJ_DECLARE(
  gemm_31_output, AI_STATIC,
  365, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_31_output_array, &gemm_31_output_array_intq)

/* Tensor #366 */
AI_TENSOR_OBJ_DECLARE(
  gemm_31_scratch0, AI_STATIC,
  366, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_31_scratch0_array, NULL)

/* Tensor #367 */
AI_TENSOR_OBJ_DECLARE(
  gemm_320_output, AI_STATIC,
  367, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_320_output_array, &gemm_320_output_array_intq)

/* Tensor #368 */
AI_TENSOR_OBJ_DECLARE(
  gemm_320_scratch0, AI_STATIC,
  368, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_320_scratch0_array, NULL)

/* Tensor #369 */
AI_TENSOR_OBJ_DECLARE(
  gemm_323_output, AI_STATIC,
  369, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_323_output_array, &gemm_323_output_array_intq)

/* Tensor #370 */
AI_TENSOR_OBJ_DECLARE(
  gemm_323_scratch0, AI_STATIC,
  370, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_323_scratch0_array, NULL)

/* Tensor #371 */
AI_TENSOR_OBJ_DECLARE(
  gemm_326_output, AI_STATIC,
  371, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_326_output_array, &gemm_326_output_array_intq)

/* Tensor #372 */
AI_TENSOR_OBJ_DECLARE(
  gemm_326_scratch0, AI_STATIC,
  372, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_326_scratch0_array, NULL)

/* Tensor #373 */
AI_TENSOR_OBJ_DECLARE(
  gemm_329_output, AI_STATIC,
  373, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_329_output_array, &gemm_329_output_array_intq)

/* Tensor #374 */
AI_TENSOR_OBJ_DECLARE(
  gemm_329_scratch0, AI_STATIC,
  374, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_329_scratch0_array, NULL)

/* Tensor #375 */
AI_TENSOR_OBJ_DECLARE(
  gemm_32_output, AI_STATIC,
  375, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_32_output_array, &gemm_32_output_array_intq)

/* Tensor #376 */
AI_TENSOR_OBJ_DECLARE(
  gemm_32_scratch0, AI_STATIC,
  376, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_32_scratch0_array, NULL)

/* Tensor #377 */
AI_TENSOR_OBJ_DECLARE(
  gemm_332_output, AI_STATIC,
  377, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_332_output_array, &gemm_332_output_array_intq)

/* Tensor #378 */
AI_TENSOR_OBJ_DECLARE(
  gemm_332_scratch0, AI_STATIC,
  378, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_332_scratch0_array, NULL)

/* Tensor #379 */
AI_TENSOR_OBJ_DECLARE(
  gemm_335_output, AI_STATIC,
  379, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_335_output_array, &gemm_335_output_array_intq)

/* Tensor #380 */
AI_TENSOR_OBJ_DECLARE(
  gemm_335_scratch0, AI_STATIC,
  380, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_335_scratch0_array, NULL)

/* Tensor #381 */
AI_TENSOR_OBJ_DECLARE(
  gemm_338_output, AI_STATIC,
  381, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_338_output_array, &gemm_338_output_array_intq)

/* Tensor #382 */
AI_TENSOR_OBJ_DECLARE(
  gemm_338_scratch0, AI_STATIC,
  382, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_338_scratch0_array, NULL)

/* Tensor #383 */
AI_TENSOR_OBJ_DECLARE(
  gemm_33_output, AI_STATIC,
  383, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_33_output_array, &gemm_33_output_array_intq)

/* Tensor #384 */
AI_TENSOR_OBJ_DECLARE(
  gemm_33_scratch0, AI_STATIC,
  384, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_33_scratch0_array, NULL)

/* Tensor #385 */
AI_TENSOR_OBJ_DECLARE(
  gemm_341_output, AI_STATIC,
  385, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_341_output_array, &gemm_341_output_array_intq)

/* Tensor #386 */
AI_TENSOR_OBJ_DECLARE(
  gemm_341_scratch0, AI_STATIC,
  386, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_341_scratch0_array, NULL)

/* Tensor #387 */
AI_TENSOR_OBJ_DECLARE(
  gemm_344_output, AI_STATIC,
  387, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_344_output_array, &gemm_344_output_array_intq)

/* Tensor #388 */
AI_TENSOR_OBJ_DECLARE(
  gemm_344_scratch0, AI_STATIC,
  388, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_344_scratch0_array, NULL)

/* Tensor #389 */
AI_TENSOR_OBJ_DECLARE(
  gemm_347_output, AI_STATIC,
  389, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_347_output_array, &gemm_347_output_array_intq)

/* Tensor #390 */
AI_TENSOR_OBJ_DECLARE(
  gemm_347_scratch0, AI_STATIC,
  390, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_347_scratch0_array, NULL)

/* Tensor #391 */
AI_TENSOR_OBJ_DECLARE(
  gemm_34_output, AI_STATIC,
  391, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_34_output_array, &gemm_34_output_array_intq)

/* Tensor #392 */
AI_TENSOR_OBJ_DECLARE(
  gemm_34_scratch0, AI_STATIC,
  392, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_34_scratch0_array, NULL)

/* Tensor #393 */
AI_TENSOR_OBJ_DECLARE(
  gemm_350_output, AI_STATIC,
  393, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_350_output_array, &gemm_350_output_array_intq)

/* Tensor #394 */
AI_TENSOR_OBJ_DECLARE(
  gemm_350_scratch0, AI_STATIC,
  394, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_350_scratch0_array, NULL)

/* Tensor #395 */
AI_TENSOR_OBJ_DECLARE(
  gemm_353_output, AI_STATIC,
  395, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_353_output_array, &gemm_353_output_array_intq)

/* Tensor #396 */
AI_TENSOR_OBJ_DECLARE(
  gemm_353_scratch0, AI_STATIC,
  396, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_353_scratch0_array, NULL)

/* Tensor #397 */
AI_TENSOR_OBJ_DECLARE(
  gemm_356_output, AI_STATIC,
  397, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_356_output_array, &gemm_356_output_array_intq)

/* Tensor #398 */
AI_TENSOR_OBJ_DECLARE(
  gemm_356_scratch0, AI_STATIC,
  398, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_356_scratch0_array, NULL)

/* Tensor #399 */
AI_TENSOR_OBJ_DECLARE(
  gemm_359_output, AI_STATIC,
  399, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_359_output_array, &gemm_359_output_array_intq)

/* Tensor #400 */
AI_TENSOR_OBJ_DECLARE(
  gemm_359_scratch0, AI_STATIC,
  400, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_359_scratch0_array, NULL)

/* Tensor #401 */
AI_TENSOR_OBJ_DECLARE(
  gemm_35_output, AI_STATIC,
  401, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_35_output_array, &gemm_35_output_array_intq)

/* Tensor #402 */
AI_TENSOR_OBJ_DECLARE(
  gemm_35_scratch0, AI_STATIC,
  402, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_35_scratch0_array, NULL)

/* Tensor #403 */
AI_TENSOR_OBJ_DECLARE(
  gemm_362_output, AI_STATIC,
  403, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_362_output_array, &gemm_362_output_array_intq)

/* Tensor #404 */
AI_TENSOR_OBJ_DECLARE(
  gemm_362_scratch0, AI_STATIC,
  404, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_362_scratch0_array, NULL)

/* Tensor #405 */
AI_TENSOR_OBJ_DECLARE(
  gemm_365_output, AI_STATIC,
  405, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_365_output_array, &gemm_365_output_array_intq)

/* Tensor #406 */
AI_TENSOR_OBJ_DECLARE(
  gemm_365_scratch0, AI_STATIC,
  406, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_365_scratch0_array, NULL)

/* Tensor #407 */
AI_TENSOR_OBJ_DECLARE(
  gemm_368_output, AI_STATIC,
  407, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_368_output_array, &gemm_368_output_array_intq)

/* Tensor #408 */
AI_TENSOR_OBJ_DECLARE(
  gemm_368_scratch0, AI_STATIC,
  408, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_368_scratch0_array, NULL)

/* Tensor #409 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_output, AI_STATIC,
  409, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_36_output_array, &gemm_36_output_array_intq)

/* Tensor #410 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_scratch0, AI_STATIC,
  410, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_36_scratch0_array, NULL)

/* Tensor #411 */
AI_TENSOR_OBJ_DECLARE(
  gemm_371_output, AI_STATIC,
  411, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_371_output_array, &gemm_371_output_array_intq)

/* Tensor #412 */
AI_TENSOR_OBJ_DECLARE(
  gemm_371_scratch0, AI_STATIC,
  412, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_371_scratch0_array, NULL)

/* Tensor #413 */
AI_TENSOR_OBJ_DECLARE(
  gemm_374_output, AI_STATIC,
  413, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_374_output_array, &gemm_374_output_array_intq)

/* Tensor #414 */
AI_TENSOR_OBJ_DECLARE(
  gemm_374_scratch0, AI_STATIC,
  414, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_374_scratch0_array, NULL)

/* Tensor #415 */
AI_TENSOR_OBJ_DECLARE(
  gemm_377_output, AI_STATIC,
  415, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_377_output_array, &gemm_377_output_array_intq)

/* Tensor #416 */
AI_TENSOR_OBJ_DECLARE(
  gemm_377_scratch0, AI_STATIC,
  416, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_377_scratch0_array, NULL)

/* Tensor #417 */
AI_TENSOR_OBJ_DECLARE(
  gemm_37_output, AI_STATIC,
  417, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_37_output_array, &gemm_37_output_array_intq)

/* Tensor #418 */
AI_TENSOR_OBJ_DECLARE(
  gemm_37_scratch0, AI_STATIC,
  418, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_37_scratch0_array, NULL)

/* Tensor #419 */
AI_TENSOR_OBJ_DECLARE(
  gemm_380_output, AI_STATIC,
  419, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_380_output_array, &gemm_380_output_array_intq)

/* Tensor #420 */
AI_TENSOR_OBJ_DECLARE(
  gemm_380_scratch0, AI_STATIC,
  420, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_380_scratch0_array, NULL)

/* Tensor #421 */
AI_TENSOR_OBJ_DECLARE(
  gemm_383_output, AI_STATIC,
  421, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_383_output_array, &gemm_383_output_array_intq)

/* Tensor #422 */
AI_TENSOR_OBJ_DECLARE(
  gemm_383_scratch0, AI_STATIC,
  422, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_383_scratch0_array, NULL)

/* Tensor #423 */
AI_TENSOR_OBJ_DECLARE(
  gemm_386_output, AI_STATIC,
  423, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_386_output_array, &gemm_386_output_array_intq)

/* Tensor #424 */
AI_TENSOR_OBJ_DECLARE(
  gemm_386_scratch0, AI_STATIC,
  424, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_386_scratch0_array, NULL)

/* Tensor #425 */
AI_TENSOR_OBJ_DECLARE(
  gemm_389_output, AI_STATIC,
  425, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_389_output_array, &gemm_389_output_array_intq)

/* Tensor #426 */
AI_TENSOR_OBJ_DECLARE(
  gemm_389_scratch0, AI_STATIC,
  426, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_389_scratch0_array, NULL)

/* Tensor #427 */
AI_TENSOR_OBJ_DECLARE(
  gemm_38_output, AI_STATIC,
  427, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_38_output_array, &gemm_38_output_array_intq)

/* Tensor #428 */
AI_TENSOR_OBJ_DECLARE(
  gemm_38_scratch0, AI_STATIC,
  428, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_38_scratch0_array, NULL)

/* Tensor #429 */
AI_TENSOR_OBJ_DECLARE(
  gemm_392_output, AI_STATIC,
  429, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_392_output_array, &gemm_392_output_array_intq)

/* Tensor #430 */
AI_TENSOR_OBJ_DECLARE(
  gemm_392_scratch0, AI_STATIC,
  430, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_392_scratch0_array, NULL)

/* Tensor #431 */
AI_TENSOR_OBJ_DECLARE(
  gemm_395_output, AI_STATIC,
  431, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_395_output_array, &gemm_395_output_array_intq)

/* Tensor #432 */
AI_TENSOR_OBJ_DECLARE(
  gemm_395_scratch0, AI_STATIC,
  432, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_395_scratch0_array, NULL)

/* Tensor #433 */
AI_TENSOR_OBJ_DECLARE(
  gemm_398_output, AI_STATIC,
  433, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_398_output_array, &gemm_398_output_array_intq)

/* Tensor #434 */
AI_TENSOR_OBJ_DECLARE(
  gemm_398_scratch0, AI_STATIC,
  434, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_398_scratch0_array, NULL)

/* Tensor #435 */
AI_TENSOR_OBJ_DECLARE(
  gemm_401_output, AI_STATIC,
  435, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_401_output_array, &gemm_401_output_array_intq)

/* Tensor #436 */
AI_TENSOR_OBJ_DECLARE(
  gemm_401_scratch0, AI_STATIC,
  436, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_401_scratch0_array, NULL)

/* Tensor #437 */
AI_TENSOR_OBJ_DECLARE(
  gemm_404_output, AI_STATIC,
  437, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_404_output_array, &gemm_404_output_array_intq)

/* Tensor #438 */
AI_TENSOR_OBJ_DECLARE(
  gemm_404_scratch0, AI_STATIC,
  438, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_404_scratch0_array, NULL)

/* Tensor #439 */
AI_TENSOR_OBJ_DECLARE(
  gemm_407_output, AI_STATIC,
  439, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_407_output_array, &gemm_407_output_array_intq)

/* Tensor #440 */
AI_TENSOR_OBJ_DECLARE(
  gemm_407_scratch0, AI_STATIC,
  440, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_407_scratch0_array, NULL)

/* Tensor #441 */
AI_TENSOR_OBJ_DECLARE(
  gemm_410_output, AI_STATIC,
  441, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_410_output_array, &gemm_410_output_array_intq)

/* Tensor #442 */
AI_TENSOR_OBJ_DECLARE(
  gemm_410_scratch0, AI_STATIC,
  442, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_410_scratch0_array, NULL)

/* Tensor #443 */
AI_TENSOR_OBJ_DECLARE(
  gemm_413_output, AI_STATIC,
  443, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_413_output_array, &gemm_413_output_array_intq)

/* Tensor #444 */
AI_TENSOR_OBJ_DECLARE(
  gemm_413_scratch0, AI_STATIC,
  444, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_413_scratch0_array, NULL)

/* Tensor #445 */
AI_TENSOR_OBJ_DECLARE(
  gemm_416_output, AI_STATIC,
  445, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_416_output_array, &gemm_416_output_array_intq)

/* Tensor #446 */
AI_TENSOR_OBJ_DECLARE(
  gemm_416_scratch0, AI_STATIC,
  446, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_416_scratch0_array, NULL)

/* Tensor #447 */
AI_TENSOR_OBJ_DECLARE(
  gemm_419_output, AI_STATIC,
  447, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_419_output_array, &gemm_419_output_array_intq)

/* Tensor #448 */
AI_TENSOR_OBJ_DECLARE(
  gemm_419_scratch0, AI_STATIC,
  448, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_419_scratch0_array, NULL)

/* Tensor #449 */
AI_TENSOR_OBJ_DECLARE(
  gemm_422_output, AI_STATIC,
  449, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_422_output_array, &gemm_422_output_array_intq)

/* Tensor #450 */
AI_TENSOR_OBJ_DECLARE(
  gemm_422_scratch0, AI_STATIC,
  450, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_422_scratch0_array, NULL)

/* Tensor #451 */
AI_TENSOR_OBJ_DECLARE(
  gemm_425_output, AI_STATIC,
  451, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_425_output_array, &gemm_425_output_array_intq)

/* Tensor #452 */
AI_TENSOR_OBJ_DECLARE(
  gemm_425_scratch0, AI_STATIC,
  452, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_425_scratch0_array, NULL)

/* Tensor #453 */
AI_TENSOR_OBJ_DECLARE(
  gemm_428_output, AI_STATIC,
  453, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_428_output_array, &gemm_428_output_array_intq)

/* Tensor #454 */
AI_TENSOR_OBJ_DECLARE(
  gemm_428_scratch0, AI_STATIC,
  454, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_428_scratch0_array, NULL)

/* Tensor #455 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_output, AI_STATIC,
  455, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_42_output_array, &gemm_42_output_array_intq)

/* Tensor #456 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_scratch0, AI_STATIC,
  456, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_42_scratch0_array, NULL)

/* Tensor #457 */
AI_TENSOR_OBJ_DECLARE(
  gemm_431_output, AI_STATIC,
  457, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_431_output_array, &gemm_431_output_array_intq)

/* Tensor #458 */
AI_TENSOR_OBJ_DECLARE(
  gemm_431_scratch0, AI_STATIC,
  458, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_431_scratch0_array, NULL)

/* Tensor #459 */
AI_TENSOR_OBJ_DECLARE(
  gemm_434_output, AI_STATIC,
  459, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_434_output_array, &gemm_434_output_array_intq)

/* Tensor #460 */
AI_TENSOR_OBJ_DECLARE(
  gemm_434_scratch0, AI_STATIC,
  460, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_434_scratch0_array, NULL)

/* Tensor #461 */
AI_TENSOR_OBJ_DECLARE(
  gemm_435_output, AI_STATIC,
  461, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_435_output_array, &gemm_435_output_array_intq)

/* Tensor #462 */
AI_TENSOR_OBJ_DECLARE(
  gemm_435_scratch0, AI_STATIC,
  462, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_435_scratch0_array, NULL)

/* Tensor #463 */
AI_TENSOR_OBJ_DECLARE(
  gemm_438_output, AI_STATIC,
  463, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_438_output_array, &gemm_438_output_array_intq)

/* Tensor #464 */
AI_TENSOR_OBJ_DECLARE(
  gemm_438_scratch0, AI_STATIC,
  464, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_438_scratch0_array, NULL)

/* Tensor #465 */
AI_TENSOR_OBJ_DECLARE(
  gemm_439_output, AI_STATIC,
  465, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_439_output_array, &gemm_439_output_array_intq)

/* Tensor #466 */
AI_TENSOR_OBJ_DECLARE(
  gemm_439_scratch0, AI_STATIC,
  466, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_439_scratch0_array, NULL)

/* Tensor #467 */
AI_TENSOR_OBJ_DECLARE(
  gemm_43_output, AI_STATIC,
  467, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_43_output_array, &gemm_43_output_array_intq)

/* Tensor #468 */
AI_TENSOR_OBJ_DECLARE(
  gemm_43_scratch0, AI_STATIC,
  468, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_43_scratch0_array, NULL)

/* Tensor #469 */
AI_TENSOR_OBJ_DECLARE(
  gemm_442_output, AI_STATIC,
  469, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_442_output_array, &gemm_442_output_array_intq)

/* Tensor #470 */
AI_TENSOR_OBJ_DECLARE(
  gemm_442_scratch0, AI_STATIC,
  470, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &gemm_442_scratch0_array, NULL)

/* Tensor #471 */
AI_TENSOR_OBJ_DECLARE(
  gemm_443_output, AI_STATIC,
  471, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_443_output_array, &gemm_443_output_array_intq)

/* Tensor #472 */
AI_TENSOR_OBJ_DECLARE(
  gemm_443_scratch0, AI_STATIC,
  472, 0x0,
  AI_SHAPE_INIT(4, 1, 104, 1, 1), AI_STRIDE_INIT(4, 2, 2, 208, 208),
  1, &gemm_443_scratch0_array, NULL)

/* Tensor #473 */
AI_TENSOR_OBJ_DECLARE(
  gemm_446_bias, AI_STATIC,
  473, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &gemm_446_bias_array, NULL)

/* Tensor #474 */
AI_TENSOR_OBJ_DECLARE(
  gemm_446_output, AI_STATIC,
  474, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_446_output_array, &gemm_446_output_array_intq)

/* Tensor #475 */
AI_TENSOR_OBJ_DECLARE(
  gemm_446_scratch0, AI_STATIC,
  475, 0x0,
  AI_SHAPE_INIT(4, 1, 136, 1, 1), AI_STRIDE_INIT(4, 2, 2, 272, 272),
  1, &gemm_446_scratch0_array, NULL)

/* Tensor #476 */
AI_TENSOR_OBJ_DECLARE(
  gemm_446_weights, AI_STATIC,
  476, 0x1,
  AI_SHAPE_INIT(4, 16, 24, 1, 1), AI_STRIDE_INIT(4, 1, 16, 384, 384),
  1, &gemm_446_weights_array, &gemm_446_weights_array_intq)

/* Tensor #477 */
AI_TENSOR_OBJ_DECLARE(
  gemm_447_bias, AI_STATIC,
  477, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &gemm_447_bias_array, NULL)

/* Tensor #478 */
AI_TENSOR_OBJ_DECLARE(
  gemm_447_output, AI_STATIC,
  478, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_447_output_array, &gemm_447_output_array_intq)

/* Tensor #479 */
AI_TENSOR_OBJ_DECLARE(
  gemm_447_scratch0, AI_STATIC,
  479, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_447_scratch0_array, NULL)

/* Tensor #480 */
AI_TENSOR_OBJ_DECLARE(
  gemm_447_weights, AI_STATIC,
  480, 0x1,
  AI_SHAPE_INIT(4, 24, 24, 1, 1), AI_STRIDE_INIT(4, 1, 24, 576, 576),
  1, &gemm_447_weights_array, &gemm_447_weights_array_intq)

/* Tensor #481 */
AI_TENSOR_OBJ_DECLARE(
  gemm_44_output, AI_STATIC,
  481, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_44_output_array, &gemm_44_output_array_intq)

/* Tensor #482 */
AI_TENSOR_OBJ_DECLARE(
  gemm_44_scratch0, AI_STATIC,
  482, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_44_scratch0_array, NULL)

/* Tensor #483 */
AI_TENSOR_OBJ_DECLARE(
  gemm_45_output, AI_STATIC,
  483, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_45_output_array, &gemm_45_output_array_intq)

/* Tensor #484 */
AI_TENSOR_OBJ_DECLARE(
  gemm_45_scratch0, AI_STATIC,
  484, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_45_scratch0_array, NULL)

/* Tensor #485 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_output, AI_STATIC,
  485, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_46_output_array, &gemm_46_output_array_intq)

/* Tensor #486 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_scratch0, AI_STATIC,
  486, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_46_scratch0_array, NULL)

/* Tensor #487 */
AI_TENSOR_OBJ_DECLARE(
  gemm_47_output, AI_STATIC,
  487, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_47_output_array, &gemm_47_output_array_intq)

/* Tensor #488 */
AI_TENSOR_OBJ_DECLARE(
  gemm_47_scratch0, AI_STATIC,
  488, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_47_scratch0_array, NULL)

/* Tensor #489 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_output, AI_STATIC,
  489, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_48_output_array, &gemm_48_output_array_intq)

/* Tensor #490 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_scratch0, AI_STATIC,
  490, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_48_scratch0_array, NULL)

/* Tensor #491 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_output, AI_STATIC,
  491, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_49_output_array, &gemm_49_output_array_intq)

/* Tensor #492 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_scratch0, AI_STATIC,
  492, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_49_scratch0_array, NULL)

/* Tensor #493 */
AI_TENSOR_OBJ_DECLARE(
  gemm_50_output, AI_STATIC,
  493, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_50_output_array, &gemm_50_output_array_intq)

/* Tensor #494 */
AI_TENSOR_OBJ_DECLARE(
  gemm_50_scratch0, AI_STATIC,
  494, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_50_scratch0_array, NULL)

/* Tensor #495 */
AI_TENSOR_OBJ_DECLARE(
  gemm_51_output, AI_STATIC,
  495, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_51_output_array, &gemm_51_output_array_intq)

/* Tensor #496 */
AI_TENSOR_OBJ_DECLARE(
  gemm_51_scratch0, AI_STATIC,
  496, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_51_scratch0_array, NULL)

/* Tensor #497 */
AI_TENSOR_OBJ_DECLARE(
  gemm_52_output, AI_STATIC,
  497, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_52_output_array, &gemm_52_output_array_intq)

/* Tensor #498 */
AI_TENSOR_OBJ_DECLARE(
  gemm_52_scratch0, AI_STATIC,
  498, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_52_scratch0_array, NULL)

/* Tensor #499 */
AI_TENSOR_OBJ_DECLARE(
  gemm_53_output, AI_STATIC,
  499, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_53_output_array, &gemm_53_output_array_intq)

/* Tensor #500 */
AI_TENSOR_OBJ_DECLARE(
  gemm_53_scratch0, AI_STATIC,
  500, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_53_scratch0_array, NULL)

/* Tensor #501 */
AI_TENSOR_OBJ_DECLARE(
  gemm_57_output, AI_STATIC,
  501, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_57_output_array, &gemm_57_output_array_intq)

/* Tensor #502 */
AI_TENSOR_OBJ_DECLARE(
  gemm_57_scratch0, AI_STATIC,
  502, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_57_scratch0_array, NULL)

/* Tensor #503 */
AI_TENSOR_OBJ_DECLARE(
  gemm_58_output, AI_STATIC,
  503, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_58_output_array, &gemm_58_output_array_intq)

/* Tensor #504 */
AI_TENSOR_OBJ_DECLARE(
  gemm_58_scratch0, AI_STATIC,
  504, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_58_scratch0_array, NULL)

/* Tensor #505 */
AI_TENSOR_OBJ_DECLARE(
  gemm_59_output, AI_STATIC,
  505, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_59_output_array, &gemm_59_output_array_intq)

/* Tensor #506 */
AI_TENSOR_OBJ_DECLARE(
  gemm_59_scratch0, AI_STATIC,
  506, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_59_scratch0_array, NULL)

/* Tensor #507 */
AI_TENSOR_OBJ_DECLARE(
  gemm_5_bias, AI_STATIC,
  507, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &gemm_5_bias_array, NULL)

/* Tensor #508 */
AI_TENSOR_OBJ_DECLARE(
  gemm_5_output, AI_STATIC,
  508, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_5_output_array, &gemm_5_output_array_intq)

/* Tensor #509 */
AI_TENSOR_OBJ_DECLARE(
  gemm_5_scratch0, AI_STATIC,
  509, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_5_scratch0_array, NULL)

/* Tensor #510 */
AI_TENSOR_OBJ_DECLARE(
  gemm_5_weights, AI_STATIC,
  510, 0x1,
  AI_SHAPE_INIT(4, 24, 24, 1, 1), AI_STRIDE_INIT(4, 1, 24, 576, 576),
  1, &gemm_5_weights_array, &gemm_5_weights_array_intq)

/* Tensor #511 */
AI_TENSOR_OBJ_DECLARE(
  gemm_60_output, AI_STATIC,
  511, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_60_output_array, &gemm_60_output_array_intq)

/* Tensor #512 */
AI_TENSOR_OBJ_DECLARE(
  gemm_60_scratch0, AI_STATIC,
  512, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_60_scratch0_array, NULL)

/* Tensor #513 */
AI_TENSOR_OBJ_DECLARE(
  gemm_61_output, AI_STATIC,
  513, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_61_output_array, &gemm_61_output_array_intq)

/* Tensor #514 */
AI_TENSOR_OBJ_DECLARE(
  gemm_61_scratch0, AI_STATIC,
  514, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_61_scratch0_array, NULL)

/* Tensor #515 */
AI_TENSOR_OBJ_DECLARE(
  gemm_62_output, AI_STATIC,
  515, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_62_output_array, &gemm_62_output_array_intq)

/* Tensor #516 */
AI_TENSOR_OBJ_DECLARE(
  gemm_62_scratch0, AI_STATIC,
  516, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_62_scratch0_array, NULL)

/* Tensor #517 */
AI_TENSOR_OBJ_DECLARE(
  gemm_63_output, AI_STATIC,
  517, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_63_output_array, &gemm_63_output_array_intq)

/* Tensor #518 */
AI_TENSOR_OBJ_DECLARE(
  gemm_63_scratch0, AI_STATIC,
  518, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_63_scratch0_array, NULL)

/* Tensor #519 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_output, AI_STATIC,
  519, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_64_output_array, &gemm_64_output_array_intq)

/* Tensor #520 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_scratch0, AI_STATIC,
  520, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_64_scratch0_array, NULL)

/* Tensor #521 */
AI_TENSOR_OBJ_DECLARE(
  gemm_65_output, AI_STATIC,
  521, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_65_output_array, &gemm_65_output_array_intq)

/* Tensor #522 */
AI_TENSOR_OBJ_DECLARE(
  gemm_65_scratch0, AI_STATIC,
  522, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_65_scratch0_array, NULL)

/* Tensor #523 */
AI_TENSOR_OBJ_DECLARE(
  gemm_66_output, AI_STATIC,
  523, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_66_output_array, &gemm_66_output_array_intq)

/* Tensor #524 */
AI_TENSOR_OBJ_DECLARE(
  gemm_66_scratch0, AI_STATIC,
  524, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_66_scratch0_array, NULL)

/* Tensor #525 */
AI_TENSOR_OBJ_DECLARE(
  gemm_67_output, AI_STATIC,
  525, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_67_output_array, &gemm_67_output_array_intq)

/* Tensor #526 */
AI_TENSOR_OBJ_DECLARE(
  gemm_67_scratch0, AI_STATIC,
  526, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_67_scratch0_array, NULL)

/* Tensor #527 */
AI_TENSOR_OBJ_DECLARE(
  gemm_68_output, AI_STATIC,
  527, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_68_output_array, &gemm_68_output_array_intq)

/* Tensor #528 */
AI_TENSOR_OBJ_DECLARE(
  gemm_68_scratch0, AI_STATIC,
  528, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_68_scratch0_array, NULL)

/* Tensor #529 */
AI_TENSOR_OBJ_DECLARE(
  gemm_72_output, AI_STATIC,
  529, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_72_output_array, &gemm_72_output_array_intq)

/* Tensor #530 */
AI_TENSOR_OBJ_DECLARE(
  gemm_72_scratch0, AI_STATIC,
  530, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_72_scratch0_array, NULL)

/* Tensor #531 */
AI_TENSOR_OBJ_DECLARE(
  gemm_76_output, AI_STATIC,
  531, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_76_output_array, &gemm_76_output_array_intq)

/* Tensor #532 */
AI_TENSOR_OBJ_DECLARE(
  gemm_76_scratch0, AI_STATIC,
  532, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_76_scratch0_array, NULL)

/* Tensor #533 */
AI_TENSOR_OBJ_DECLARE(
  gemm_80_output, AI_STATIC,
  533, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_80_output_array, &gemm_80_output_array_intq)

/* Tensor #534 */
AI_TENSOR_OBJ_DECLARE(
  gemm_80_scratch0, AI_STATIC,
  534, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_80_scratch0_array, NULL)

/* Tensor #535 */
AI_TENSOR_OBJ_DECLARE(
  gemm_84_output, AI_STATIC,
  535, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_84_output_array, &gemm_84_output_array_intq)

/* Tensor #536 */
AI_TENSOR_OBJ_DECLARE(
  gemm_84_scratch0, AI_STATIC,
  536, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_84_scratch0_array, NULL)

/* Tensor #537 */
AI_TENSOR_OBJ_DECLARE(
  gemm_88_output, AI_STATIC,
  537, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_88_output_array, &gemm_88_output_array_intq)

/* Tensor #538 */
AI_TENSOR_OBJ_DECLARE(
  gemm_88_scratch0, AI_STATIC,
  538, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_88_scratch0_array, NULL)

/* Tensor #539 */
AI_TENSOR_OBJ_DECLARE(
  gemm_8_bias, AI_STATIC,
  539, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &gemm_8_bias_array, NULL)

/* Tensor #540 */
AI_TENSOR_OBJ_DECLARE(
  gemm_8_output, AI_STATIC,
  540, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_8_output_array, &gemm_8_output_array_intq)

/* Tensor #541 */
AI_TENSOR_OBJ_DECLARE(
  gemm_8_scratch0, AI_STATIC,
  541, 0x0,
  AI_SHAPE_INIT(4, 1, 129, 1, 1), AI_STRIDE_INIT(4, 2, 2, 258, 258),
  1, &gemm_8_scratch0_array, NULL)

/* Tensor #542 */
AI_TENSOR_OBJ_DECLARE(
  gemm_8_weights, AI_STATIC,
  542, 0x1,
  AI_SHAPE_INIT(4, 9, 24, 1, 1), AI_STRIDE_INIT(4, 1, 9, 216, 216),
  1, &gemm_8_weights_array, &gemm_8_weights_array_intq)

/* Tensor #543 */
AI_TENSOR_OBJ_DECLARE(
  gemm_92_output, AI_STATIC,
  543, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_92_output_array, &gemm_92_output_array_intq)

/* Tensor #544 */
AI_TENSOR_OBJ_DECLARE(
  gemm_92_scratch0, AI_STATIC,
  544, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_92_scratch0_array, NULL)

/* Tensor #545 */
AI_TENSOR_OBJ_DECLARE(
  gemm_96_output, AI_STATIC,
  545, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_96_output_array, &gemm_96_output_array_intq)

/* Tensor #546 */
AI_TENSOR_OBJ_DECLARE(
  gemm_96_scratch0, AI_STATIC,
  546, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_96_scratch0_array, NULL)

/* Tensor #547 */
AI_TENSOR_OBJ_DECLARE(
  nl_102_output, AI_STATIC,
  547, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_102_output_array, &nl_102_output_array_intq)

/* Tensor #548 */
AI_TENSOR_OBJ_DECLARE(
  nl_106_output, AI_STATIC,
  548, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_106_output_array, &nl_106_output_array_intq)

/* Tensor #549 */
AI_TENSOR_OBJ_DECLARE(
  nl_10_output, AI_STATIC,
  549, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_10_output_array, &nl_10_output_array_intq)

/* Tensor #550 */
AI_TENSOR_OBJ_DECLARE(
  nl_110_output, AI_STATIC,
  550, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_110_output_array, &nl_110_output_array_intq)

/* Tensor #551 */
AI_TENSOR_OBJ_DECLARE(
  nl_114_output, AI_STATIC,
  551, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_114_output_array, &nl_114_output_array_intq)

/* Tensor #552 */
AI_TENSOR_OBJ_DECLARE(
  nl_118_output, AI_STATIC,
  552, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_118_output_array, &nl_118_output_array_intq)

/* Tensor #553 */
AI_TENSOR_OBJ_DECLARE(
  nl_122_output, AI_STATIC,
  553, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_122_output_array, &nl_122_output_array_intq)

/* Tensor #554 */
AI_TENSOR_OBJ_DECLARE(
  nl_126_output, AI_STATIC,
  554, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_126_output_array, &nl_126_output_array_intq)

/* Tensor #555 */
AI_TENSOR_OBJ_DECLARE(
  nl_130_output, AI_STATIC,
  555, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_130_output_array, &nl_130_output_array_intq)

/* Tensor #556 */
AI_TENSOR_OBJ_DECLARE(
  nl_134_output, AI_STATIC,
  556, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_134_output_array, &nl_134_output_array_intq)

/* Tensor #557 */
AI_TENSOR_OBJ_DECLARE(
  nl_138_output, AI_STATIC,
  557, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_138_output_array, &nl_138_output_array_intq)

/* Tensor #558 */
AI_TENSOR_OBJ_DECLARE(
  nl_142_output, AI_STATIC,
  558, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_142_output_array, &nl_142_output_array_intq)

/* Tensor #559 */
AI_TENSOR_OBJ_DECLARE(
  nl_146_output, AI_STATIC,
  559, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_146_output_array, &nl_146_output_array_intq)

/* Tensor #560 */
AI_TENSOR_OBJ_DECLARE(
  nl_150_output, AI_STATIC,
  560, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_150_output_array, &nl_150_output_array_intq)

/* Tensor #561 */
AI_TENSOR_OBJ_DECLARE(
  nl_154_output, AI_STATIC,
  561, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_154_output_array, &nl_154_output_array_intq)

/* Tensor #562 */
AI_TENSOR_OBJ_DECLARE(
  nl_158_output, AI_STATIC,
  562, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_158_output_array, &nl_158_output_array_intq)

/* Tensor #563 */
AI_TENSOR_OBJ_DECLARE(
  nl_162_output, AI_STATIC,
  563, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_162_output_array, &nl_162_output_array_intq)

/* Tensor #564 */
AI_TENSOR_OBJ_DECLARE(
  nl_166_output, AI_STATIC,
  564, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_166_output_array, &nl_166_output_array_intq)

/* Tensor #565 */
AI_TENSOR_OBJ_DECLARE(
  nl_170_output, AI_STATIC,
  565, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_170_output_array, &nl_170_output_array_intq)

/* Tensor #566 */
AI_TENSOR_OBJ_DECLARE(
  nl_174_output, AI_STATIC,
  566, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_174_output_array, &nl_174_output_array_intq)

/* Tensor #567 */
AI_TENSOR_OBJ_DECLARE(
  nl_178_output, AI_STATIC,
  567, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_178_output_array, &nl_178_output_array_intq)

/* Tensor #568 */
AI_TENSOR_OBJ_DECLARE(
  nl_182_output, AI_STATIC,
  568, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_182_output_array, &nl_182_output_array_intq)

/* Tensor #569 */
AI_TENSOR_OBJ_DECLARE(
  nl_186_output, AI_STATIC,
  569, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_186_output_array, &nl_186_output_array_intq)

/* Tensor #570 */
AI_TENSOR_OBJ_DECLARE(
  nl_190_output, AI_STATIC,
  570, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_190_output_array, &nl_190_output_array_intq)

/* Tensor #571 */
AI_TENSOR_OBJ_DECLARE(
  nl_194_output, AI_STATIC,
  571, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_194_output_array, &nl_194_output_array_intq)

/* Tensor #572 */
AI_TENSOR_OBJ_DECLARE(
  nl_198_output, AI_STATIC,
  572, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_198_output_array, &nl_198_output_array_intq)

/* Tensor #573 */
AI_TENSOR_OBJ_DECLARE(
  nl_202_output, AI_STATIC,
  573, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_202_output_array, &nl_202_output_array_intq)

/* Tensor #574 */
AI_TENSOR_OBJ_DECLARE(
  nl_206_output, AI_STATIC,
  574, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_206_output_array, &nl_206_output_array_intq)

/* Tensor #575 */
AI_TENSOR_OBJ_DECLARE(
  nl_210_output, AI_STATIC,
  575, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_210_output_array, &nl_210_output_array_intq)

/* Tensor #576 */
AI_TENSOR_OBJ_DECLARE(
  nl_214_output, AI_STATIC,
  576, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_214_output_array, &nl_214_output_array_intq)

/* Tensor #577 */
AI_TENSOR_OBJ_DECLARE(
  nl_218_output, AI_STATIC,
  577, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_218_output_array, &nl_218_output_array_intq)

/* Tensor #578 */
AI_TENSOR_OBJ_DECLARE(
  nl_222_output, AI_STATIC,
  578, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_222_output_array, &nl_222_output_array_intq)

/* Tensor #579 */
AI_TENSOR_OBJ_DECLARE(
  nl_226_output, AI_STATIC,
  579, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_226_output_array, &nl_226_output_array_intq)

/* Tensor #580 */
AI_TENSOR_OBJ_DECLARE(
  nl_230_output, AI_STATIC,
  580, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_230_output_array, &nl_230_output_array_intq)

/* Tensor #581 */
AI_TENSOR_OBJ_DECLARE(
  nl_235_output, AI_STATIC,
  581, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_235_output_array, &nl_235_output_array_intq)

/* Tensor #582 */
AI_TENSOR_OBJ_DECLARE(
  nl_240_output, AI_STATIC,
  582, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_240_output_array, &nl_240_output_array_intq)

/* Tensor #583 */
AI_TENSOR_OBJ_DECLARE(
  nl_245_output, AI_STATIC,
  583, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_245_output_array, &nl_245_output_array_intq)

/* Tensor #584 */
AI_TENSOR_OBJ_DECLARE(
  nl_257_output, AI_STATIC,
  584, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_257_output_array, &nl_257_output_array_intq)

/* Tensor #585 */
AI_TENSOR_OBJ_DECLARE(
  nl_25_output, AI_STATIC,
  585, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_25_output_array, &nl_25_output_array_intq)

/* Tensor #586 */
AI_TENSOR_OBJ_DECLARE(
  nl_271_output, AI_STATIC,
  586, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_271_output_array, &nl_271_output_array_intq)

/* Tensor #587 */
AI_TENSOR_OBJ_DECLARE(
  nl_285_output, AI_STATIC,
  587, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_285_output_array, &nl_285_output_array_intq)

/* Tensor #588 */
AI_TENSOR_OBJ_DECLARE(
  nl_299_output, AI_STATIC,
  588, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_299_output_array, &nl_299_output_array_intq)

/* Tensor #589 */
AI_TENSOR_OBJ_DECLARE(
  nl_313_output, AI_STATIC,
  589, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_313_output_array, &nl_313_output_array_intq)

/* Tensor #590 */
AI_TENSOR_OBJ_DECLARE(
  nl_316_output, AI_STATIC,
  590, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_316_output_array, &nl_316_output_array_intq)

/* Tensor #591 */
AI_TENSOR_OBJ_DECLARE(
  nl_319_output, AI_STATIC,
  591, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_319_output_array, &nl_319_output_array_intq)

/* Tensor #592 */
AI_TENSOR_OBJ_DECLARE(
  nl_322_output, AI_STATIC,
  592, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_322_output_array, &nl_322_output_array_intq)

/* Tensor #593 */
AI_TENSOR_OBJ_DECLARE(
  nl_325_output, AI_STATIC,
  593, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_325_output_array, &nl_325_output_array_intq)

/* Tensor #594 */
AI_TENSOR_OBJ_DECLARE(
  nl_328_output, AI_STATIC,
  594, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_328_output_array, &nl_328_output_array_intq)

/* Tensor #595 */
AI_TENSOR_OBJ_DECLARE(
  nl_331_output, AI_STATIC,
  595, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_331_output_array, &nl_331_output_array_intq)

/* Tensor #596 */
AI_TENSOR_OBJ_DECLARE(
  nl_334_output, AI_STATIC,
  596, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_334_output_array, &nl_334_output_array_intq)

/* Tensor #597 */
AI_TENSOR_OBJ_DECLARE(
  nl_337_output, AI_STATIC,
  597, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_337_output_array, &nl_337_output_array_intq)

/* Tensor #598 */
AI_TENSOR_OBJ_DECLARE(
  nl_340_output, AI_STATIC,
  598, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_340_output_array, &nl_340_output_array_intq)

/* Tensor #599 */
AI_TENSOR_OBJ_DECLARE(
  nl_343_output, AI_STATIC,
  599, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_343_output_array, &nl_343_output_array_intq)

/* Tensor #600 */
AI_TENSOR_OBJ_DECLARE(
  nl_346_output, AI_STATIC,
  600, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_346_output_array, &nl_346_output_array_intq)

/* Tensor #601 */
AI_TENSOR_OBJ_DECLARE(
  nl_349_output, AI_STATIC,
  601, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_349_output_array, &nl_349_output_array_intq)

/* Tensor #602 */
AI_TENSOR_OBJ_DECLARE(
  nl_352_output, AI_STATIC,
  602, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_352_output_array, &nl_352_output_array_intq)

/* Tensor #603 */
AI_TENSOR_OBJ_DECLARE(
  nl_355_output, AI_STATIC,
  603, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_355_output_array, &nl_355_output_array_intq)

/* Tensor #604 */
AI_TENSOR_OBJ_DECLARE(
  nl_358_output, AI_STATIC,
  604, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_358_output_array, &nl_358_output_array_intq)

/* Tensor #605 */
AI_TENSOR_OBJ_DECLARE(
  nl_361_output, AI_STATIC,
  605, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_361_output_array, &nl_361_output_array_intq)

/* Tensor #606 */
AI_TENSOR_OBJ_DECLARE(
  nl_364_output, AI_STATIC,
  606, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_364_output_array, &nl_364_output_array_intq)

/* Tensor #607 */
AI_TENSOR_OBJ_DECLARE(
  nl_367_output, AI_STATIC,
  607, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_367_output_array, &nl_367_output_array_intq)

/* Tensor #608 */
AI_TENSOR_OBJ_DECLARE(
  nl_370_output, AI_STATIC,
  608, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_370_output_array, &nl_370_output_array_intq)

/* Tensor #609 */
AI_TENSOR_OBJ_DECLARE(
  nl_373_output, AI_STATIC,
  609, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_373_output_array, &nl_373_output_array_intq)

/* Tensor #610 */
AI_TENSOR_OBJ_DECLARE(
  nl_376_output, AI_STATIC,
  610, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_376_output_array, &nl_376_output_array_intq)

/* Tensor #611 */
AI_TENSOR_OBJ_DECLARE(
  nl_379_output, AI_STATIC,
  611, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_379_output_array, &nl_379_output_array_intq)

/* Tensor #612 */
AI_TENSOR_OBJ_DECLARE(
  nl_382_output, AI_STATIC,
  612, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_382_output_array, &nl_382_output_array_intq)

/* Tensor #613 */
AI_TENSOR_OBJ_DECLARE(
  nl_385_output, AI_STATIC,
  613, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_385_output_array, &nl_385_output_array_intq)

/* Tensor #614 */
AI_TENSOR_OBJ_DECLARE(
  nl_388_output, AI_STATIC,
  614, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_388_output_array, &nl_388_output_array_intq)

/* Tensor #615 */
AI_TENSOR_OBJ_DECLARE(
  nl_391_output, AI_STATIC,
  615, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_391_output_array, &nl_391_output_array_intq)

/* Tensor #616 */
AI_TENSOR_OBJ_DECLARE(
  nl_394_output, AI_STATIC,
  616, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_394_output_array, &nl_394_output_array_intq)

/* Tensor #617 */
AI_TENSOR_OBJ_DECLARE(
  nl_397_output, AI_STATIC,
  617, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_397_output_array, &nl_397_output_array_intq)

/* Tensor #618 */
AI_TENSOR_OBJ_DECLARE(
  nl_400_output, AI_STATIC,
  618, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_400_output_array, &nl_400_output_array_intq)

/* Tensor #619 */
AI_TENSOR_OBJ_DECLARE(
  nl_403_output, AI_STATIC,
  619, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_403_output_array, &nl_403_output_array_intq)

/* Tensor #620 */
AI_TENSOR_OBJ_DECLARE(
  nl_406_output, AI_STATIC,
  620, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_406_output_array, &nl_406_output_array_intq)

/* Tensor #621 */
AI_TENSOR_OBJ_DECLARE(
  nl_409_output, AI_STATIC,
  621, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_409_output_array, &nl_409_output_array_intq)

/* Tensor #622 */
AI_TENSOR_OBJ_DECLARE(
  nl_40_output, AI_STATIC,
  622, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_40_output_array, &nl_40_output_array_intq)

/* Tensor #623 */
AI_TENSOR_OBJ_DECLARE(
  nl_412_output, AI_STATIC,
  623, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_412_output_array, &nl_412_output_array_intq)

/* Tensor #624 */
AI_TENSOR_OBJ_DECLARE(
  nl_415_output, AI_STATIC,
  624, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_415_output_array, &nl_415_output_array_intq)

/* Tensor #625 */
AI_TENSOR_OBJ_DECLARE(
  nl_418_output, AI_STATIC,
  625, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_418_output_array, &nl_418_output_array_intq)

/* Tensor #626 */
AI_TENSOR_OBJ_DECLARE(
  nl_421_output, AI_STATIC,
  626, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_421_output_array, &nl_421_output_array_intq)

/* Tensor #627 */
AI_TENSOR_OBJ_DECLARE(
  nl_424_output, AI_STATIC,
  627, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_424_output_array, &nl_424_output_array_intq)

/* Tensor #628 */
AI_TENSOR_OBJ_DECLARE(
  nl_427_output, AI_STATIC,
  628, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_427_output_array, &nl_427_output_array_intq)

/* Tensor #629 */
AI_TENSOR_OBJ_DECLARE(
  nl_430_output, AI_STATIC,
  629, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_430_output_array, &nl_430_output_array_intq)

/* Tensor #630 */
AI_TENSOR_OBJ_DECLARE(
  nl_433_output, AI_STATIC,
  630, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_433_output_array, &nl_433_output_array_intq)

/* Tensor #631 */
AI_TENSOR_OBJ_DECLARE(
  nl_437_output, AI_STATIC,
  631, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_437_output_array, &nl_437_output_array_intq)

/* Tensor #632 */
AI_TENSOR_OBJ_DECLARE(
  nl_441_output, AI_STATIC,
  632, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_441_output_array, &nl_441_output_array_intq)

/* Tensor #633 */
AI_TENSOR_OBJ_DECLARE(
  nl_445_output, AI_STATIC,
  633, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &nl_445_output_array, &nl_445_output_array_intq)

/* Tensor #634 */
AI_TENSOR_OBJ_DECLARE(
  nl_55_output, AI_STATIC,
  634, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_55_output_array, &nl_55_output_array_intq)

/* Tensor #635 */
AI_TENSOR_OBJ_DECLARE(
  nl_70_output, AI_STATIC,
  635, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_70_output_array, &nl_70_output_array_intq)

/* Tensor #636 */
AI_TENSOR_OBJ_DECLARE(
  nl_74_output, AI_STATIC,
  636, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_74_output_array, &nl_74_output_array_intq)

/* Tensor #637 */
AI_TENSOR_OBJ_DECLARE(
  nl_78_output, AI_STATIC,
  637, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_78_output_array, &nl_78_output_array_intq)

/* Tensor #638 */
AI_TENSOR_OBJ_DECLARE(
  nl_82_output, AI_STATIC,
  638, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_82_output_array, &nl_82_output_array_intq)

/* Tensor #639 */
AI_TENSOR_OBJ_DECLARE(
  nl_86_output, AI_STATIC,
  639, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_86_output_array, &nl_86_output_array_intq)

/* Tensor #640 */
AI_TENSOR_OBJ_DECLARE(
  nl_90_output, AI_STATIC,
  640, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_90_output_array, &nl_90_output_array_intq)

/* Tensor #641 */
AI_TENSOR_OBJ_DECLARE(
  nl_94_output, AI_STATIC,
  641, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_94_output_array, &nl_94_output_array_intq)

/* Tensor #642 */
AI_TENSOR_OBJ_DECLARE(
  nl_98_output, AI_STATIC,
  642, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &nl_98_output_array, &nl_98_output_array_intq)

/* Tensor #643 */
AI_TENSOR_OBJ_DECLARE(
  pack_247_output, AI_STATIC,
  643, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 48), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &pack_247_output_array, &pack_247_output_array_intq)

/* Tensor #644 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_sensor_history0_output, AI_STATIC,
  644, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 48), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &serving_default_sensor_history0_output_array, &serving_default_sensor_history0_output_array_intq)

/* Tensor #645 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_sensor_history0_output0, AI_STATIC,
  645, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 48, 1), AI_STRIDE_INIT(4, 1, 1, 9, 432),
  1, &serving_default_sensor_history0_output_array, &serving_default_sensor_history0_output_array_intq)

/* Tensor #646 */
AI_TENSOR_OBJ_DECLARE(
  slice_0_output, AI_STATIC,
  646, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 24), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &slice_0_output_array, &slice_0_output_array_intq)

/* Tensor #647 */
AI_TENSOR_OBJ_DECLARE(
  slice_0_output0, AI_STATIC,
  647, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &slice_0_output_array, &slice_0_output_array_intq)

/* Tensor #648 */
AI_TENSOR_OBJ_DECLARE(
  transpose_6_output, AI_STATIC,
  648, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 48), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &transpose_6_output_array, &transpose_6_output_array_intq)

/* Tensor #649 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output0, AI_STATIC,
  649, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output0_array, &unpack_254_output0_array_intq)

/* Tensor #650 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output1, AI_STATIC,
  650, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output1_array, &unpack_254_output1_array_intq)

/* Tensor #651 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output10, AI_STATIC,
  651, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output10_array, &unpack_254_output10_array_intq)

/* Tensor #652 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output11, AI_STATIC,
  652, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output11_array, &unpack_254_output11_array_intq)

/* Tensor #653 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output12, AI_STATIC,
  653, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output12_array, &unpack_254_output12_array_intq)

/* Tensor #654 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output13, AI_STATIC,
  654, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output13_array, &unpack_254_output13_array_intq)

/* Tensor #655 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output14, AI_STATIC,
  655, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output14_array, &unpack_254_output14_array_intq)

/* Tensor #656 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output15, AI_STATIC,
  656, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output15_array, &unpack_254_output15_array_intq)

/* Tensor #657 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output16, AI_STATIC,
  657, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output16_array, &unpack_254_output16_array_intq)

/* Tensor #658 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output17, AI_STATIC,
  658, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output17_array, &unpack_254_output17_array_intq)

/* Tensor #659 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output18, AI_STATIC,
  659, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output18_array, &unpack_254_output18_array_intq)

/* Tensor #660 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output19, AI_STATIC,
  660, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output19_array, &unpack_254_output19_array_intq)

/* Tensor #661 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output2, AI_STATIC,
  661, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output2_array, &unpack_254_output2_array_intq)

/* Tensor #662 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output20, AI_STATIC,
  662, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output20_array, &unpack_254_output20_array_intq)

/* Tensor #663 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output21, AI_STATIC,
  663, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output21_array, &unpack_254_output21_array_intq)

/* Tensor #664 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output22, AI_STATIC,
  664, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output22_array, &unpack_254_output22_array_intq)

/* Tensor #665 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output23, AI_STATIC,
  665, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output23_array, &unpack_254_output23_array_intq)

/* Tensor #666 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output24, AI_STATIC,
  666, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output24_array, &unpack_254_output24_array_intq)

/* Tensor #667 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output25, AI_STATIC,
  667, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output25_array, &unpack_254_output25_array_intq)

/* Tensor #668 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output26, AI_STATIC,
  668, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output26_array, &unpack_254_output26_array_intq)

/* Tensor #669 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output27, AI_STATIC,
  669, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output27_array, &unpack_254_output27_array_intq)

/* Tensor #670 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output28, AI_STATIC,
  670, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output28_array, &unpack_254_output28_array_intq)

/* Tensor #671 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output29, AI_STATIC,
  671, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output29_array, &unpack_254_output29_array_intq)

/* Tensor #672 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output3, AI_STATIC,
  672, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output3_array, &unpack_254_output3_array_intq)

/* Tensor #673 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output30, AI_STATIC,
  673, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output30_array, &unpack_254_output30_array_intq)

/* Tensor #674 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output31, AI_STATIC,
  674, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output31_array, &unpack_254_output31_array_intq)

/* Tensor #675 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output32, AI_STATIC,
  675, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output32_array, &unpack_254_output32_array_intq)

/* Tensor #676 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output33, AI_STATIC,
  676, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output33_array, &unpack_254_output33_array_intq)

/* Tensor #677 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output34, AI_STATIC,
  677, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output34_array, &unpack_254_output34_array_intq)

/* Tensor #678 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output35, AI_STATIC,
  678, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output35_array, &unpack_254_output35_array_intq)

/* Tensor #679 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output36, AI_STATIC,
  679, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output36_array, &unpack_254_output36_array_intq)

/* Tensor #680 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output37, AI_STATIC,
  680, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output37_array, &unpack_254_output37_array_intq)

/* Tensor #681 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output38, AI_STATIC,
  681, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output38_array, &unpack_254_output38_array_intq)

/* Tensor #682 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output39, AI_STATIC,
  682, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output39_array, &unpack_254_output39_array_intq)

/* Tensor #683 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output4, AI_STATIC,
  683, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output4_array, &unpack_254_output4_array_intq)

/* Tensor #684 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output40, AI_STATIC,
  684, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output40_array, &unpack_254_output40_array_intq)

/* Tensor #685 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output41, AI_STATIC,
  685, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output41_array, &unpack_254_output41_array_intq)

/* Tensor #686 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output42, AI_STATIC,
  686, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output42_array, &unpack_254_output42_array_intq)

/* Tensor #687 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output43, AI_STATIC,
  687, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output43_array, &unpack_254_output43_array_intq)

/* Tensor #688 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output44, AI_STATIC,
  688, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output44_array, &unpack_254_output44_array_intq)

/* Tensor #689 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output45, AI_STATIC,
  689, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output45_array, &unpack_254_output45_array_intq)

/* Tensor #690 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output46, AI_STATIC,
  690, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output46_array, &unpack_254_output46_array_intq)

/* Tensor #691 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output47, AI_STATIC,
  691, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output47_array, &unpack_254_output47_array_intq)

/* Tensor #692 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output5, AI_STATIC,
  692, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output5_array, &unpack_254_output5_array_intq)

/* Tensor #693 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output6, AI_STATIC,
  693, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output6_array, &unpack_254_output6_array_intq)

/* Tensor #694 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output7, AI_STATIC,
  694, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output7_array, &unpack_254_output7_array_intq)

/* Tensor #695 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output8, AI_STATIC,
  695, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output8_array, &unpack_254_output8_array_intq)

/* Tensor #696 */
AI_TENSOR_OBJ_DECLARE(
  unpack_254_output9, AI_STATIC,
  696, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &unpack_254_output9_array, &unpack_254_output9_array_intq)

/* Tensor #697 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output0, AI_STATIC,
  697, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output0_array, &unpack_7_output0_array_intq)

/* Tensor #698 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output1, AI_STATIC,
  698, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output1_array, &unpack_7_output1_array_intq)

/* Tensor #699 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output10, AI_STATIC,
  699, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output10_array, &unpack_7_output10_array_intq)

/* Tensor #700 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output11, AI_STATIC,
  700, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output11_array, &unpack_7_output11_array_intq)

/* Tensor #701 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output12, AI_STATIC,
  701, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output12_array, &unpack_7_output12_array_intq)

/* Tensor #702 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output13, AI_STATIC,
  702, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output13_array, &unpack_7_output13_array_intq)

/* Tensor #703 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output14, AI_STATIC,
  703, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output14_array, &unpack_7_output14_array_intq)

/* Tensor #704 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output15, AI_STATIC,
  704, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output15_array, &unpack_7_output15_array_intq)

/* Tensor #705 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output16, AI_STATIC,
  705, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output16_array, &unpack_7_output16_array_intq)

/* Tensor #706 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output17, AI_STATIC,
  706, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output17_array, &unpack_7_output17_array_intq)

/* Tensor #707 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output18, AI_STATIC,
  707, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output18_array, &unpack_7_output18_array_intq)

/* Tensor #708 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output19, AI_STATIC,
  708, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output19_array, &unpack_7_output19_array_intq)

/* Tensor #709 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output2, AI_STATIC,
  709, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output2_array, &unpack_7_output2_array_intq)

/* Tensor #710 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output20, AI_STATIC,
  710, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output20_array, &unpack_7_output20_array_intq)

/* Tensor #711 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output21, AI_STATIC,
  711, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output21_array, &unpack_7_output21_array_intq)

/* Tensor #712 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output22, AI_STATIC,
  712, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output22_array, &unpack_7_output22_array_intq)

/* Tensor #713 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output23, AI_STATIC,
  713, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output23_array, &unpack_7_output23_array_intq)

/* Tensor #714 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output24, AI_STATIC,
  714, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output24_array, &unpack_7_output24_array_intq)

/* Tensor #715 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output25, AI_STATIC,
  715, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output25_array, &unpack_7_output25_array_intq)

/* Tensor #716 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output26, AI_STATIC,
  716, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output26_array, &unpack_7_output26_array_intq)

/* Tensor #717 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output27, AI_STATIC,
  717, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output27_array, &unpack_7_output27_array_intq)

/* Tensor #718 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output28, AI_STATIC,
  718, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output28_array, &unpack_7_output28_array_intq)

/* Tensor #719 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output29, AI_STATIC,
  719, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output29_array, &unpack_7_output29_array_intq)

/* Tensor #720 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output3, AI_STATIC,
  720, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output3_array, &unpack_7_output3_array_intq)

/* Tensor #721 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output30, AI_STATIC,
  721, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output30_array, &unpack_7_output30_array_intq)

/* Tensor #722 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output31, AI_STATIC,
  722, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output31_array, &unpack_7_output31_array_intq)

/* Tensor #723 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output32, AI_STATIC,
  723, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output32_array, &unpack_7_output32_array_intq)

/* Tensor #724 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output33, AI_STATIC,
  724, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output33_array, &unpack_7_output33_array_intq)

/* Tensor #725 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output34, AI_STATIC,
  725, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output34_array, &unpack_7_output34_array_intq)

/* Tensor #726 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output35, AI_STATIC,
  726, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output35_array, &unpack_7_output35_array_intq)

/* Tensor #727 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output36, AI_STATIC,
  727, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output36_array, &unpack_7_output36_array_intq)

/* Tensor #728 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output37, AI_STATIC,
  728, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output37_array, &unpack_7_output37_array_intq)

/* Tensor #729 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output38, AI_STATIC,
  729, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output38_array, &unpack_7_output38_array_intq)

/* Tensor #730 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output39, AI_STATIC,
  730, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output39_array, &unpack_7_output39_array_intq)

/* Tensor #731 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output4, AI_STATIC,
  731, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output4_array, &unpack_7_output4_array_intq)

/* Tensor #732 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output40, AI_STATIC,
  732, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output40_array, &unpack_7_output40_array_intq)

/* Tensor #733 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output41, AI_STATIC,
  733, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output41_array, &unpack_7_output41_array_intq)

/* Tensor #734 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output42, AI_STATIC,
  734, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output42_array, &unpack_7_output42_array_intq)

/* Tensor #735 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output43, AI_STATIC,
  735, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output43_array, &unpack_7_output43_array_intq)

/* Tensor #736 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output44, AI_STATIC,
  736, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output44_array, &unpack_7_output44_array_intq)

/* Tensor #737 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output45, AI_STATIC,
  737, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output45_array, &unpack_7_output45_array_intq)

/* Tensor #738 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output46, AI_STATIC,
  738, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output46_array, &unpack_7_output46_array_intq)

/* Tensor #739 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output47, AI_STATIC,
  739, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output47_array, &unpack_7_output47_array_intq)

/* Tensor #740 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output5, AI_STATIC,
  740, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output5_array, &unpack_7_output5_array_intq)

/* Tensor #741 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output6, AI_STATIC,
  741, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output6_array, &unpack_7_output6_array_intq)

/* Tensor #742 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output7, AI_STATIC,
  742, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output7_array, &unpack_7_output7_array_intq)

/* Tensor #743 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output8, AI_STATIC,
  743, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output8_array, &unpack_7_output8_array_intq)

/* Tensor #744 */
AI_TENSOR_OBJ_DECLARE(
  unpack_7_output9, AI_STATIC,
  744, 0x1,
  AI_SHAPE_INIT(4, 1, 9, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9, 9),
  1, &unpack_7_output9_array, &unpack_7_output9_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_448_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &slice_0_output0, &gemm_447_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_448_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_448_layer, 448,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_448_chain,
  NULL, &eltwise_448_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_0_axes_data[] = { 0, 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_0_axes, AI_ARRAY_FORMAT_U8,
    slice_0_axes_data, slice_0_axes_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_0_starts_data[] = { 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_0_starts, AI_ARRAY_FORMAT_S16,
    slice_0_starts_data, slice_0_starts_data, 2, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_0_ends_data[] = { 24, 1 };
AI_ARRAY_OBJ_DECLARE(
    slice_0_ends, AI_ARRAY_FORMAT_S16,
    slice_0_ends_data, slice_0_ends_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_sensor_history0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_0_layer, 0,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_0_chain,
  NULL, &eltwise_448_layer, AI_STATIC, 
  .axes = &slice_0_axes, 
  .starts = &slice_0_starts, 
  .ends = &slice_0_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_447_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_446_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_447_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_447_weights, &gemm_447_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_447_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_447_layer, 447,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_447_chain,
  NULL, &slice_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_446_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_445_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_446_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_446_weights, &gemm_446_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_446_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_446_layer, 446,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_446_chain,
  NULL, &gemm_447_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_445_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -108, -106, -104, -101, -99, -96, -94, -90, -87, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 87, 90, 94, 96, 99, 101, 104, 106, 108, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_445_nl_params, AI_ARRAY_FORMAT_S8,
    nl_445_nl_params_data, nl_445_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_445_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_444_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_445_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_445_layer, 445,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_445_chain,
  NULL, &gemm_446_layer, AI_STATIC, 
  .nl_params = &nl_445_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_444_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_443_output, &gemm_442_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_444_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_444_layer, 444,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_444_chain,
  NULL, &nl_445_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_443_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output47),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_443_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_443_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_443_layer, 443,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_443_chain,
  NULL, &eltwise_444_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_442_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_441_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_442_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_442_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_442_layer, 442,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_442_chain,
  NULL, &gemm_443_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_441_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -98, -96, -93, -90, -87, -83, -80, -76, -71, -67, -62, -58, -53, -47, -42, -36, -30, -25, -18, -12, -6, 0, 6, 12, 18, 25, 30, 36, 42, 47, 53, 58, 62, 67, 71, 76, 80, 83, 87, 90, 93, 96, 98, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_441_nl_params, AI_ARRAY_FORMAT_S8,
    nl_441_nl_params_data, nl_441_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_441_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_440_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_441_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_441_layer, 441,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_441_chain,
  NULL, &gemm_442_layer, AI_STATIC, 
  .nl_params = &nl_441_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_440_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_439_output, &gemm_438_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_440_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_440_layer, 440,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_440_chain,
  NULL, &nl_441_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_439_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output46),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_439_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_439_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_439_layer, 439,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_439_chain,
  NULL, &eltwise_440_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_438_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_437_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_438_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_438_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_438_layer, 438,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_438_chain,
  NULL, &gemm_439_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_437_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_437_nl_params, AI_ARRAY_FORMAT_S8,
    nl_437_nl_params_data, nl_437_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_437_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_436_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_437_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_437_layer, 437,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_437_chain,
  NULL, &gemm_438_layer, AI_STATIC, 
  .nl_params = &nl_437_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_436_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_435_output, &gemm_434_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_436_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_436_layer, 436,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_436_chain,
  NULL, &nl_437_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_435_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output45),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_435_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_435_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_435_layer, 435,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_435_chain,
  NULL, &eltwise_436_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_434_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_433_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_434_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_434_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_434_layer, 434,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_434_chain,
  NULL, &gemm_435_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_433_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_433_nl_params, AI_ARRAY_FORMAT_S8,
    nl_433_nl_params_data, nl_433_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_433_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_432_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_433_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_433_layer, 433,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_433_chain,
  NULL, &gemm_434_layer, AI_STATIC, 
  .nl_params = &nl_433_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_432_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_310_output, &gemm_431_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_432_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_432_layer, 432,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_432_chain,
  NULL, &nl_433_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_310_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output44),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_310_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_310_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_310_layer, 310,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_310_chain,
  NULL, &eltwise_432_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_431_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_430_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_431_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_431_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_431_layer, 431,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_431_chain,
  NULL, &gemm_310_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_430_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_430_nl_params, AI_ARRAY_FORMAT_S8,
    nl_430_nl_params_data, nl_430_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_430_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_429_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_430_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_430_layer, 430,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_430_chain,
  NULL, &gemm_431_layer, AI_STATIC, 
  .nl_params = &nl_430_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_429_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_309_output, &gemm_428_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_429_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_429_layer, 429,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_429_chain,
  NULL, &nl_430_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_309_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output43),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_309_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_309_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_309_layer, 309,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_309_chain,
  NULL, &eltwise_429_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_428_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_427_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_428_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_428_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_428_layer, 428,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_428_chain,
  NULL, &gemm_309_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_427_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_427_nl_params, AI_ARRAY_FORMAT_S8,
    nl_427_nl_params_data, nl_427_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_427_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_426_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_427_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_427_layer, 427,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_427_chain,
  NULL, &gemm_428_layer, AI_STATIC, 
  .nl_params = &nl_427_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_426_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_308_output, &gemm_425_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_426_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_426_layer, 426,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_426_chain,
  NULL, &nl_427_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_308_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output42),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_308_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_308_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_308_layer, 308,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_308_chain,
  NULL, &eltwise_426_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_425_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_424_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_425_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_425_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_425_layer, 425,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_425_chain,
  NULL, &gemm_308_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_424_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -114, -113, -112, -110, -108, -106, -104, -102, -100, -97, -94, -91, -88, -85, -81, -77, -73, -68, -64, -59, -54, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 54, 59, 64, 68, 73, 77, 81, 85, 88, 91, 94, 97, 100, 102, 104, 106, 108, 110, 112, 113, 114, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_424_nl_params, AI_ARRAY_FORMAT_S8,
    nl_424_nl_params_data, nl_424_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_424_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_423_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_424_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_424_layer, 424,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_424_chain,
  NULL, &gemm_425_layer, AI_STATIC, 
  .nl_params = &nl_424_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_423_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_307_output, &gemm_422_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_423_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_423_layer, 423,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_423_chain,
  NULL, &nl_424_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_307_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output41),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_307_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_307_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_307_layer, 307,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_307_chain,
  NULL, &eltwise_423_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_422_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_421_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_422_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_422_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_422_layer, 422,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_422_chain,
  NULL, &gemm_307_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_421_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -119, -119, -118, -117, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -81, -77, -73, -68, -63, -59, -53, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 53, 59, 63, 68, 73, 77, 81, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 117, 118, 119, 119, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_421_nl_params, AI_ARRAY_FORMAT_S8,
    nl_421_nl_params_data, nl_421_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_421_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_420_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_421_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_421_layer, 421,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_421_chain,
  NULL, &gemm_422_layer, AI_STATIC, 
  .nl_params = &nl_421_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_420_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_306_output, &gemm_419_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_420_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_420_layer, 420,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_420_chain,
  NULL, &nl_421_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_306_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output40),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_306_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_306_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_306_layer, 306,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_306_chain,
  NULL, &eltwise_420_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_419_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_418_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_419_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_419_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_419_layer, 419,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_419_chain,
  NULL, &gemm_306_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_418_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -108, -106, -104, -101, -99, -96, -93, -90, -87, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 87, 90, 93, 96, 99, 101, 104, 106, 108, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_418_nl_params, AI_ARRAY_FORMAT_S8,
    nl_418_nl_params_data, nl_418_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_418_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_417_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_418_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_418_layer, 418,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_418_chain,
  NULL, &gemm_419_layer, AI_STATIC, 
  .nl_params = &nl_418_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_417_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_305_output, &gemm_416_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_417_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_417_layer, 417,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_417_chain,
  NULL, &nl_418_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_305_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output39),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_305_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_305_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_305_layer, 305,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_305_chain,
  NULL, &eltwise_417_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_416_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_415_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_416_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_416_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_416_layer, 416,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_416_chain,
  NULL, &gemm_305_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_415_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -113, -112, -110, -108, -107, -104, -102, -100, -97, -94, -91, -88, -85, -81, -77, -73, -69, -64, -59, -54, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 54, 59, 64, 69, 73, 77, 81, 85, 88, 91, 94, 97, 100, 102, 104, 107, 108, 110, 112, 113, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_415_nl_params, AI_ARRAY_FORMAT_S8,
    nl_415_nl_params_data, nl_415_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_415_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_414_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_415_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_415_layer, 415,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_415_chain,
  NULL, &gemm_416_layer, AI_STATIC, 
  .nl_params = &nl_415_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_414_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_304_output, &gemm_413_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_414_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_414_layer, 414,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_414_chain,
  NULL, &nl_415_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_304_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output38),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_304_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_304_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_304_layer, 304,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_304_chain,
  NULL, &eltwise_414_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_413_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_412_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_413_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_413_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_413_layer, 413,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_413_chain,
  NULL, &gemm_304_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_412_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -100, -98, -95, -92, -89, -85, -82, -78, -73, -69, -64, -59, -54, -49, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 49, 54, 59, 64, 69, 73, 78, 82, 85, 89, 92, 95, 98, 100, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_412_nl_params, AI_ARRAY_FORMAT_S8,
    nl_412_nl_params_data, nl_412_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_412_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_411_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_412_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_412_layer, 412,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_412_chain,
  NULL, &gemm_413_layer, AI_STATIC, 
  .nl_params = &nl_412_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_411_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_303_output, &gemm_410_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_411_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_411_layer, 411,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_411_chain,
  NULL, &nl_412_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_303_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output37),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_303_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_303_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_303_layer, 303,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_303_chain,
  NULL, &eltwise_411_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_410_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_409_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_410_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_410_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_410_layer, 410,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_410_chain,
  NULL, &gemm_303_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_409_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_409_nl_params, AI_ARRAY_FORMAT_S8,
    nl_409_nl_params_data, nl_409_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_409_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_408_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_409_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_409_layer, 409,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_409_chain,
  NULL, &gemm_410_layer, AI_STATIC, 
  .nl_params = &nl_409_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_408_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_302_output, &gemm_407_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_408_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_408_layer, 408,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_408_chain,
  NULL, &nl_409_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_302_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output36),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_302_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_302_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_302_layer, 302,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_302_chain,
  NULL, &eltwise_408_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_407_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_406_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_407_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_407_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_407_layer, 407,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_407_chain,
  NULL, &gemm_302_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_406_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_406_nl_params, AI_ARRAY_FORMAT_S8,
    nl_406_nl_params_data, nl_406_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_406_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_405_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_406_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_406_layer, 406,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_406_chain,
  NULL, &gemm_407_layer, AI_STATIC, 
  .nl_params = &nl_406_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_405_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_301_output, &gemm_404_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_405_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_405_layer, 405,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_405_chain,
  NULL, &nl_406_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_301_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output35),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_301_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_301_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_301_layer, 301,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_301_chain,
  NULL, &eltwise_405_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_404_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_403_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_404_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_404_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_404_layer, 404,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_404_chain,
  NULL, &gemm_301_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_403_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -113, -112, -111, -109, -107, -105, -103, -101, -98, -96, -93, -90, -87, -83, -79, -76, -71, -67, -62, -58, -52, -47, -42, -36, -30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30, 36, 42, 47, 52, 58, 62, 67, 71, 76, 79, 83, 87, 90, 93, 96, 98, 101, 103, 105, 107, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_403_nl_params, AI_ARRAY_FORMAT_S8,
    nl_403_nl_params_data, nl_403_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_403_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_402_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_403_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_403_layer, 403,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_403_chain,
  NULL, &gemm_404_layer, AI_STATIC, 
  .nl_params = &nl_403_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_402_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_296_output, &gemm_401_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_402_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_402_layer, 402,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_402_chain,
  NULL, &nl_403_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_296_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output34),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_296_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_296_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_296_layer, 296,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_296_chain,
  NULL, &eltwise_402_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_401_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_400_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_401_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_401_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_401_layer, 401,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_401_chain,
  NULL, &gemm_296_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_400_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -113, -111, -109, -108, -106, -104, -101, -99, -96, -94, -91, -87, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 87, 91, 94, 96, 99, 101, 104, 106, 108, 109, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_400_nl_params, AI_ARRAY_FORMAT_S8,
    nl_400_nl_params_data, nl_400_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_400_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_399_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_400_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_400_layer, 400,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_400_chain,
  NULL, &gemm_401_layer, AI_STATIC, 
  .nl_params = &nl_400_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_399_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_295_output, &gemm_398_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_399_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_399_layer, 399,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_399_chain,
  NULL, &nl_400_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_295_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output33),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_295_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_295_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_295_layer, 295,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_295_chain,
  NULL, &eltwise_399_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_398_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_397_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_398_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_398_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_398_layer, 398,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_398_chain,
  NULL, &gemm_295_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_397_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -119, -119, -118, -117, -116, -114, -113, -112, -110, -109, -107, -105, -103, -101, -98, -95, -93, -90, -86, -83, -79, -75, -71, -67, -62, -57, -52, -47, -42, -36, -30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30, 36, 42, 47, 52, 57, 62, 67, 71, 75, 79, 83, 86, 90, 93, 95, 98, 101, 103, 105, 107, 109, 110, 112, 113, 114, 116, 117, 118, 119, 119, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_397_nl_params, AI_ARRAY_FORMAT_S8,
    nl_397_nl_params_data, nl_397_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_397_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_396_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_397_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_397_layer, 397,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_397_chain,
  NULL, &gemm_398_layer, AI_STATIC, 
  .nl_params = &nl_397_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_396_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_294_output, &gemm_395_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_396_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_396_layer, 396,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_396_chain,
  NULL, &nl_397_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_294_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output32),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_294_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_294_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_294_layer, 294,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_294_chain,
  NULL, &eltwise_396_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_395_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_394_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_395_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_395_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_395_layer, 395,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_395_chain,
  NULL, &gemm_294_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_394_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -30, -25, -19, -12, -6, 0, 6, 12, 19, 25, 30, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_394_nl_params, AI_ARRAY_FORMAT_S8,
    nl_394_nl_params_data, nl_394_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_394_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_393_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_394_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_394_layer, 394,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_394_chain,
  NULL, &gemm_395_layer, AI_STATIC, 
  .nl_params = &nl_394_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_393_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_293_output, &gemm_392_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_393_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_393_layer, 393,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_393_chain,
  NULL, &nl_394_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_293_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output31),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_293_layer, 293,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_293_chain,
  NULL, &eltwise_393_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_392_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_391_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_392_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_392_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_392_layer, 392,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_392_chain,
  NULL, &gemm_293_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_391_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_391_nl_params, AI_ARRAY_FORMAT_S8,
    nl_391_nl_params_data, nl_391_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_391_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_390_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_391_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_391_layer, 391,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_391_chain,
  NULL, &gemm_392_layer, AI_STATIC, 
  .nl_params = &nl_391_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_390_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_292_output, &gemm_389_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_390_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_390_layer, 390,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_390_chain,
  NULL, &nl_391_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_292_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output30),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_292_layer, 292,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_292_chain,
  NULL, &eltwise_390_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_389_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_388_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_389_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_389_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_389_layer, 389,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_389_chain,
  NULL, &gemm_292_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_388_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_388_nl_params, AI_ARRAY_FORMAT_S8,
    nl_388_nl_params_data, nl_388_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_388_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_387_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_388_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_388_layer, 388,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_388_chain,
  NULL, &gemm_389_layer, AI_STATIC, 
  .nl_params = &nl_388_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_387_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_291_output, &gemm_386_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_387_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_387_layer, 387,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_387_chain,
  NULL, &nl_388_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_291_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output29),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_291_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_291_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_291_layer, 291,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_291_chain,
  NULL, &eltwise_387_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_386_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_385_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_386_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_386_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_386_layer, 386,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_386_chain,
  NULL, &gemm_291_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_385_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -113, -111, -109, -108, -106, -104, -101, -99, -96, -94, -91, -87, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 87, 91, 94, 96, 99, 101, 104, 106, 108, 109, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_385_nl_params, AI_ARRAY_FORMAT_S8,
    nl_385_nl_params_data, nl_385_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_385_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_384_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_385_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_385_layer, 385,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_385_chain,
  NULL, &gemm_386_layer, AI_STATIC, 
  .nl_params = &nl_385_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_384_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_290_output, &gemm_383_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_384_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_384_layer, 384,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_384_chain,
  NULL, &nl_385_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_290_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output28),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_290_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_290_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_290_layer, 290,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_290_chain,
  NULL, &eltwise_384_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_383_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_382_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_383_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_383_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_383_layer, 383,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_383_chain,
  NULL, &gemm_290_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_382_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -108, -106, -104, -101, -99, -96, -93, -90, -87, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 87, 90, 93, 96, 99, 101, 104, 106, 108, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_382_nl_params, AI_ARRAY_FORMAT_S8,
    nl_382_nl_params_data, nl_382_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_382_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_381_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_382_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_382_layer, 382,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_382_chain,
  NULL, &gemm_383_layer, AI_STATIC, 
  .nl_params = &nl_382_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_381_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_289_output, &gemm_380_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_381_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_381_layer, 381,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_381_chain,
  NULL, &nl_382_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_289_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output27),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_289_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_289_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_289_layer, 289,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_289_chain,
  NULL, &eltwise_381_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_380_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_379_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_380_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_380_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_380_layer, 380,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_380_chain,
  NULL, &gemm_289_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_379_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -119, -119, -118, -117, -116, -114, -113, -112, -110, -108, -106, -104, -102, -100, -97, -94, -91, -88, -84, -81, -77, -73, -68, -64, -59, -54, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 54, 59, 64, 68, 73, 77, 81, 84, 88, 91, 94, 97, 100, 102, 104, 106, 108, 110, 112, 113, 114, 116, 117, 118, 119, 119, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_379_nl_params, AI_ARRAY_FORMAT_S8,
    nl_379_nl_params_data, nl_379_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_379_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_378_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_379_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_379_layer, 379,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_379_chain,
  NULL, &gemm_380_layer, AI_STATIC, 
  .nl_params = &nl_379_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_378_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_288_output, &gemm_377_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_378_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_378_layer, 378,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_378_chain,
  NULL, &nl_379_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_288_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output26),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_288_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_288_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_288_layer, 288,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_288_chain,
  NULL, &eltwise_378_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_377_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_376_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_377_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_377_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_377_layer, 377,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_377_chain,
  NULL, &gemm_288_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_376_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -106, -103, -101, -99, -96, -93, -90, -87, -84, -80, -76, -72, -67, -63, -58, -53, -48, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 48, 53, 58, 63, 67, 72, 76, 80, 84, 87, 90, 93, 96, 99, 101, 103, 106, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_376_nl_params, AI_ARRAY_FORMAT_S8,
    nl_376_nl_params_data, nl_376_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_376_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_375_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_376_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_376_layer, 376,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_376_chain,
  NULL, &gemm_377_layer, AI_STATIC, 
  .nl_params = &nl_376_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_375_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_287_output, &gemm_374_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_375_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_375_layer, 375,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_375_chain,
  NULL, &nl_376_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_287_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output25),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_287_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_287_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_287_layer, 287,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_287_chain,
  NULL, &eltwise_375_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_374_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_373_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_374_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_374_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_374_layer, 374,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_374_chain,
  NULL, &gemm_287_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_373_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -113, -112, -111, -109, -107, -105, -103, -101, -98, -96, -93, -90, -87, -83, -80, -76, -71, -67, -62, -58, -53, -47, -42, -36, -30, -25, -18, -12, -6, 0, 6, 12, 18, 25, 30, 36, 42, 47, 53, 58, 62, 67, 71, 76, 80, 83, 87, 90, 93, 96, 98, 101, 103, 105, 107, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_373_nl_params, AI_ARRAY_FORMAT_S8,
    nl_373_nl_params_data, nl_373_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_373_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_372_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_373_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_373_layer, 373,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_373_chain,
  NULL, &gemm_374_layer, AI_STATIC, 
  .nl_params = &nl_373_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_372_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_282_output, &gemm_371_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_372_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_372_layer, 372,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_372_chain,
  NULL, &nl_373_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_282_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output24),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_282_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_282_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_282_layer, 282,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_282_chain,
  NULL, &eltwise_372_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_371_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_370_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_371_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_371_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_371_layer, 371,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_371_chain,
  NULL, &gemm_282_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_370_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_370_nl_params, AI_ARRAY_FORMAT_S8,
    nl_370_nl_params_data, nl_370_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_370_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_369_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_370_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_370_layer, 370,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_370_chain,
  NULL, &gemm_371_layer, AI_STATIC, 
  .nl_params = &nl_370_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_369_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_281_output, &gemm_368_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_369_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_369_layer, 369,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_369_chain,
  NULL, &nl_370_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_281_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output23),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_281_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_281_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_281_layer, 281,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_281_chain,
  NULL, &eltwise_369_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_368_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_367_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_368_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_368_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_368_layer, 368,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_368_chain,
  NULL, &gemm_281_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_367_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -84, -80, -76, -72, -67, -63, -58, -53, -48, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 48, 53, 58, 63, 67, 72, 76, 80, 84, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_367_nl_params, AI_ARRAY_FORMAT_S8,
    nl_367_nl_params_data, nl_367_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_367_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_366_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_367_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_367_layer, 367,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_367_chain,
  NULL, &gemm_368_layer, AI_STATIC, 
  .nl_params = &nl_367_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_366_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_280_output, &gemm_365_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_366_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_366_layer, 366,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_366_chain,
  NULL, &nl_367_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_280_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output22),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_280_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_280_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_280_layer, 280,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_280_chain,
  NULL, &eltwise_366_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_365_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_364_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_365_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_365_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_365_layer, 365,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_365_chain,
  NULL, &gemm_280_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_364_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -113, -112, -110, -109, -107, -105, -103, -101, -98, -96, -93, -90, -86, -83, -79, -75, -71, -67, -62, -57, -52, -47, -42, -36, -30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30, 36, 42, 47, 52, 57, 62, 67, 71, 75, 79, 83, 86, 90, 93, 96, 98, 101, 103, 105, 107, 109, 110, 112, 113, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_364_nl_params, AI_ARRAY_FORMAT_S8,
    nl_364_nl_params_data, nl_364_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_364_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_363_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_364_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_364_layer, 364,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_364_chain,
  NULL, &gemm_365_layer, AI_STATIC, 
  .nl_params = &nl_364_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_363_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_279_output, &gemm_362_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_363_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_363_layer, 363,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_363_chain,
  NULL, &nl_364_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_279_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output21),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_279_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_279_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_279_layer, 279,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_279_chain,
  NULL, &eltwise_363_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_362_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_361_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_362_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_362_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_362_layer, 362,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_362_chain,
  NULL, &gemm_279_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_361_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_361_nl_params, AI_ARRAY_FORMAT_S8,
    nl_361_nl_params_data, nl_361_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_361_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_360_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_361_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_361_layer, 361,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_361_chain,
  NULL, &gemm_362_layer, AI_STATIC, 
  .nl_params = &nl_361_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_360_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_278_output, &gemm_359_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_360_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_360_layer, 360,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_360_chain,
  NULL, &nl_361_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_278_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output20),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_278_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_278_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_278_layer, 278,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_278_chain,
  NULL, &eltwise_360_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_359_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_358_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_359_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_359_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_359_layer, 359,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_359_chain,
  NULL, &gemm_278_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_358_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -108, -106, -104, -101, -99, -96, -93, -90, -87, -84, -80, -76, -72, -67, -63, -58, -53, -48, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 48, 53, 58, 63, 67, 72, 76, 80, 84, 87, 90, 93, 96, 99, 101, 104, 106, 108, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_358_nl_params, AI_ARRAY_FORMAT_S8,
    nl_358_nl_params_data, nl_358_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_358_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_357_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_358_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_358_layer, 358,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_358_chain,
  NULL, &gemm_359_layer, AI_STATIC, 
  .nl_params = &nl_358_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_357_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_277_output, &gemm_356_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_357_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_357_layer, 357,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_357_chain,
  NULL, &nl_358_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_277_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output19),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_277_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_277_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_277_layer, 277,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_277_chain,
  NULL, &eltwise_357_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_356_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_355_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_356_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_356_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_356_layer, 356,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_356_chain,
  NULL, &gemm_277_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_355_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -121, -121, -120, -119, -118, -118, -116, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -80, -77, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 77, 80, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 116, 118, 118, 119, 120, 121, 121, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_355_nl_params, AI_ARRAY_FORMAT_S8,
    nl_355_nl_params_data, nl_355_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_355_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_354_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_355_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_355_layer, 355,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_355_chain,
  NULL, &gemm_356_layer, AI_STATIC, 
  .nl_params = &nl_355_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_354_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_276_output, &gemm_353_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_354_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_354_layer, 354,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_354_chain,
  NULL, &nl_355_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_276_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output18),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_276_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_276_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_276_layer, 276,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_276_chain,
  NULL, &eltwise_354_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_353_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_352_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_353_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_353_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_353_layer, 353,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_353_chain,
  NULL, &gemm_276_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_352_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -113, -112, -111, -109, -107, -105, -103, -101, -98, -96, -93, -90, -87, -83, -80, -76, -71, -67, -62, -58, -53, -47, -42, -36, -30, -25, -18, -12, -6, 0, 6, 12, 18, 25, 30, 36, 42, 47, 53, 58, 62, 67, 71, 76, 80, 83, 87, 90, 93, 96, 98, 101, 103, 105, 107, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_352_nl_params, AI_ARRAY_FORMAT_S8,
    nl_352_nl_params_data, nl_352_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_352_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_351_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_352_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_352_layer, 352,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_352_chain,
  NULL, &gemm_353_layer, AI_STATIC, 
  .nl_params = &nl_352_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_351_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_275_output, &gemm_350_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_351_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_351_layer, 351,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_351_chain,
  NULL, &nl_352_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_275_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output17),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_275_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_275_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_275_layer, 275,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_275_chain,
  NULL, &eltwise_351_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_350_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_349_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_350_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_350_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_350_layer, 350,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_350_chain,
  NULL, &gemm_275_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_349_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -119, -119, -118, -117, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -81, -77, -73, -68, -63, -59, -53, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 53, 59, 63, 68, 73, 77, 81, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 117, 118, 119, 119, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_349_nl_params, AI_ARRAY_FORMAT_S8,
    nl_349_nl_params_data, nl_349_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_349_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_348_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_349_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_349_layer, 349,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_349_chain,
  NULL, &gemm_350_layer, AI_STATIC, 
  .nl_params = &nl_349_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_348_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_274_output, &gemm_347_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_348_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_348_layer, 348,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_348_chain,
  NULL, &nl_349_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_274_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output16),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_274_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_274_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_274_layer, 274,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_274_chain,
  NULL, &eltwise_348_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_347_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_346_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_347_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_347_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_347_layer, 347,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_347_chain,
  NULL, &gemm_274_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_346_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_346_nl_params, AI_ARRAY_FORMAT_S8,
    nl_346_nl_params_data, nl_346_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_346_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_345_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_346_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_346_layer, 346,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_346_chain,
  NULL, &gemm_347_layer, AI_STATIC, 
  .nl_params = &nl_346_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_345_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_273_output, &gemm_344_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_345_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_345_layer, 345,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_345_chain,
  NULL, &nl_346_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_273_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output15),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_273_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_273_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_273_layer, 273,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_273_chain,
  NULL, &eltwise_345_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_344_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_343_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_344_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_344_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_344_layer, 344,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_344_chain,
  NULL, &gemm_273_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_343_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -108, -106, -104, -101, -99, -96, -94, -91, -87, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 87, 91, 94, 96, 99, 101, 104, 106, 108, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_343_nl_params, AI_ARRAY_FORMAT_S8,
    nl_343_nl_params_data, nl_343_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_343_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_342_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_343_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_343_layer, 343,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_343_chain,
  NULL, &gemm_344_layer, AI_STATIC, 
  .nl_params = &nl_343_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_342_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_268_output, &gemm_341_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_342_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_342_layer, 342,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_342_chain,
  NULL, &nl_343_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_268_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output14),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_268_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_268_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_268_layer, 268,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_268_chain,
  NULL, &eltwise_342_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_341_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_340_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_341_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_341_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_341_layer, 341,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_341_chain,
  NULL, &gemm_268_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_340_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -119, -118, -118, -116, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -80, -77, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 77, 80, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 116, 118, 118, 119, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_340_nl_params, AI_ARRAY_FORMAT_S8,
    nl_340_nl_params_data, nl_340_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_340_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_339_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_340_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_340_layer, 340,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_340_chain,
  NULL, &gemm_341_layer, AI_STATIC, 
  .nl_params = &nl_340_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_339_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_267_output, &gemm_338_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_339_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_339_layer, 339,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_339_chain,
  NULL, &nl_340_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_267_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output13),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_267_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_267_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_267_layer, 267,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_267_chain,
  NULL, &eltwise_339_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_338_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_337_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_338_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_338_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_338_layer, 338,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_338_chain,
  NULL, &gemm_267_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_337_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -119, -119, -118, -117, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -81, -77, -72, -68, -63, -59, -53, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 53, 59, 63, 68, 72, 77, 81, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 117, 118, 119, 119, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_337_nl_params, AI_ARRAY_FORMAT_S8,
    nl_337_nl_params_data, nl_337_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_337_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_336_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_337_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_337_layer, 337,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_337_chain,
  NULL, &gemm_338_layer, AI_STATIC, 
  .nl_params = &nl_337_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_336_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_266_output, &gemm_335_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_336_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_336_layer, 336,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_336_chain,
  NULL, &nl_337_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_266_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output12),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_266_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_266_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_266_layer, 266,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_266_chain,
  NULL, &eltwise_336_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_335_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_334_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_335_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_335_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_335_layer, 335,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_335_chain,
  NULL, &gemm_266_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_334_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -108, -106, -104, -101, -99, -96, -94, -90, -87, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 87, 90, 94, 96, 99, 101, 104, 106, 108, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_334_nl_params, AI_ARRAY_FORMAT_S8,
    nl_334_nl_params_data, nl_334_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_334_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_333_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_334_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_334_layer, 334,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_334_chain,
  NULL, &gemm_335_layer, AI_STATIC, 
  .nl_params = &nl_334_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_333_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_265_output, &gemm_332_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_333_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_333_layer, 333,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_333_chain,
  NULL, &nl_334_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_265_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output11),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_265_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_265_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_265_layer, 265,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_265_chain,
  NULL, &eltwise_333_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_332_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_331_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_332_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_332_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_332_layer, 332,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_332_chain,
  NULL, &gemm_265_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_331_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -113, -112, -110, -109, -107, -105, -103, -100, -98, -95, -92, -89, -85, -82, -78, -73, -69, -64, -59, -54, -49, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 49, 54, 59, 64, 69, 73, 78, 82, 85, 89, 92, 95, 98, 100, 103, 105, 107, 109, 110, 112, 113, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_331_nl_params, AI_ARRAY_FORMAT_S8,
    nl_331_nl_params_data, nl_331_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_331_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_330_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_331_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_331_layer, 331,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_331_chain,
  NULL, &gemm_332_layer, AI_STATIC, 
  .nl_params = &nl_331_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_330_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_264_output, &gemm_329_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_330_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_330_layer, 330,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_330_chain,
  NULL, &nl_331_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_264_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output10),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_264_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_264_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_264_layer, 264,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_264_chain,
  NULL, &eltwise_330_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_329_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_328_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_329_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_329_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_329_layer, 329,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_329_chain,
  NULL, &gemm_264_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_328_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -80, -76, -72, -68, -63, -58, -53, -48, -42, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 42, 48, 53, 58, 63, 68, 72, 76, 80, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_328_nl_params, AI_ARRAY_FORMAT_S8,
    nl_328_nl_params_data, nl_328_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_328_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_327_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_328_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_328_layer, 328,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_328_chain,
  NULL, &gemm_329_layer, AI_STATIC, 
  .nl_params = &nl_328_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_327_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_263_output, &gemm_326_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_327_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_327_layer, 327,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_327_chain,
  NULL, &nl_328_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_263_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output9),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_263_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_263_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_263_layer, 263,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_263_chain,
  NULL, &eltwise_327_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_326_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_325_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_326_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_326_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_326_layer, 326,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_326_chain,
  NULL, &gemm_263_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_325_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_325_nl_params, AI_ARRAY_FORMAT_S8,
    nl_325_nl_params_data, nl_325_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_325_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_324_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_325_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_325_layer, 325,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_325_chain,
  NULL, &gemm_326_layer, AI_STATIC, 
  .nl_params = &nl_325_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_324_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_262_output, &gemm_323_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_324_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_324_layer, 324,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_324_chain,
  NULL, &nl_325_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_262_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output8),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_262_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_262_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_262_layer, 262,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_262_chain,
  NULL, &eltwise_324_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_323_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_322_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_323_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_323_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_323_layer, 323,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_323_chain,
  NULL, &gemm_262_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_322_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -106, -103, -101, -99, -96, -93, -90, -87, -84, -80, -76, -72, -67, -63, -58, -53, -48, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 48, 53, 58, 63, 67, 72, 76, 80, 84, 87, 90, 93, 96, 99, 101, 103, 106, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_322_nl_params, AI_ARRAY_FORMAT_S8,
    nl_322_nl_params_data, nl_322_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_322_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_321_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_322_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_322_layer, 322,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_322_chain,
  NULL, &gemm_323_layer, AI_STATIC, 
  .nl_params = &nl_322_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_321_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_261_output, &gemm_320_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_321_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_321_layer, 321,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_321_chain,
  NULL, &nl_322_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_261_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output7),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_261_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_261_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_261_layer, 261,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_261_chain,
  NULL, &eltwise_321_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_320_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_319_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_320_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_320_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_320_layer, 320,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_320_chain,
  NULL, &gemm_261_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_319_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -107, -105, -103, -101, -99, -96, -93, -90, -87, -83, -80, -76, -72, -67, -63, -58, -53, -47, -42, -36, -30, -25, -19, -12, -6, 0, 6, 12, 19, 25, 30, 36, 42, 47, 53, 58, 63, 67, 72, 76, 80, 83, 87, 90, 93, 96, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_319_nl_params, AI_ARRAY_FORMAT_S8,
    nl_319_nl_params_data, nl_319_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_319_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_318_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_319_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_319_layer, 319,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_319_chain,
  NULL, &gemm_320_layer, AI_STATIC, 
  .nl_params = &nl_319_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_318_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_260_output, &gemm_317_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_318_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_318_layer, 318,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_318_chain,
  NULL, &nl_319_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_260_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output6),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_260_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_260_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_260_layer, 260,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_260_chain,
  NULL, &eltwise_318_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_317_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_316_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_317_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_317_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_317_layer, 317,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_317_chain,
  NULL, &gemm_260_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_316_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -119, -119, -118, -117, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -81, -77, -73, -68, -63, -59, -53, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 53, 59, 63, 68, 73, 77, 81, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 117, 118, 119, 119, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_316_nl_params, AI_ARRAY_FORMAT_S8,
    nl_316_nl_params_data, nl_316_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_316_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_315_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_316_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_316_layer, 316,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_316_chain,
  NULL, &gemm_317_layer, AI_STATIC, 
  .nl_params = &nl_316_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_315_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_259_output, &gemm_314_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_315_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_315_layer, 315,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_315_chain,
  NULL, &nl_316_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_259_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output5),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_259_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_259_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_259_layer, 259,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_259_chain,
  NULL, &eltwise_315_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_314_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_313_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_314_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_314_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_314_layer, 314,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_314_chain,
  NULL, &gemm_259_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_313_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -119, -118, -117, -116, -115, -114, -112, -111, -109, -108, -106, -104, -101, -99, -96, -93, -90, -87, -84, -80, -76, -72, -67, -63, -58, -53, -48, -42, -36, -31, -25, -19, -12, -6, 0, 6, 12, 19, 25, 31, 36, 42, 48, 53, 58, 63, 67, 72, 76, 80, 84, 87, 90, 93, 96, 99, 101, 104, 106, 108, 109, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_313_nl_params, AI_ARRAY_FORMAT_S8,
    nl_313_nl_params_data, nl_313_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_313_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_312_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_313_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_313_layer, 313,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_313_chain,
  NULL, &gemm_314_layer, AI_STATIC, 
  .nl_params = &nl_313_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_312_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_311_output, &gemm_300_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_312_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_312_layer, 312,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_312_chain,
  NULL, &nl_313_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_311_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output4),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_311_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_311_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_311_layer, 311,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_311_chain,
  NULL, &eltwise_312_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_300_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_299_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_300_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_300_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_300_layer, 300,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_300_chain,
  NULL, &gemm_311_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_299_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -120, -119, -118, -117, -116, -114, -113, -112, -110, -108, -106, -104, -102, -100, -97, -94, -91, -88, -85, -81, -77, -73, -68, -64, -59, -54, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 54, 59, 64, 68, 73, 77, 81, 85, 88, 91, 94, 97, 100, 102, 104, 106, 108, 110, 112, 113, 114, 116, 117, 118, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_299_nl_params, AI_ARRAY_FORMAT_S8,
    nl_299_nl_params_data, nl_299_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_299_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_298_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_299_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_299_layer, 299,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_299_chain,
  NULL, &gemm_300_layer, AI_STATIC, 
  .nl_params = &nl_299_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_298_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_297_output, &gemm_286_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_298_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_298_layer, 298,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_298_chain,
  NULL, &nl_299_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_297_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output3),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_297_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_297_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_297_layer, 297,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_297_chain,
  NULL, &eltwise_298_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_286_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_285_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_286_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_286_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_286_layer, 286,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_286_chain,
  NULL, &gemm_297_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_285_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -123, -123, -122, -122, -121, -120, -119, -119, -118, -117, -115, -114, -113, -111, -110, -108, -106, -104, -102, -99, -97, -94, -91, -88, -84, -81, -77, -72, -68, -63, -59, -53, -48, -43, -37, -31, -25, -19, -13, -6, 0, 6, 13, 19, 25, 31, 37, 43, 48, 53, 59, 63, 68, 72, 77, 81, 84, 88, 91, 94, 97, 99, 102, 104, 106, 108, 110, 111, 113, 114, 115, 117, 118, 119, 119, 120, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_285_nl_params, AI_ARRAY_FORMAT_S8,
    nl_285_nl_params_data, nl_285_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_285_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_284_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_285_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_285_layer, 285,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_285_chain,
  NULL, &gemm_286_layer, AI_STATIC, 
  .nl_params = &nl_285_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_284_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_283_output, &gemm_272_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_284_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_284_layer, 284,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_284_chain,
  NULL, &nl_285_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_283_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output2),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_283_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_283_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_283_layer, 283,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_283_chain,
  NULL, &eltwise_284_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_272_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_271_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_272_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_272_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_272_layer, 272,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_272_chain,
  NULL, &gemm_283_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_271_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -123, -123, -123, -122, -121, -121, -120, -119, -119, -118, -117, -116, -115, -113, -112, -111, -109, -107, -105, -103, -101, -99, -96, -94, -91, -88, -85, -81, -77, -74, -69, -65, -61, -56, -51, -46, -40, -35, -29, -24, -18, -12, -6, 0, 6, 12, 18, 24, 29, 35, 40, 46, 51, 56, 61, 65, 69, 74, 77, 81, 85, 88, 91, 94, 96, 99, 101, 103, 105, 107, 109, 111, 112, 113, 115, 116, 117, 118, 119, 119, 120, 121, 121, 122, 123, 123, 123, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_271_nl_params, AI_ARRAY_FORMAT_S8,
    nl_271_nl_params_data, nl_271_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_271_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_270_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_271_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_271_layer, 271,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_271_chain,
  NULL, &gemm_272_layer, AI_STATIC, 
  .nl_params = &nl_271_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_270_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_269_output, &gemm_258_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_270_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_270_layer, 270,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_270_chain,
  NULL, &nl_271_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_269_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_269_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_269_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_269_layer, 269,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_269_chain,
  NULL, &eltwise_270_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_258_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_257_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_258_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_258_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_258_layer, 258,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_258_chain,
  NULL, &gemm_269_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_257_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -121, -120, -120, -119, -118, -118, -117, -116, -116, -115, -114, -113, -112, -111, -110, -109, -108, -106, -105, -104, -102, -101, -99, -97, -95, -93, -91, -89, -87, -85, -82, -80, -77, -74, -71, -68, -65, -62, -59, -55, -52, -48, -45, -41, -37, -33, -29, -25, -21, -17, -13, -8, -4, 0, 4, 8, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48, 52, 55, 59, 62, 65, 68, 71, 74, 77, 80, 82, 85, 87, 89, 91, 93, 95, 97, 99, 101, 102, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 116, 117, 118, 118, 119, 120, 120, 121, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_257_nl_params, AI_ARRAY_FORMAT_S8,
    nl_257_nl_params_data, nl_257_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_257_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_256_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_257_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_257_layer, 257,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_257_chain,
  NULL, &gemm_258_layer, AI_STATIC, 
  .nl_params = &nl_257_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_256_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_output, &gemm_253_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_256_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_256_layer, 256,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_256_chain,
  NULL, &nl_257_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_255_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_254_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_255_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_255_weights, &gemm_255_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_255_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_255_layer, 255,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_255_chain,
  NULL, &eltwise_256_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  unpack_254_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pack_247_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 48, &unpack_254_output0, &unpack_254_output1, &unpack_254_output2, &unpack_254_output3, &unpack_254_output4, &unpack_254_output5, &unpack_254_output6, &unpack_254_output7, &unpack_254_output8, &unpack_254_output9, &unpack_254_output10, &unpack_254_output11, &unpack_254_output12, &unpack_254_output13, &unpack_254_output14, &unpack_254_output15, &unpack_254_output16, &unpack_254_output17, &unpack_254_output18, &unpack_254_output19, &unpack_254_output20, &unpack_254_output21, &unpack_254_output22, &unpack_254_output23, &unpack_254_output24, &unpack_254_output25, &unpack_254_output26, &unpack_254_output27, &unpack_254_output28, &unpack_254_output29, &unpack_254_output30, &unpack_254_output31, &unpack_254_output32, &unpack_254_output33, &unpack_254_output34, &unpack_254_output35, &unpack_254_output36, &unpack_254_output37, &unpack_254_output38, &unpack_254_output39, &unpack_254_output40, &unpack_254_output41, &unpack_254_output42, &unpack_254_output43, &unpack_254_output44, &unpack_254_output45, &unpack_254_output46, &unpack_254_output47),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  unpack_254_layer, 254,
  UNPACK_TYPE, 0x0, NULL,
  unpack, forward_unpack,
  &unpack_254_chain,
  NULL, &gemm_255_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pack_247_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 48, &conversion_11_output, &conversion_26_output, &conversion_41_output, &conversion_56_output, &conversion_71_output, &conversion_75_output, &conversion_79_output, &conversion_83_output, &conversion_87_output, &conversion_91_output, &conversion_95_output, &conversion_99_output, &conversion_103_output, &conversion_107_output, &conversion_111_output, &conversion_115_output, &conversion_119_output, &conversion_123_output, &conversion_127_output, &conversion_131_output, &conversion_135_output, &conversion_139_output, &conversion_143_output, &conversion_147_output, &conversion_151_output, &conversion_155_output, &conversion_159_output, &conversion_163_output, &conversion_167_output, &conversion_171_output, &conversion_175_output, &conversion_179_output, &conversion_183_output, &conversion_187_output, &conversion_191_output, &conversion_195_output, &conversion_199_output, &conversion_203_output, &conversion_207_output, &conversion_211_output, &conversion_215_output, &conversion_219_output, &conversion_223_output, &conversion_227_output, &conversion_231_output, &conversion_236_output, &conversion_241_output, &conversion_246_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pack_247_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pack_247_layer, 247,
  PACK_TYPE, 0x0, NULL,
  pack, forward_pack,
  &pack_247_chain,
  NULL, &unpack_254_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_246_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_245_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_246_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_246_layer, 246,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_246_chain,
  NULL, &pack_247_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_245_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -117, -113, -109, -103, -96, -88, -77, -65, -51, -35, -18, 0, 18, 35, 51, 65, 77, 88, 96, 103, 109, 113, 117, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_245_nl_params, AI_ARRAY_FORMAT_S8,
    nl_245_nl_params_data, nl_245_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_245_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_244_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_245_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_245_layer, 245,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_245_chain,
  NULL, &conversion_246_layer, AI_STATIC, 
  .nl_params = &nl_245_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_244_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_243_output, &gemm_242_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_244_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_244_layer, 244,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_244_chain,
  NULL, &nl_245_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_243_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output47),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_243_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_243_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_243_layer, 243,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_243_chain,
  NULL, &eltwise_244_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_242_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_241_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_242_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_242_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_242_layer, 242,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_242_chain,
  NULL, &gemm_243_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_241_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_240_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_241_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_241_layer, 241,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_241_chain,
  NULL, &gemm_242_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_240_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -122, -120, -117, -114, -109, -104, -97, -88, -78, -66, -51, -35, -18, 0, 18, 35, 51, 66, 78, 88, 97, 104, 109, 114, 117, 120, 122, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_240_nl_params, AI_ARRAY_FORMAT_S8,
    nl_240_nl_params_data, nl_240_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_240_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_239_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_240_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_240_layer, 240,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_240_chain,
  NULL, &conversion_241_layer, AI_STATIC, 
  .nl_params = &nl_240_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_239_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_238_output, &gemm_237_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_239_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_239_layer, 239,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_239_chain,
  NULL, &nl_240_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_238_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output46),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_238_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_238_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_238_layer, 238,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_238_chain,
  NULL, &eltwise_239_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_237_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_236_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_237_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_237_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_237_layer, 237,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_237_chain,
  NULL, &gemm_238_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_236_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_235_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_236_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_236_layer, 236,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_236_chain,
  NULL, &gemm_237_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_235_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -120, -118, -115, -110, -105, -98, -90, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 90, 98, 105, 110, 115, 118, 120, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_235_nl_params, AI_ARRAY_FORMAT_S8,
    nl_235_nl_params_data, nl_235_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_235_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_234_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_235_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_235_layer, 235,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_235_chain,
  NULL, &conversion_236_layer, AI_STATIC, 
  .nl_params = &nl_235_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_234_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_233_output, &gemm_232_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_234_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_234_layer, 234,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_234_chain,
  NULL, &nl_235_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_233_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output45),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_233_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_233_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_233_layer, 233,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_233_chain,
  NULL, &eltwise_234_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_232_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_231_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_232_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_232_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_232_layer, 232,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_232_chain,
  NULL, &gemm_233_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_231_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_230_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_231_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_231_layer, 231,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_231_chain,
  NULL, &gemm_232_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_230_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -124, -122, -120, -118, -114, -110, -105, -98, -89, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 89, 98, 105, 110, 114, 118, 120, 122, 124, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_230_nl_params, AI_ARRAY_FORMAT_S8,
    nl_230_nl_params_data, nl_230_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_230_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_229_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_230_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_230_layer, 230,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_230_chain,
  NULL, &conversion_231_layer, AI_STATIC, 
  .nl_params = &nl_230_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_229_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_67_output, &gemm_228_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_229_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_229_layer, 229,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_229_chain,
  NULL, &nl_230_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_67_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output44),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_67_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_67_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_67_layer, 67,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_67_chain,
  NULL, &eltwise_229_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_228_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_227_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_228_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_228_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_228_layer, 228,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_228_chain,
  NULL, &gemm_67_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_227_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_226_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_227_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_227_layer, 227,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_227_chain,
  NULL, &gemm_228_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_226_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -123, -122, -120, -117, -113, -108, -102, -93, -83, -70, -55, -38, -20, 0, 20, 38, 55, 70, 83, 93, 102, 108, 113, 117, 120, 122, 123, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_226_nl_params, AI_ARRAY_FORMAT_S8,
    nl_226_nl_params_data, nl_226_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_226_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_225_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_226_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_226_layer, 226,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_226_chain,
  NULL, &conversion_227_layer, AI_STATIC, 
  .nl_params = &nl_226_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_225_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_66_output, &gemm_224_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_225_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_225_layer, 225,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_225_chain,
  NULL, &nl_226_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_66_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output43),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_66_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_66_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_66_layer, 66,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_66_chain,
  NULL, &eltwise_225_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_224_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_223_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_224_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_224_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_224_layer, 224,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_224_chain,
  NULL, &gemm_66_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_223_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_222_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_223_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_223_layer, 223,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_223_chain,
  NULL, &gemm_224_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_222_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -123, -122, -119, -117, -113, -108, -101, -93, -82, -70, -55, -38, -19, 0, 19, 38, 55, 70, 82, 93, 101, 108, 113, 117, 119, 122, 123, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_222_nl_params, AI_ARRAY_FORMAT_S8,
    nl_222_nl_params_data, nl_222_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_222_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_221_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_222_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_222_layer, 222,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_222_chain,
  NULL, &conversion_223_layer, AI_STATIC, 
  .nl_params = &nl_222_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_221_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_65_output, &gemm_220_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_221_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_221_layer, 221,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_221_chain,
  NULL, &nl_222_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output42),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_65_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_65_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_65_layer, 65,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_65_chain,
  NULL, &eltwise_221_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_220_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_219_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_220_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_220_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_220_layer, 220,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_220_chain,
  NULL, &gemm_65_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_219_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_218_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_219_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_219_layer, 219,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_219_chain,
  NULL, &gemm_220_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_218_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -118, -115, -111, -106, -99, -91, -80, -68, -53, -37, -19, 0, 19, 37, 53, 68, 80, 91, 99, 106, 111, 115, 118, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_218_nl_params, AI_ARRAY_FORMAT_S8,
    nl_218_nl_params_data, nl_218_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_218_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_217_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_218_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_218_layer, 218,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_218_chain,
  NULL, &conversion_219_layer, AI_STATIC, 
  .nl_params = &nl_218_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_217_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_64_output, &gemm_216_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_217_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_217_layer, 217,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_217_chain,
  NULL, &nl_218_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_64_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output41),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_64_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_64_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_64_layer, 64,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_64_chain,
  NULL, &eltwise_217_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_216_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_215_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_216_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_216_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_216_layer, 216,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_216_chain,
  NULL, &gemm_64_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_215_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_214_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_215_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_215_layer, 215,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_215_chain,
  NULL, &gemm_216_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_214_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -118, -115, -111, -106, -99, -91, -80, -68, -53, -37, -19, 0, 19, 37, 53, 68, 80, 91, 99, 106, 111, 115, 118, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_214_nl_params, AI_ARRAY_FORMAT_S8,
    nl_214_nl_params_data, nl_214_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_214_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_213_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_214_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_214_layer, 214,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_214_chain,
  NULL, &conversion_215_layer, AI_STATIC, 
  .nl_params = &nl_214_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_213_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_63_output, &gemm_212_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_213_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_213_layer, 213,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_213_chain,
  NULL, &nl_214_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_63_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output40),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_63_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_63_layer, 63,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_63_chain,
  NULL, &eltwise_213_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_212_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_211_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_212_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_212_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_212_layer, 212,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_212_chain,
  NULL, &gemm_63_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_211_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_210_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_211_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_211_layer, 211,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_211_chain,
  NULL, &gemm_212_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_210_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -115, -111, -106, -99, -91, -81, -68, -53, -37, -19, 0, 19, 37, 53, 68, 81, 91, 99, 106, 111, 115, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_210_nl_params, AI_ARRAY_FORMAT_S8,
    nl_210_nl_params_data, nl_210_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_210_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_209_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_210_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_210_layer, 210,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_210_chain,
  NULL, &conversion_211_layer, AI_STATIC, 
  .nl_params = &nl_210_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_209_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_62_output, &gemm_208_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_209_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_209_layer, 209,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_209_chain,
  NULL, &nl_210_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_62_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output39),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_62_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_62_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_62_layer, 62,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_62_chain,
  NULL, &eltwise_209_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_208_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_207_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_208_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_208_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_208_layer, 208,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_208_chain,
  NULL, &gemm_62_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_207_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_206_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_207_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_207_layer, 207,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_207_chain,
  NULL, &gemm_208_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_206_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -121, -118, -115, -111, -106, -99, -90, -80, -67, -53, -36, -19, 0, 19, 36, 53, 67, 80, 90, 99, 106, 111, 115, 118, 121, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_206_nl_params, AI_ARRAY_FORMAT_S8,
    nl_206_nl_params_data, nl_206_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_206_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_205_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_206_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_206_layer, 206,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_206_chain,
  NULL, &conversion_207_layer, AI_STATIC, 
  .nl_params = &nl_206_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_205_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_61_output, &gemm_204_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_205_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_205_layer, 205,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_205_chain,
  NULL, &nl_206_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_61_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output38),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_61_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_61_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_61_layer, 61,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_61_chain,
  NULL, &eltwise_205_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_204_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_203_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_204_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_204_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_204_layer, 204,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_204_chain,
  NULL, &gemm_61_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_203_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_202_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_203_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_203_layer, 203,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_203_chain,
  NULL, &gemm_204_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_202_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -116, -112, -107, -100, -92, -81, -69, -54, -37, -19, 0, 19, 37, 54, 69, 81, 92, 100, 107, 112, 116, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_202_nl_params, AI_ARRAY_FORMAT_S8,
    nl_202_nl_params_data, nl_202_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_202_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_201_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_202_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_202_layer, 202,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_202_chain,
  NULL, &conversion_203_layer, AI_STATIC, 
  .nl_params = &nl_202_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_201_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_60_output, &gemm_200_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_201_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_201_layer, 201,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_201_chain,
  NULL, &nl_202_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_60_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output37),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_60_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_60_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_60_layer, 60,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_60_chain,
  NULL, &eltwise_201_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_200_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_199_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_200_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_200_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_200_layer, 200,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_200_chain,
  NULL, &gemm_60_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_199_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_198_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_199_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_199_layer, 199,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_199_chain,
  NULL, &gemm_200_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_198_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -118, -115, -111, -106, -99, -91, -80, -68, -53, -37, -19, 0, 19, 37, 53, 68, 80, 91, 99, 106, 111, 115, 118, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_198_nl_params, AI_ARRAY_FORMAT_S8,
    nl_198_nl_params_data, nl_198_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_198_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_197_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_198_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_198_layer, 198,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_198_chain,
  NULL, &conversion_199_layer, AI_STATIC, 
  .nl_params = &nl_198_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_197_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_59_output, &gemm_196_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_197_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_197_layer, 197,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_197_chain,
  NULL, &nl_198_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_59_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output36),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_59_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_59_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_59_layer, 59,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_59_chain,
  NULL, &eltwise_197_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_196_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_195_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_196_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_196_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_196_layer, 196,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_196_chain,
  NULL, &gemm_59_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_195_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_194_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_195_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_195_layer, 195,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_195_chain,
  NULL, &gemm_196_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_194_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -121, -118, -115, -111, -105, -99, -90, -80, -67, -53, -36, -19, 0, 19, 36, 53, 67, 80, 90, 99, 105, 111, 115, 118, 121, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_194_nl_params, AI_ARRAY_FORMAT_S8,
    nl_194_nl_params_data, nl_194_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_194_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_193_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_194_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_194_layer, 194,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_194_chain,
  NULL, &conversion_195_layer, AI_STATIC, 
  .nl_params = &nl_194_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_193_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_58_output, &gemm_192_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_193_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_193_layer, 193,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_193_chain,
  NULL, &nl_194_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output35),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_58_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_58_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_58_layer, 58,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_58_chain,
  NULL, &eltwise_193_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_192_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_191_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_192_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_192_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_192_layer, 192,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_192_chain,
  NULL, &gemm_58_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_191_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_190_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_191_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_191_layer, 191,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_191_chain,
  NULL, &gemm_192_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_190_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -124, -122, -120, -118, -114, -110, -105, -98, -89, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 89, 98, 105, 110, 114, 118, 120, 122, 124, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_190_nl_params, AI_ARRAY_FORMAT_S8,
    nl_190_nl_params_data, nl_190_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_190_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_189_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_190_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_190_layer, 190,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_190_chain,
  NULL, &conversion_191_layer, AI_STATIC, 
  .nl_params = &nl_190_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_189_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_52_output, &gemm_188_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_189_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_189_layer, 189,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_189_chain,
  NULL, &nl_190_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output34),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_52_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_52_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_52_layer, 52,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_52_chain,
  NULL, &eltwise_189_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_188_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_187_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_188_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_188_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_188_layer, 188,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_188_chain,
  NULL, &gemm_52_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_187_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_186_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_187_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_187_layer, 187,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_187_chain,
  NULL, &gemm_188_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_186_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -123, -122, -120, -117, -113, -108, -102, -93, -83, -70, -55, -38, -20, 0, 20, 38, 55, 70, 83, 93, 102, 108, 113, 117, 120, 122, 123, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_186_nl_params, AI_ARRAY_FORMAT_S8,
    nl_186_nl_params_data, nl_186_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_186_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_185_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_186_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_186_layer, 186,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_186_chain,
  NULL, &conversion_187_layer, AI_STATIC, 
  .nl_params = &nl_186_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_185_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_51_output, &gemm_184_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_185_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_185_layer, 185,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_185_chain,
  NULL, &nl_186_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output33),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_51_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_51_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_51_layer, 51,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_51_chain,
  NULL, &eltwise_185_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_184_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_183_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_184_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_184_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_184_layer, 184,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_184_chain,
  NULL, &gemm_51_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_183_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_182_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_183_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_183_layer, 183,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_183_chain,
  NULL, &gemm_184_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_182_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -123, -122, -119, -117, -113, -108, -101, -93, -82, -70, -55, -38, -19, 0, 19, 38, 55, 70, 82, 93, 101, 108, 113, 117, 119, 122, 123, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_182_nl_params, AI_ARRAY_FORMAT_S8,
    nl_182_nl_params_data, nl_182_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_182_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_181_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_182_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_182_layer, 182,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_182_chain,
  NULL, &conversion_183_layer, AI_STATIC, 
  .nl_params = &nl_182_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_181_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_50_output, &gemm_180_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_181_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_181_layer, 181,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_181_chain,
  NULL, &nl_182_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output32),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_50_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_50_layer, 50,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_50_chain,
  NULL, &eltwise_181_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_180_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_179_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_180_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_180_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_180_layer, 180,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_180_chain,
  NULL, &gemm_50_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_179_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_178_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_179_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_179_layer, 179,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_179_chain,
  NULL, &gemm_180_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_178_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -120, -118, -115, -110, -105, -98, -90, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 90, 98, 105, 110, 115, 118, 120, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_178_nl_params, AI_ARRAY_FORMAT_S8,
    nl_178_nl_params_data, nl_178_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_178_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_177_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_178_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_178_layer, 178,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_178_chain,
  NULL, &conversion_179_layer, AI_STATIC, 
  .nl_params = &nl_178_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_177_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_49_output, &gemm_176_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_177_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_177_layer, 177,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_177_chain,
  NULL, &nl_178_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output31),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_49_layer, 49,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_49_chain,
  NULL, &eltwise_177_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_176_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_175_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_176_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_176_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_176_layer, 176,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_176_chain,
  NULL, &gemm_49_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_175_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_174_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_175_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_175_layer, 175,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_175_chain,
  NULL, &gemm_176_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_174_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -124, -122, -120, -118, -114, -110, -105, -98, -89, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 89, 98, 105, 110, 114, 118, 120, 122, 124, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_174_nl_params, AI_ARRAY_FORMAT_S8,
    nl_174_nl_params_data, nl_174_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_174_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_173_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_174_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_174_layer, 174,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_174_chain,
  NULL, &conversion_175_layer, AI_STATIC, 
  .nl_params = &nl_174_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_173_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_48_output, &gemm_172_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_173_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_173_layer, 173,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_173_chain,
  NULL, &nl_174_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output30),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_48_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_48_layer, 48,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_48_chain,
  NULL, &eltwise_173_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_172_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_171_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_172_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_172_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_172_layer, 172,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_172_chain,
  NULL, &gemm_48_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_171_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_170_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_171_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_171_layer, 171,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_171_chain,
  NULL, &gemm_172_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_170_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -126, -126, -125, -124, -123, -122, -120, -117, -114, -110, -104, -97, -89, -78, -66, -51, -35, -18, 0, 18, 35, 51, 66, 78, 89, 97, 104, 110, 114, 117, 120, 122, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_170_nl_params, AI_ARRAY_FORMAT_S8,
    nl_170_nl_params_data, nl_170_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_170_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_169_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_170_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_170_layer, 170,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_170_chain,
  NULL, &conversion_171_layer, AI_STATIC, 
  .nl_params = &nl_170_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_169_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_47_output, &gemm_168_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_169_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_169_layer, 169,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_169_chain,
  NULL, &nl_170_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output29),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_47_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_47_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_47_layer, 47,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_47_chain,
  NULL, &eltwise_169_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_168_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_167_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_168_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_168_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_168_layer, 168,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_168_chain,
  NULL, &gemm_47_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_167_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_166_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_167_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_167_layer, 167,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_167_chain,
  NULL, &gemm_168_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_166_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -116, -113, -108, -103, -95, -87, -76, -64, -50, -34, -18, 0, 18, 34, 50, 64, 76, 87, 95, 103, 108, 113, 116, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_166_nl_params, AI_ARRAY_FORMAT_S8,
    nl_166_nl_params_data, nl_166_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_166_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_165_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_166_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_166_layer, 166,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_166_chain,
  NULL, &conversion_167_layer, AI_STATIC, 
  .nl_params = &nl_166_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_165_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_46_output, &gemm_164_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_165_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_165_layer, 165,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_165_chain,
  NULL, &nl_166_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output28),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_46_layer, 46,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_46_chain,
  NULL, &eltwise_165_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_164_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_163_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_164_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_164_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_164_layer, 164,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_164_chain,
  NULL, &gemm_46_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_163_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_162_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_163_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_163_layer, 163,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_163_chain,
  NULL, &gemm_164_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_162_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -125, -124, -122, -121, -118, -116, -112, -107, -102, -94, -86, -75, -63, -49, -34, -17, 0, 17, 34, 49, 63, 75, 86, 94, 102, 107, 112, 116, 118, 121, 122, 124, 125, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_162_nl_params, AI_ARRAY_FORMAT_S8,
    nl_162_nl_params_data, nl_162_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_162_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_161_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_162_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_162_layer, 162,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_162_chain,
  NULL, &conversion_163_layer, AI_STATIC, 
  .nl_params = &nl_162_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_161_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_45_output, &gemm_160_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_161_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_161_layer, 161,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_161_chain,
  NULL, &nl_162_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output27),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_45_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_45_layer, 45,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_45_chain,
  NULL, &eltwise_161_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_160_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_159_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_160_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_160_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_160_layer, 160,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_160_chain,
  NULL, &gemm_45_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_159_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_158_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_159_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_159_layer, 159,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_159_chain,
  NULL, &gemm_160_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_158_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -122, -120, -117, -113, -109, -104, -97, -88, -78, -65, -51, -35, -18, 0, 18, 35, 51, 65, 78, 88, 97, 104, 109, 113, 117, 120, 122, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_158_nl_params, AI_ARRAY_FORMAT_S8,
    nl_158_nl_params_data, nl_158_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_158_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_157_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_158_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_158_layer, 158,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_158_chain,
  NULL, &conversion_159_layer, AI_STATIC, 
  .nl_params = &nl_158_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_157_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_44_output, &gemm_156_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_157_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_157_layer, 157,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_157_chain,
  NULL, &nl_158_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_44_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output26),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_44_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_44_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_44_layer, 44,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_44_chain,
  NULL, &eltwise_157_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_156_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_155_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_156_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_156_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_156_layer, 156,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_156_chain,
  NULL, &gemm_44_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_155_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_154_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_155_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_155_layer, 155,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_155_chain,
  NULL, &gemm_156_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_154_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -117, -113, -109, -103, -96, -88, -77, -65, -51, -35, -18, 0, 18, 35, 51, 65, 77, 88, 96, 103, 109, 113, 117, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_154_nl_params, AI_ARRAY_FORMAT_S8,
    nl_154_nl_params_data, nl_154_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_154_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_153_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_154_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_154_layer, 154,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_154_chain,
  NULL, &conversion_155_layer, AI_STATIC, 
  .nl_params = &nl_154_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_153_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_43_output, &gemm_152_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_153_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_153_layer, 153,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_153_chain,
  NULL, &nl_154_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output25),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_43_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_43_layer, 43,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_43_chain,
  NULL, &eltwise_153_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_152_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_151_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_152_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_152_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_152_layer, 152,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_152_chain,
  NULL, &gemm_43_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_151_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_150_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_151_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_151_layer, 151,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_151_chain,
  NULL, &gemm_152_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_150_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -120, -118, -115, -110, -105, -98, -90, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 90, 98, 105, 110, 115, 118, 120, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_150_nl_params, AI_ARRAY_FORMAT_S8,
    nl_150_nl_params_data, nl_150_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_150_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_149_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_150_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_150_layer, 150,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_150_chain,
  NULL, &conversion_151_layer, AI_STATIC, 
  .nl_params = &nl_150_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_149_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_37_output, &gemm_148_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_149_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_149_layer, 149,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_149_chain,
  NULL, &nl_150_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output24),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_37_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_37_layer, 37,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_37_chain,
  NULL, &eltwise_149_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_148_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_147_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_148_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_148_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_148_layer, 148,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_148_chain,
  NULL, &gemm_37_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_147_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_146_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_147_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_147_layer, 147,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_147_chain,
  NULL, &gemm_148_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_146_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -117, -113, -109, -103, -96, -88, -77, -65, -51, -35, -18, 0, 18, 35, 51, 65, 77, 88, 96, 103, 109, 113, 117, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_146_nl_params, AI_ARRAY_FORMAT_S8,
    nl_146_nl_params_data, nl_146_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_146_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_145_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_146_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_146_layer, 146,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_146_chain,
  NULL, &conversion_147_layer, AI_STATIC, 
  .nl_params = &nl_146_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_145_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_36_output, &gemm_144_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_145_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_145_layer, 145,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_145_chain,
  NULL, &nl_146_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output23),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_36_layer, 36,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_36_chain,
  NULL, &eltwise_145_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_144_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_143_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_144_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_144_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_144_layer, 144,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_144_chain,
  NULL, &gemm_36_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_143_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_142_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_143_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_143_layer, 143,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_143_chain,
  NULL, &gemm_144_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_142_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -126, -126, -125, -124, -123, -122, -120, -117, -114, -110, -104, -97, -89, -78, -66, -51, -35, -18, 0, 18, 35, 51, 66, 78, 89, 97, 104, 110, 114, 117, 120, 122, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_142_nl_params, AI_ARRAY_FORMAT_S8,
    nl_142_nl_params_data, nl_142_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_142_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_141_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_142_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_142_layer, 142,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_142_chain,
  NULL, &conversion_143_layer, AI_STATIC, 
  .nl_params = &nl_142_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_141_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_35_output, &gemm_140_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_141_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_141_layer, 141,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_141_chain,
  NULL, &nl_142_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output22),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_35_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_35_layer, 35,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_35_chain,
  NULL, &eltwise_141_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_140_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_139_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_140_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_140_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_140_layer, 140,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_140_chain,
  NULL, &gemm_35_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_139_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_138_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_139_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_139_layer, 139,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_139_chain,
  NULL, &gemm_140_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_138_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -116, -112, -108, -102, -95, -86, -76, -64, -50, -34, -17, 0, 17, 34, 50, 64, 76, 86, 95, 102, 108, 112, 116, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_138_nl_params, AI_ARRAY_FORMAT_S8,
    nl_138_nl_params_data, nl_138_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_138_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_137_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_138_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_138_layer, 138,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_138_chain,
  NULL, &conversion_139_layer, AI_STATIC, 
  .nl_params = &nl_138_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_137_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_34_output, &gemm_136_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_137_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_137_layer, 137,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_137_chain,
  NULL, &nl_138_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output21),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_34_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_34_layer, 34,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_34_chain,
  NULL, &eltwise_137_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_136_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_135_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_136_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_136_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_136_layer, 136,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_136_chain,
  NULL, &gemm_34_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_135_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_134_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_135_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_135_layer, 135,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_135_chain,
  NULL, &gemm_136_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_134_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -125, -123, -122, -120, -118, -115, -112, -107, -101, -94, -85, -75, -63, -49, -33, -17, 0, 17, 33, 49, 63, 75, 85, 94, 101, 107, 112, 115, 118, 120, 122, 123, 125, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_134_nl_params, AI_ARRAY_FORMAT_S8,
    nl_134_nl_params_data, nl_134_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_134_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_133_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_134_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_134_layer, 134,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_134_chain,
  NULL, &conversion_135_layer, AI_STATIC, 
  .nl_params = &nl_134_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_133_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_33_output, &gemm_132_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_133_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_133_layer, 133,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_133_chain,
  NULL, &nl_134_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output20),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_33_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_33_layer, 33,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_33_chain,
  NULL, &eltwise_133_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_132_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_131_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_132_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_132_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_132_layer, 132,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_132_chain,
  NULL, &gemm_33_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_131_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_130_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_131_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_131_layer, 131,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_131_chain,
  NULL, &gemm_132_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_130_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -116, -113, -108, -103, -96, -87, -77, -64, -50, -35, -18, 0, 18, 35, 50, 64, 77, 87, 96, 103, 108, 113, 116, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_130_nl_params, AI_ARRAY_FORMAT_S8,
    nl_130_nl_params_data, nl_130_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_130_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_129_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_130_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_130_layer, 130,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_130_chain,
  NULL, &conversion_131_layer, AI_STATIC, 
  .nl_params = &nl_130_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_129_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_32_output, &gemm_128_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_129_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_129_layer, 129,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_129_chain,
  NULL, &nl_130_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output19),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_32_layer, 32,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_32_chain,
  NULL, &eltwise_129_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_128_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_127_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_128_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_128_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_128_layer, 128,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_128_chain,
  NULL, &gemm_32_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_127_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_126_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_127_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_127_layer, 127,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_127_chain,
  NULL, &gemm_128_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_126_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -118, -115, -111, -106, -99, -91, -80, -68, -53, -37, -19, 0, 19, 37, 53, 68, 80, 91, 99, 106, 111, 115, 118, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_126_nl_params, AI_ARRAY_FORMAT_S8,
    nl_126_nl_params_data, nl_126_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_126_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_125_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_126_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_126_layer, 126,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_126_chain,
  NULL, &conversion_127_layer, AI_STATIC, 
  .nl_params = &nl_126_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_125_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_31_output, &gemm_124_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_125_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_125_layer, 125,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_125_chain,
  NULL, &nl_126_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output18),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_31_layer, 31,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_31_chain,
  NULL, &eltwise_125_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_124_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_123_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_124_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_124_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_124_layer, 124,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_124_chain,
  NULL, &gemm_31_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_123_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_122_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_123_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_123_layer, 123,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_123_chain,
  NULL, &gemm_124_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_122_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -120, -118, -115, -111, -105, -99, -90, -80, -67, -53, -36, -19, 0, 19, 36, 53, 67, 80, 90, 99, 105, 111, 115, 118, 120, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_122_nl_params, AI_ARRAY_FORMAT_S8,
    nl_122_nl_params_data, nl_122_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_122_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_121_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_122_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_122_layer, 122,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_122_chain,
  NULL, &conversion_123_layer, AI_STATIC, 
  .nl_params = &nl_122_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_121_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_30_output, &gemm_120_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_121_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_121_layer, 121,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_121_chain,
  NULL, &nl_122_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output17),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_30_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_30_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_30_layer, 30,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_30_chain,
  NULL, &eltwise_121_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_120_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_119_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_120_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_120_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_120_layer, 120,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_120_chain,
  NULL, &gemm_30_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_119_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_118_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_119_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_119_layer, 119,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_119_chain,
  NULL, &gemm_120_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_118_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -123, -122, -120, -117, -114, -110, -105, -98, -89, -79, -66, -52, -36, -18, 0, 18, 36, 52, 66, 79, 89, 98, 105, 110, 114, 117, 120, 122, 123, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_118_nl_params, AI_ARRAY_FORMAT_S8,
    nl_118_nl_params_data, nl_118_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_118_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_117_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_118_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_118_layer, 118,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_118_chain,
  NULL, &conversion_119_layer, AI_STATIC, 
  .nl_params = &nl_118_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_117_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_29_output, &gemm_116_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_117_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_117_layer, 117,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_117_chain,
  NULL, &nl_118_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output16),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_29_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_29_layer, 29,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_29_chain,
  NULL, &eltwise_117_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_116_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_115_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_116_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_116_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_116_layer, 116,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_116_chain,
  NULL, &gemm_29_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_115_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_114_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_115_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_115_layer, 115,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_115_chain,
  NULL, &gemm_116_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_114_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -120, -118, -115, -110, -105, -98, -90, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 90, 98, 105, 110, 115, 118, 120, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_114_nl_params, AI_ARRAY_FORMAT_S8,
    nl_114_nl_params_data, nl_114_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_114_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_113_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_114_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_114_layer, 114,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_114_chain,
  NULL, &conversion_115_layer, AI_STATIC, 
  .nl_params = &nl_114_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_113_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_28_output, &gemm_112_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_113_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_113_layer, 113,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_113_chain,
  NULL, &nl_114_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output15),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_28_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_28_layer, 28,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_28_chain,
  NULL, &eltwise_113_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_112_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_111_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_112_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_112_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_112_layer, 112,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_112_chain,
  NULL, &gemm_28_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_111_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_110_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_111_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_111_layer, 111,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_111_chain,
  NULL, &gemm_112_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_110_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -120, -118, -115, -111, -105, -98, -90, -80, -67, -53, -36, -18, 0, 18, 36, 53, 67, 80, 90, 98, 105, 111, 115, 118, 120, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_110_nl_params, AI_ARRAY_FORMAT_S8,
    nl_110_nl_params_data, nl_110_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_110_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_109_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_110_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_110_layer, 110,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_110_chain,
  NULL, &conversion_111_layer, AI_STATIC, 
  .nl_params = &nl_110_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_109_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_22_output, &gemm_108_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_109_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_109_layer, 109,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_109_chain,
  NULL, &nl_110_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output14),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_22_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_22_layer, 22,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_22_chain,
  NULL, &eltwise_109_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_108_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_107_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_108_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_108_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_108_layer, 108,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_108_chain,
  NULL, &gemm_22_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_107_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_106_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_107_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_107_layer, 107,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_107_chain,
  NULL, &gemm_108_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_106_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -121, -118, -115, -111, -106, -99, -90, -80, -67, -53, -36, -19, 0, 19, 36, 53, 67, 80, 90, 99, 106, 111, 115, 118, 121, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_106_nl_params, AI_ARRAY_FORMAT_S8,
    nl_106_nl_params_data, nl_106_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_106_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_105_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_106_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_106_layer, 106,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_106_chain,
  NULL, &conversion_107_layer, AI_STATIC, 
  .nl_params = &nl_106_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_105_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_21_output, &gemm_104_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_105_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_105_layer, 105,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_105_chain,
  NULL, &nl_106_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output13),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_21_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_21_layer, 21,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_21_chain,
  NULL, &eltwise_105_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_104_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_103_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_104_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_104_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_104_layer, 104,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_104_chain,
  NULL, &gemm_21_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_103_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_102_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_103_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_103_layer, 103,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_103_chain,
  NULL, &gemm_104_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_102_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -121, -118, -115, -111, -105, -99, -90, -80, -67, -53, -36, -19, 0, 19, 36, 53, 67, 80, 90, 99, 105, 111, 115, 118, 121, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_102_nl_params, AI_ARRAY_FORMAT_S8,
    nl_102_nl_params_data, nl_102_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_102_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_101_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_102_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_102_layer, 102,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_102_chain,
  NULL, &conversion_103_layer, AI_STATIC, 
  .nl_params = &nl_102_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_101_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_20_output, &gemm_100_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_101_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_101_layer, 101,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_101_chain,
  NULL, &nl_102_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output12),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_20_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_20_layer, 20,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_20_chain,
  NULL, &eltwise_101_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_100_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_99_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_100_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_100_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_100_layer, 100,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_100_chain,
  NULL, &gemm_20_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_99_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_98_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_99_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_99_layer, 99,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_99_chain,
  NULL, &gemm_100_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_98_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -122, -119, -117, -113, -109, -104, -97, -88, -78, -65, -51, -35, -18, 0, 18, 35, 51, 65, 78, 88, 97, 104, 109, 113, 117, 119, 122, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_98_nl_params, AI_ARRAY_FORMAT_S8,
    nl_98_nl_params_data, nl_98_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_98_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_97_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_98_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_98_layer, 98,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_98_chain,
  NULL, &conversion_99_layer, AI_STATIC, 
  .nl_params = &nl_98_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_97_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_19_output, &gemm_96_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_97_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_97_layer, 97,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_97_chain,
  NULL, &nl_98_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output11),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_19_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_19_layer, 19,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_19_chain,
  NULL, &eltwise_97_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_96_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_95_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_96_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_96_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_96_layer, 96,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_96_chain,
  NULL, &gemm_19_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_95_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_94_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_95_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_95_layer, 95,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_95_chain,
  NULL, &gemm_96_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_94_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -118, -115, -111, -106, -99, -91, -81, -68, -53, -37, -19, 0, 19, 37, 53, 68, 81, 91, 99, 106, 111, 115, 118, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_94_nl_params, AI_ARRAY_FORMAT_S8,
    nl_94_nl_params_data, nl_94_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_94_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_94_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_94_layer, 94,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_94_chain,
  NULL, &conversion_95_layer, AI_STATIC, 
  .nl_params = &nl_94_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_93_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_18_output, &gemm_92_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_93_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_93_layer, 93,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_93_chain,
  NULL, &nl_94_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output10),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_18_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_18_layer, 18,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_18_chain,
  NULL, &eltwise_93_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_92_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_91_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_92_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_92_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_92_layer, 92,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_92_chain,
  NULL, &gemm_18_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_91_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_90_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_91_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_91_layer, 91,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_91_chain,
  NULL, &gemm_92_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_90_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -126, -126, -125, -124, -123, -121, -119, -116, -112, -107, -100, -92, -81, -69, -54, -37, -19, 0, 19, 37, 54, 69, 81, 92, 100, 107, 112, 116, 119, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_90_nl_params, AI_ARRAY_FORMAT_S8,
    nl_90_nl_params_data, nl_90_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_90_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_90_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_90_layer, 90,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_90_chain,
  NULL, &conversion_91_layer, AI_STATIC, 
  .nl_params = &nl_90_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_89_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_17_output, &gemm_88_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_89_layer, 89,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_89_chain,
  NULL, &nl_90_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output9),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_17_layer, 17,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_17_chain,
  NULL, &eltwise_89_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_88_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_87_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_88_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_88_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_88_layer, 88,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_88_chain,
  NULL, &gemm_17_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_87_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_86_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_87_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_87_layer, 87,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_87_chain,
  NULL, &gemm_88_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_86_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -118, -115, -111, -106, -99, -91, -80, -68, -53, -37, -19, 0, 19, 37, 53, 68, 80, 91, 99, 106, 111, 115, 118, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_86_nl_params, AI_ARRAY_FORMAT_S8,
    nl_86_nl_params_data, nl_86_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_86_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_86_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_86_layer, 86,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_86_chain,
  NULL, &conversion_87_layer, AI_STATIC, 
  .nl_params = &nl_86_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_85_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_16_output, &gemm_84_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_85_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_85_layer, 85,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_85_chain,
  NULL, &nl_86_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output8),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_16_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_16_chain,
  NULL, &eltwise_85_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_84_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_83_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_84_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_84_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_84_layer, 84,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_84_chain,
  NULL, &gemm_16_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_82_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_83_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_83_layer, 83,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_83_chain,
  NULL, &gemm_84_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_82_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -124, -122, -120, -118, -114, -110, -105, -98, -89, -79, -66, -52, -36, -18, 0, 18, 36, 52, 66, 79, 89, 98, 105, 110, 114, 118, 120, 122, 124, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_82_nl_params, AI_ARRAY_FORMAT_S8,
    nl_82_nl_params_data, nl_82_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_82_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_82_layer, 82,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_82_chain,
  NULL, &conversion_83_layer, AI_STATIC, 
  .nl_params = &nl_82_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_15_output, &gemm_80_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_81_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_81_layer, 81,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_81_chain,
  NULL, &nl_82_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output7),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_15_layer, 15,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_15_chain,
  NULL, &eltwise_81_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_80_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_80_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_80_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_80_layer, 80,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_80_chain,
  NULL, &gemm_15_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_78_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_79_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_79_layer, 79,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_79_chain,
  NULL, &gemm_80_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_78_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -121, -118, -115, -111, -106, -99, -91, -80, -68, -53, -37, -19, 0, 19, 37, 53, 68, 80, 91, 99, 106, 111, 115, 118, 121, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_78_nl_params, AI_ARRAY_FORMAT_S8,
    nl_78_nl_params_data, nl_78_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_78_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_77_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_78_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_78_layer, 78,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_78_chain,
  NULL, &conversion_79_layer, AI_STATIC, 
  .nl_params = &nl_78_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_77_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_14_output, &gemm_76_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_77_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_77_layer, 77,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_77_chain,
  NULL, &nl_78_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output6),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_14_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_14_chain,
  NULL, &eltwise_77_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_76_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_76_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_76_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_76_layer, 76,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_76_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_75_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_74_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_75_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_75_layer, 75,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_75_chain,
  NULL, &gemm_76_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_74_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -122, -120, -118, -115, -110, -105, -98, -90, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 90, 98, 105, 110, 115, 118, 120, 122, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_74_nl_params, AI_ARRAY_FORMAT_S8,
    nl_74_nl_params_data, nl_74_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_74_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_73_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_74_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_74_layer, 74,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_74_chain,
  NULL, &conversion_75_layer, AI_STATIC, 
  .nl_params = &nl_74_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_73_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_13_output, &gemm_72_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_73_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_73_layer, 73,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_73_chain,
  NULL, &nl_74_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output5),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_13_layer, 13,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_13_chain,
  NULL, &eltwise_73_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_72_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_71_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_72_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_72_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_72_layer, 72,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_72_chain,
  NULL, &gemm_13_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_71_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_70_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_71_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_71_layer, 71,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_71_chain,
  NULL, &gemm_72_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_70_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -124, -123, -122, -120, -117, -114, -109, -104, -97, -88, -78, -66, -51, -35, -18, 0, 18, 35, 51, 66, 78, 88, 97, 104, 109, 114, 117, 120, 122, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_70_nl_params, AI_ARRAY_FORMAT_S8,
    nl_70_nl_params_data, nl_70_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_70_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_70_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_70_layer, 70,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_70_chain,
  NULL, &conversion_71_layer, AI_STATIC, 
  .nl_params = &nl_70_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_69_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_68_output, &gemm_57_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_69_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_69_layer, 69,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_69_chain,
  NULL, &nl_70_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_68_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output4),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_68_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_68_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_68_layer, 68,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_68_chain,
  NULL, &eltwise_69_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_57_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_57_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_57_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_57_layer, 57,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_57_chain,
  NULL, &gemm_68_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_56_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_56_layer, 56,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_56_chain,
  NULL, &gemm_57_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_55_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -126, -126, -125, -124, -123, -122, -120, -117, -114, -110, -104, -97, -89, -78, -66, -51, -35, -18, 0, 18, 35, 51, 66, 78, 89, 97, 104, 110, 114, 117, 120, 122, 123, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_55_nl_params, AI_ARRAY_FORMAT_S8,
    nl_55_nl_params_data, nl_55_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_55_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_55_layer, 55,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_55_chain,
  NULL, &conversion_56_layer, AI_STATIC, 
  .nl_params = &nl_55_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_53_output, &gemm_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_54_layer, 54,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_54_chain,
  NULL, &nl_55_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output3),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_53_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_53_layer, 53,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_53_chain,
  NULL, &eltwise_54_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_42_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_42_layer, 42,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_42_chain,
  NULL, &gemm_53_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_41_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_40_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_41_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_41_layer, 41,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_41_chain,
  NULL, &gemm_42_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_40_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -124, -122, -120, -118, -114, -110, -105, -98, -89, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 89, 98, 105, 110, 114, 118, 120, 122, 124, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_40_nl_params, AI_ARRAY_FORMAT_S8,
    nl_40_nl_params_data, nl_40_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_40_layer, 40,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_40_chain,
  NULL, &conversion_41_layer, AI_STATIC, 
  .nl_params = &nl_40_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_38_output, &gemm_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_39_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_39_layer, 39,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_39_chain,
  NULL, &nl_40_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output2),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_38_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_38_layer, 38,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_38_chain,
  NULL, &eltwise_39_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_27_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_27_layer, 27,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_27_chain,
  NULL, &gemm_38_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_26_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_26_layer, 26,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_26_chain,
  NULL, &gemm_27_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_25_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -125, -125, -124, -122, -120, -118, -114, -110, -105, -98, -89, -79, -67, -52, -36, -18, 0, 18, 36, 52, 67, 79, 89, 98, 105, 110, 114, 118, 120, 122, 124, 125, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_25_nl_params, AI_ARRAY_FORMAT_S8,
    nl_25_nl_params_data, nl_25_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_25_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_25_chain,
  NULL, &conversion_26_layer, AI_STATIC, 
  .nl_params = &nl_25_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_23_output, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_24_layer, 24,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_24_chain,
  NULL, &nl_25_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_23_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_23_layer, 23,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_23_chain,
  NULL, &eltwise_24_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_12_layer, 12,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_12_chain,
  NULL, &gemm_23_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_11_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_11_layer, 11,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_11_chain,
  NULL, &gemm_12_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_10_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -126, -126, -125, -125, -124, -122, -121, -118, -115, -112, -107, -101, -94, -86, -75, -63, -49, -34, -17, 0, 17, 34, 49, 63, 75, 86, 94, 101, 107, 112, 115, 118, 121, 122, 124, 125, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_10_nl_params, AI_ARRAY_FORMAT_S8,
    nl_10_nl_params_data, nl_10_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_10_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_10_chain,
  NULL, &conversion_11_layer, AI_STATIC, 
  .nl_params = &nl_10_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_output, &gemm_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_9_layer, 9,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_9_chain,
  NULL, &nl_10_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_7_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_8_weights, &gemm_8_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_8_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_8_layer, 8,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_8_chain,
  NULL, &eltwise_9_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  unpack_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 48, &unpack_7_output0, &unpack_7_output1, &unpack_7_output2, &unpack_7_output3, &unpack_7_output4, &unpack_7_output5, &unpack_7_output6, &unpack_7_output7, &unpack_7_output8, &unpack_7_output9, &unpack_7_output10, &unpack_7_output11, &unpack_7_output12, &unpack_7_output13, &unpack_7_output14, &unpack_7_output15, &unpack_7_output16, &unpack_7_output17, &unpack_7_output18, &unpack_7_output19, &unpack_7_output20, &unpack_7_output21, &unpack_7_output22, &unpack_7_output23, &unpack_7_output24, &unpack_7_output25, &unpack_7_output26, &unpack_7_output27, &unpack_7_output28, &unpack_7_output29, &unpack_7_output30, &unpack_7_output31, &unpack_7_output32, &unpack_7_output33, &unpack_7_output34, &unpack_7_output35, &unpack_7_output36, &unpack_7_output37, &unpack_7_output38, &unpack_7_output39, &unpack_7_output40, &unpack_7_output41, &unpack_7_output42, &unpack_7_output43, &unpack_7_output44, &unpack_7_output45, &unpack_7_output46, &unpack_7_output47),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  unpack_7_layer, 7,
  UNPACK_TYPE, 0x0, NULL,
  unpack, forward_unpack,
  &unpack_7_chain,
  NULL, &gemm_8_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_sensor_history0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_6_layer, 6,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_6_chain,
  NULL, &unpack_7_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_253_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &constantofshape_252_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_253_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_253_weights, &gemm_253_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_253_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_253_layer, 253,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_253_chain,
  NULL, &transpose_6_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &constantofshape_4_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_5_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_5_layer, 5,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_5_chain,
  NULL, &gemm_253_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2944, 1, 1),
    2944, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 3064, 1, 1),
    3064, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_sensor_history0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &eltwise_448_output),
  &gemm_5_layer, 0x2abcdd36, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2944, 1, 1),
      2944, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 3064, 1, 1),
      3064, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_sensor_history0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &eltwise_448_output),
  &gemm_5_layer, 0x2abcdd36, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool forecast_temp_ml_model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_forecast_temp_ml_model_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_sensor_history0_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2316);
    serving_default_sensor_history0_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2316);
    gemm_5_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_5_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_5_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    gemm_5_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    gemm_253_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2124);
    gemm_253_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2124);
    gemm_253_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2772);
    gemm_253_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2772);
    transpose_6_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1884);
    transpose_6_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1884);
    unpack_7_output0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2788);
    unpack_7_output0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2788);
    unpack_7_output1_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1872);
    unpack_7_output1_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1872);
    unpack_7_output2_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1860);
    unpack_7_output2_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1860);
    unpack_7_output3_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1848);
    unpack_7_output3_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1848);
    unpack_7_output4_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2800);
    unpack_7_output4_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2800);
    unpack_7_output5_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1836);
    unpack_7_output5_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1836);
    unpack_7_output6_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2812);
    unpack_7_output6_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2812);
    unpack_7_output7_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    unpack_7_output7_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    unpack_7_output8_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2824);
    unpack_7_output8_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2824);
    unpack_7_output9_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1812);
    unpack_7_output9_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1812);
    unpack_7_output10_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2836);
    unpack_7_output10_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2836);
    unpack_7_output11_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1800);
    unpack_7_output11_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1800);
    unpack_7_output12_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    unpack_7_output12_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    unpack_7_output13_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1788);
    unpack_7_output13_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1788);
    unpack_7_output14_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2860);
    unpack_7_output14_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2860);
    unpack_7_output15_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1776);
    unpack_7_output15_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1776);
    unpack_7_output16_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2872);
    unpack_7_output16_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2872);
    unpack_7_output17_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1764);
    unpack_7_output17_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1764);
    unpack_7_output18_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2884);
    unpack_7_output18_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2884);
    unpack_7_output19_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1752);
    unpack_7_output19_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1752);
    unpack_7_output20_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2896);
    unpack_7_output20_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2896);
    unpack_7_output21_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1740);
    unpack_7_output21_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1740);
    unpack_7_output22_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2908);
    unpack_7_output22_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2908);
    unpack_7_output23_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1728);
    unpack_7_output23_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1728);
    unpack_7_output24_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2920);
    unpack_7_output24_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2920);
    unpack_7_output25_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1716);
    unpack_7_output25_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1716);
    unpack_7_output26_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2932);
    unpack_7_output26_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2932);
    unpack_7_output27_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1704);
    unpack_7_output27_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1704);
    unpack_7_output28_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2944);
    unpack_7_output28_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2944);
    unpack_7_output29_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1692);
    unpack_7_output29_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1692);
    unpack_7_output30_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2956);
    unpack_7_output30_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2956);
    unpack_7_output31_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1680);
    unpack_7_output31_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1680);
    unpack_7_output32_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2968);
    unpack_7_output32_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2968);
    unpack_7_output33_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1668);
    unpack_7_output33_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1668);
    unpack_7_output34_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2980);
    unpack_7_output34_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2980);
    unpack_7_output35_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1656);
    unpack_7_output35_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1656);
    unpack_7_output36_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2992);
    unpack_7_output36_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2992);
    unpack_7_output37_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1644);
    unpack_7_output37_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1644);
    unpack_7_output38_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3004);
    unpack_7_output38_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3004);
    unpack_7_output39_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    unpack_7_output39_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    unpack_7_output40_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3016);
    unpack_7_output40_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3016);
    unpack_7_output41_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1620);
    unpack_7_output41_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1620);
    unpack_7_output42_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3028);
    unpack_7_output42_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3028);
    unpack_7_output43_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1608);
    unpack_7_output43_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1608);
    unpack_7_output44_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1596);
    unpack_7_output44_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1596);
    unpack_7_output45_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    unpack_7_output45_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3040);
    unpack_7_output46_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1584);
    unpack_7_output46_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1584);
    unpack_7_output47_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3052);
    unpack_7_output47_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3052);
    gemm_8_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1884);
    gemm_8_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1884);
    gemm_8_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    gemm_8_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2144);
    eltwise_9_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1884);
    eltwise_9_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1884);
    nl_10_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    nl_10_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    conversion_11_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1884);
    conversion_11_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1884);
    gemm_12_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    gemm_12_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    gemm_12_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2196);
    gemm_12_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2196);
    gemm_23_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    gemm_23_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    gemm_23_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2168);
    gemm_23_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2168);
    eltwise_24_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    eltwise_24_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    nl_25_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    nl_25_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    conversion_26_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    conversion_26_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1908);
    gemm_27_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_27_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_27_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2220);
    gemm_27_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2220);
    gemm_38_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_38_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_38_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2192);
    gemm_38_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2192);
    eltwise_39_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1860);
    eltwise_39_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1860);
    nl_40_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    nl_40_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    conversion_41_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1860);
    conversion_41_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1860);
    gemm_42_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_42_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_42_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2220);
    gemm_42_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2220);
    gemm_53_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_53_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_53_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2192);
    gemm_53_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2192);
    eltwise_54_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    eltwise_54_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    nl_55_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    nl_55_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    conversion_56_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    conversion_56_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1932);
    gemm_57_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    gemm_57_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    gemm_57_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2244);
    gemm_57_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2244);
    gemm_68_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    gemm_68_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    gemm_68_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2216);
    gemm_68_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2216);
    eltwise_69_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    eltwise_69_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    nl_70_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    nl_70_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    conversion_71_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    conversion_71_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1956);
    gemm_72_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_72_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_72_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2268);
    gemm_72_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2268);
    gemm_13_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_13_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_13_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2240);
    gemm_13_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2240);
    eltwise_73_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1836);
    eltwise_73_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1836);
    nl_74_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    nl_74_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    conversion_75_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1836);
    conversion_75_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1836);
    gemm_76_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_76_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_76_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2268);
    gemm_76_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2268);
    gemm_14_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_14_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_14_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2240);
    gemm_14_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2240);
    eltwise_77_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    eltwise_77_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    nl_78_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    nl_78_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    conversion_79_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    conversion_79_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1980);
    gemm_80_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    gemm_80_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    gemm_80_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2292);
    gemm_80_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2292);
    gemm_15_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    gemm_15_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    gemm_15_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2264);
    gemm_15_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2264);
    eltwise_81_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    eltwise_81_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    nl_82_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_82_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_83_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    conversion_83_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2004);
    gemm_84_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_84_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_84_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    gemm_84_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    gemm_16_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_16_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_16_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_16_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_85_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_85_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_86_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2052);
    nl_86_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2052);
    conversion_87_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    conversion_87_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2748);
    gemm_88_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_88_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_88_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2788);
    gemm_88_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2788);
    gemm_17_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_17_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_17_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_17_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_89_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1812);
    eltwise_89_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1812);
    nl_90_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_90_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_91_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1812);
    conversion_91_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1812);
    gemm_92_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_92_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_92_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2788);
    gemm_92_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2788);
    gemm_18_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_18_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_18_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_18_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_93_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_93_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_94_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_94_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_95_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2824);
    conversion_95_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2824);
    gemm_96_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_96_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_96_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2800);
    gemm_96_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2800);
    gemm_19_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_19_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_19_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2292);
    gemm_19_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2292);
    eltwise_97_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2292);
    eltwise_97_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2292);
    nl_98_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2268);
    nl_98_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2268);
    conversion_99_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2800);
    conversion_99_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2800);
    gemm_100_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_100_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_100_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    gemm_100_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    gemm_20_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_20_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_20_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_20_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_101_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_101_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_102_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    nl_102_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    conversion_103_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    conversion_103_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    gemm_104_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_104_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_104_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    gemm_104_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    gemm_21_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_21_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_21_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_21_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_105_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1788);
    eltwise_105_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1788);
    nl_106_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    nl_106_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    conversion_107_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1788);
    conversion_107_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1788);
    gemm_108_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_108_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_108_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    gemm_108_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    gemm_22_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_22_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_22_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_22_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_109_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_109_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_110_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    nl_110_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    conversion_111_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    conversion_111_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2848);
    gemm_112_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_112_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_112_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    gemm_112_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    gemm_28_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_28_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_28_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_28_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_113_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_113_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_114_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    nl_114_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    conversion_115_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    conversion_115_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    gemm_116_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_116_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_116_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1512);
    gemm_116_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1512);
    gemm_29_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_29_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_29_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_29_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_117_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_117_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_118_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1512);
    nl_118_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1512);
    conversion_119_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1512);
    conversion_119_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1512);
    gemm_120_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_120_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_120_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    gemm_120_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    gemm_30_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_30_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_30_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_30_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_121_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1764);
    eltwise_121_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1764);
    nl_122_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    nl_122_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    conversion_123_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1764);
    conversion_123_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1764);
    gemm_124_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_124_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_124_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    gemm_124_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    gemm_31_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_31_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_31_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_31_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_125_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_125_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_126_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    nl_126_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    conversion_127_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2872);
    conversion_127_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2872);
    gemm_128_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_128_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_128_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    gemm_128_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    gemm_32_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_32_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_32_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_32_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_129_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_129_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_130_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_130_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_131_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    conversion_131_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    gemm_132_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_132_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_132_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1464);
    gemm_132_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1464);
    gemm_33_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_33_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_33_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_33_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_133_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_133_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_134_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_134_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_135_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1464);
    conversion_135_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1464);
    gemm_136_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_136_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_136_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_136_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_34_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_34_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_34_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_34_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_137_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1740);
    eltwise_137_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1740);
    nl_138_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    nl_138_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    conversion_139_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1740);
    conversion_139_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1740);
    gemm_140_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_140_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_140_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_140_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_35_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_35_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_35_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_35_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_141_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_141_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_142_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    nl_142_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    conversion_143_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2896);
    conversion_143_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2896);
    gemm_144_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_144_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_144_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_144_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_36_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_36_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_36_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_36_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_145_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_145_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_146_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_146_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_147_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    conversion_147_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    gemm_148_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_148_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_148_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1416);
    gemm_148_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1416);
    gemm_37_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_37_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_37_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_37_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_149_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_149_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_150_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_150_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_151_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1416);
    conversion_151_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1416);
    gemm_152_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_152_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_152_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    gemm_152_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    gemm_43_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_43_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_43_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_43_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_153_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1716);
    eltwise_153_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1716);
    nl_154_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    nl_154_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    conversion_155_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1716);
    conversion_155_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1716);
    gemm_156_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_156_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_156_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    gemm_156_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    gemm_44_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_44_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_44_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_44_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_157_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_157_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_158_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    nl_158_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    conversion_159_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2920);
    conversion_159_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2920);
    gemm_160_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_160_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_160_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    gemm_160_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    gemm_45_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_45_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_45_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_45_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_161_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_161_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_162_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_162_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_163_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    conversion_163_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    gemm_164_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_164_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_164_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1368);
    gemm_164_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1368);
    gemm_46_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_46_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_46_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_46_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_165_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_165_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_166_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_166_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_167_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1368);
    conversion_167_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1368);
    gemm_168_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_168_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_168_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    gemm_168_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    gemm_47_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_47_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_47_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_47_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_169_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1692);
    eltwise_169_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1692);
    nl_170_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    nl_170_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    conversion_171_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1692);
    conversion_171_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1692);
    gemm_172_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_172_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_172_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    gemm_172_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    gemm_48_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_48_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_48_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_48_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_173_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_173_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_174_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    nl_174_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    conversion_175_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2944);
    conversion_175_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2944);
    gemm_176_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_176_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_176_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    gemm_176_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    gemm_49_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_49_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_49_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_49_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_177_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_177_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_178_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_178_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_179_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    conversion_179_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    gemm_180_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_180_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_180_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1320);
    gemm_180_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1320);
    gemm_50_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_50_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_50_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_50_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_181_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_181_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_182_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_182_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_183_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1320);
    conversion_183_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1320);
    gemm_184_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_184_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_184_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    gemm_184_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    gemm_51_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_51_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_51_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_51_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_185_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1668);
    eltwise_185_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1668);
    nl_186_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    nl_186_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    conversion_187_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1668);
    conversion_187_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1668);
    gemm_188_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_188_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_188_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    gemm_188_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    gemm_52_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_52_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_52_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_52_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_189_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_189_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_190_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    nl_190_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    conversion_191_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2968);
    conversion_191_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2968);
    gemm_192_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_192_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_192_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    gemm_192_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    gemm_58_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_58_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_58_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_58_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_193_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_193_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_194_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_194_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_195_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    conversion_195_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    gemm_196_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_196_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_196_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1272);
    gemm_196_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1272);
    gemm_59_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_59_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_59_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_59_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_197_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_197_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_198_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_198_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    conversion_199_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1272);
    conversion_199_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1272);
    gemm_200_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_200_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_200_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_200_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_60_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_60_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_60_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_60_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_201_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1644);
    eltwise_201_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1644);
    nl_202_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    nl_202_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    conversion_203_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1644);
    conversion_203_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1644);
    gemm_204_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_204_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_204_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_204_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_61_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_61_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_61_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_61_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_205_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_205_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_206_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    nl_206_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    conversion_207_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2992);
    conversion_207_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2992);
    gemm_208_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_208_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_208_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_208_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_62_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_62_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_62_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_62_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_209_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    eltwise_209_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    nl_210_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    nl_210_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    conversion_211_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    conversion_211_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    gemm_212_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_212_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_212_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    gemm_212_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    gemm_63_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_63_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_63_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_63_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_213_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    eltwise_213_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    nl_214_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    nl_214_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    conversion_215_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    conversion_215_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    gemm_216_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_216_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_216_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    gemm_216_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    gemm_64_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_64_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_64_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_64_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_217_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1620);
    eltwise_217_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1620);
    nl_218_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    nl_218_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    conversion_219_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1620);
    conversion_219_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1620);
    gemm_220_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_220_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_220_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    gemm_220_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    gemm_65_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_65_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_65_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_65_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_221_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    eltwise_221_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    nl_222_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    nl_222_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    conversion_223_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3016);
    conversion_223_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3016);
    gemm_224_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_224_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_224_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    gemm_224_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    gemm_66_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_66_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_66_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_66_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_225_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    eltwise_225_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    nl_226_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    nl_226_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    conversion_227_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    conversion_227_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    gemm_228_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_228_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_228_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    gemm_228_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    gemm_67_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_67_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_67_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_67_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_229_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1596);
    eltwise_229_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1596);
    nl_230_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    nl_230_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    conversion_231_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1596);
    conversion_231_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1596);
    gemm_232_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_232_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_232_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    gemm_232_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    gemm_233_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_233_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_233_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_233_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_234_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    eltwise_234_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    nl_235_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    nl_235_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    conversion_236_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    conversion_236_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    gemm_237_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_237_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_237_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    gemm_237_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    gemm_238_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_238_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_238_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    gemm_238_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2288);
    eltwise_239_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    eltwise_239_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    nl_240_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    nl_240_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    conversion_241_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    conversion_241_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    gemm_242_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_242_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2028);
    gemm_242_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1128);
    gemm_242_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1128);
    gemm_243_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2056);
    gemm_243_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2056);
    gemm_243_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2032);
    gemm_243_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2032);
    eltwise_244_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1128);
    eltwise_244_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1128);
    nl_245_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1104);
    nl_245_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1104);
    conversion_246_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2292);
    conversion_246_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2292);
    pack_247_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    pack_247_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    unpack_254_output0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    unpack_254_output0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1152);
    unpack_254_output1_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    unpack_254_output1_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1176);
    unpack_254_output2_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    unpack_254_output2_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1200);
    unpack_254_output3_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    unpack_254_output3_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1224);
    unpack_254_output4_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    unpack_254_output4_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1248);
    unpack_254_output5_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1272);
    unpack_254_output5_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1272);
    unpack_254_output6_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    unpack_254_output6_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1296);
    unpack_254_output7_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1320);
    unpack_254_output7_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1320);
    unpack_254_output8_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    unpack_254_output8_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1344);
    unpack_254_output9_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1368);
    unpack_254_output9_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1368);
    unpack_254_output10_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    unpack_254_output10_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1392);
    unpack_254_output11_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1416);
    unpack_254_output11_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1416);
    unpack_254_output12_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    unpack_254_output12_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1440);
    unpack_254_output13_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1464);
    unpack_254_output13_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1464);
    unpack_254_output14_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    unpack_254_output14_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1488);
    unpack_254_output15_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1512);
    unpack_254_output15_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1512);
    unpack_254_output16_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    unpack_254_output16_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    unpack_254_output17_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    unpack_254_output17_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    unpack_254_output18_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1584);
    unpack_254_output18_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1584);
    unpack_254_output19_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1608);
    unpack_254_output19_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1608);
    unpack_254_output20_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    unpack_254_output20_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1632);
    unpack_254_output21_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1656);
    unpack_254_output21_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1656);
    unpack_254_output22_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1680);
    unpack_254_output22_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1680);
    unpack_254_output23_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1704);
    unpack_254_output23_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1704);
    unpack_254_output24_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1728);
    unpack_254_output24_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1728);
    unpack_254_output25_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1752);
    unpack_254_output25_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1752);
    unpack_254_output26_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1776);
    unpack_254_output26_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1776);
    unpack_254_output27_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1800);
    unpack_254_output27_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1800);
    unpack_254_output28_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    unpack_254_output28_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1824);
    unpack_254_output29_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1848);
    unpack_254_output29_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1848);
    unpack_254_output30_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1872);
    unpack_254_output30_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1872);
    unpack_254_output31_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1896);
    unpack_254_output31_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1896);
    unpack_254_output32_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1920);
    unpack_254_output32_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1920);
    unpack_254_output33_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1944);
    unpack_254_output33_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1944);
    unpack_254_output34_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1968);
    unpack_254_output34_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1968);
    unpack_254_output35_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1992);
    unpack_254_output35_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1992);
    unpack_254_output36_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    unpack_254_output36_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2016);
    unpack_254_output37_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2040);
    unpack_254_output37_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2040);
    unpack_254_output38_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2064);
    unpack_254_output38_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2064);
    unpack_254_output39_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2088);
    unpack_254_output39_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2088);
    unpack_254_output40_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2112);
    unpack_254_output40_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2112);
    unpack_254_output41_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2136);
    unpack_254_output41_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2136);
    unpack_254_output42_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2160);
    unpack_254_output42_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2160);
    unpack_254_output43_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2184);
    unpack_254_output43_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2184);
    unpack_254_output44_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    unpack_254_output44_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2208);
    unpack_254_output45_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2232);
    unpack_254_output45_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2232);
    unpack_254_output46_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2256);
    unpack_254_output46_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2256);
    unpack_254_output47_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2280);
    unpack_254_output47_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2280);
    gemm_255_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_255_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_255_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_255_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    eltwise_256_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_256_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_257_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_257_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_258_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_258_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_258_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_258_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_269_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_269_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_269_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_269_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_270_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_270_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_271_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_271_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_272_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_272_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_272_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_272_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_283_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_283_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_283_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_283_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_284_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_284_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_285_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_285_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_286_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_286_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_286_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_286_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_297_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_297_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_297_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_297_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_298_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_298_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_299_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_299_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_300_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_300_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_300_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_300_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_311_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_311_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_311_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_311_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_312_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_312_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_313_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_313_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_314_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_314_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_314_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_314_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_259_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_259_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_259_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_259_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_315_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_315_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_316_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_316_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_317_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_317_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_317_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_317_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_260_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_260_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_260_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_260_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_318_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_318_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_319_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_319_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_320_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_320_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_320_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_320_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_261_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_261_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_261_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_261_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_321_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_321_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_322_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_322_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_323_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_323_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_323_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_323_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_262_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_262_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_262_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_262_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_324_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_324_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_325_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_325_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_326_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_326_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_326_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_326_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_263_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_263_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_263_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_263_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_327_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_327_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_328_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_328_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_329_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_329_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_329_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_329_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_264_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_264_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_264_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_264_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_330_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_330_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_331_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_331_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_332_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_332_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_332_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_332_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_265_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_265_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_265_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_265_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_333_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_333_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_334_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_334_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_335_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_335_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_335_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_335_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_266_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_266_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_266_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_266_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_336_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_336_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_337_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_337_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_338_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_338_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_338_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_338_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_267_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_267_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_267_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_267_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_339_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_339_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_340_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_340_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_341_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_341_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_341_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_341_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_268_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_268_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_268_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_268_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_342_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_342_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_343_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_343_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_344_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_344_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_344_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_344_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_273_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_273_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_273_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_273_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_345_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_345_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_346_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_346_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_347_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_347_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_347_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_347_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_274_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_274_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_274_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_274_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_348_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_348_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_349_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_349_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_350_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_350_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_350_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_350_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_275_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_275_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_275_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_275_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_351_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_351_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_352_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_352_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_353_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_353_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_353_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_353_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_276_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_276_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_276_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_276_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_354_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_354_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_355_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_355_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_356_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_356_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_356_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_356_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_277_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_277_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_277_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_277_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_357_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_357_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_358_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_358_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_359_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_359_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_359_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_359_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_278_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_278_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_278_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_278_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_360_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_360_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_361_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_361_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_362_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_362_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_362_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_362_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_279_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_279_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_279_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_279_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_363_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_363_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_364_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_364_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_365_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_365_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_365_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_365_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_280_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_280_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_280_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_280_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_366_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_366_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_367_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_367_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_368_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_368_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_368_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_368_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_281_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_281_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_281_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_281_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_369_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_369_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_370_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_370_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_371_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_371_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_371_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_371_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_282_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_282_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_282_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_282_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_372_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_372_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_373_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_373_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_374_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_374_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_374_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_374_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_287_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_287_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_287_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_287_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_375_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_375_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_376_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_376_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_377_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_377_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_377_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_377_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_288_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_288_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_288_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_288_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_378_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_378_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_379_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_379_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_380_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_380_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_380_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_380_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_289_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_289_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_289_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_289_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_381_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_381_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_382_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_382_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_383_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_383_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_383_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_383_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_290_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_290_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_290_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_290_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_384_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_384_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_385_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_385_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_386_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_386_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_386_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_386_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_291_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_291_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_291_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_291_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_387_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_387_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_388_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_388_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_389_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_389_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_389_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_389_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_292_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_292_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_292_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_292_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_390_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_390_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_391_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_391_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_392_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_392_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_392_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_392_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_293_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_293_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_293_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_293_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_393_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_393_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_394_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_394_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_395_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_395_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_395_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_395_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_294_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_294_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_294_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_294_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_396_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_396_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_397_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_397_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_398_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_398_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_398_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_398_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_295_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_295_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_295_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_295_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_399_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_399_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_400_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_400_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_401_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_401_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_401_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_401_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_296_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_296_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_296_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_296_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_402_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_402_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_403_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_403_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_404_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_404_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_404_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_404_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_301_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_301_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_301_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_301_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_405_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_405_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_406_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_406_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_407_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_407_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_407_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_407_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_302_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_302_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_302_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_302_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_408_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_408_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_409_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_409_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_410_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_410_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_410_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_410_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_303_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_303_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_303_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_303_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_411_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_411_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_412_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_412_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_413_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_413_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_413_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_413_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_304_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_304_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_304_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_304_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_414_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_414_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_415_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_415_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_416_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_416_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_416_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_416_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_305_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_305_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_305_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_305_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_417_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_417_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_418_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_418_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_419_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_419_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_419_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_419_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_306_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_306_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_306_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_306_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_420_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_420_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_421_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_421_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_422_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_422_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_422_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_422_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_307_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_307_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_307_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_307_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_423_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_423_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_424_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_424_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_425_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_425_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_425_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_425_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_308_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_308_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_308_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_308_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_426_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_426_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_427_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_427_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_428_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_428_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_428_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_428_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_309_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_309_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_309_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_309_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_429_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_429_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_430_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_430_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_431_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_431_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_431_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_431_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_310_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_310_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_310_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_310_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_432_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_432_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_433_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_433_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_434_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_434_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_434_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_434_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_435_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_435_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_435_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_435_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_436_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_436_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_437_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_437_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_438_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_438_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_438_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_438_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 208);
    gemm_439_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_439_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_439_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_439_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_440_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_440_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_441_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_441_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_442_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_442_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 32);
    gemm_442_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_442_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_443_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_443_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_443_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    gemm_443_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 224);
    eltwise_444_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    eltwise_444_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    nl_445_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    nl_445_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_446_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_446_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    gemm_446_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    gemm_446_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    gemm_447_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_447_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_447_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 312);
    gemm_447_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 312);
    slice_0_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    slice_0_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    eltwise_448_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 24);
    eltwise_448_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 24);
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
    
    constantofshape_4_const_array.format |= AI_FMT_FLAG_CONST;
    constantofshape_4_const_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 0);
    constantofshape_4_const_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 0);
    gemm_5_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_5_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 24);
    gemm_5_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 24);
    gemm_5_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_5_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 600);
    gemm_5_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 600);
    constantofshape_252_const_array.format |= AI_FMT_FLAG_CONST;
    constantofshape_252_const_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 696);
    constantofshape_252_const_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 696);
    gemm_253_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_253_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 712);
    gemm_253_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 712);
    gemm_253_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_253_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 968);
    gemm_253_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 968);
    gemm_8_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_8_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1032);
    gemm_8_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1032);
    gemm_8_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_8_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1248);
    gemm_8_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1248);
    gemm_255_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_255_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1344);
    gemm_255_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1344);
    gemm_255_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_255_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1728);
    gemm_255_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1728);
    gemm_446_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_446_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1792);
    gemm_446_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1792);
    gemm_446_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_446_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2176);
    gemm_446_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2176);
    gemm_447_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_447_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2272);
    gemm_447_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2272);
    gemm_447_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_447_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2848);
    gemm_447_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2848);
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
      
      .n_macc            = 79968,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x2abcdd36,
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
      
      .n_macc            = 79968,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x2abcdd36,
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

