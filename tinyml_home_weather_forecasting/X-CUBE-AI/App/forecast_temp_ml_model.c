/**
  ******************************************************************************
  * @file    forecast_temp_ml_model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-11-03T20:24:15-0400
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
#define AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE     "0xf9b44ffa62fb0fcc3d9267a4cc9f1574"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-11-03T20:24:15-0400"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_FORECAST_TEMP_ML_MODEL_N_BATCHES
#define AI_FORECAST_TEMP_ML_MODEL_N_BATCHES         (1)

static ai_ptr g_forecast_temp_ml_model_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_forecast_temp_ml_model_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_pruned_model_input0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 42, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 324, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 432, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  pool_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  pool_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1890, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 54, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7776, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3072, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4676, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6440, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 913, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 832, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 64, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03053707629442215f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_10_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001513218623585999f, 0.0008158868877217174f, 0.0016049511032178998f, 0.0009392249048687518f, 0.0019405147759243846f, 0.0015654991148039699f, 0.0011894471244886518f, 0.00120908475946635f, 0.0017842965899035335f, 0.0009286694112233818f, 0.0015419271076098084f, 0.000860819302033633f, 0.0018535066628828645f, 0.0016082831425592303f, 0.0015214589657261968f, 0.0016378357540816069f, 0.001288314932025969f, 0.0015197311295196414f, 0.0018756604986265302f, 0.0014131100615486503f, 0.0016397930448874831f, 0.000869953422807157f, 0.001788427121937275f, 0.0012704780092462897f, 0.0016492429422214627f, 0.0017154640518128872f, 0.0017629271605983377f, 0.0012216736795380712f, 0.0012882305309176445f, 0.0015216232277452946f, 0.0015435474924743176f, 0.0018502428429201245f, 0.0013941326178610325f, 0.001521180965937674f, 0.0014344330411404371f, 0.0017793950391933322f, 0.0013059645425528288f, 0.0016169542213901877f, 0.0011679227463901043f, 0.0016569760628044605f, 0.0011892644688487053f, 0.0016613008920103312f, 0.0016871027182787657f, 0.0016369182849302888f, 0.0015396164963021874f, 0.0017680132295936346f, 0.0017442085081711411f, 0.001695237006060779f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0401119627058506f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0021398633252829313f, 0.002232820726931095f, 0.0019497355679050088f, 0.0017757248133420944f, 0.001816415460780263f, 0.002539180452004075f, 0.0017520029796287417f, 0.0018127326620742679f, 0.0018645086092874408f, 0.0017988765612244606f, 0.0025916725862771273f, 0.0018275115871801972f, 0.002102351514622569f, 0.0023048315197229385f, 0.002786541124805808f, 0.0020826871041208506f, 0.0019141793018206954f, 0.002896229736506939f, 0.0025869328528642654f, 0.0017361431382596493f, 0.001828549662604928f, 0.002154225017875433f, 0.0017481122631579638f, 0.00181098363827914f, 0.001742471707984805f, 0.0023933856282383204f, 0.0019428232917562127f, 0.00209842249751091f, 0.0017340274062007666f, 0.001626485725864768f, 0.0022855019196867943f, 0.0017821809742599726f, 0.0022147241979837418f, 0.0017597861588001251f, 0.0025042062625288963f, 0.001756221754476428f, 0.0020487764850258827f, 0.0024383962154388428f, 0.002537874272093177f, 0.00200229836627841f, 0.0021805106662213802f, 0.002230873564258218f, 0.0018106455681845546f, 0.001716943341307342f, 0.0018549259984865785f, 0.0018214575247839093f, 0.0021506655029952526f, 0.0024729673750698566f, 0.002451990032568574f, 0.0017354529118165374f, 0.0025853889528661966f, 0.002417336916550994f, 0.0018379755783826113f, 0.002495571505278349f, 0.001931060804054141f, 0.0020290084648877382f, 0.0015915599651634693f, 0.0022542504593729973f, 0.0025647389702498913f, 0.0018907715566456318f, 0.0018020820571109653f, 0.001683534006588161f, 0.0016893489519134164f, 0.0026467936113476753f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015420956537127495f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 54,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0018904425669461489f, 0.0013871983392164111f, 0.001594594563357532f, 0.0017144547309726477f, 0.0016048818361014128f, 0.0015175340231508017f, 0.002247604075819254f, 0.001697071478702128f, 0.0014861678937450051f, 0.0019028395181521773f, 0.001656202832236886f, 0.0014373271260410547f, 0.001690123463049531f, 0.002202454721555114f, 0.001424757530912757f, 0.0019230882171541452f, 0.0017398529453203082f, 0.0019712906796485186f, 0.0014836100162938237f, 0.0013809435768052936f, 0.0017627340275794268f, 0.0015537026338279247f, 0.0014610722428187728f, 0.0016936163883656263f, 0.0016923160292208195f, 0.0017046574503183365f, 0.001914253574796021f, 0.0014352693688124418f, 0.001526659936644137f, 0.0011861041421070695f, 0.0017062417464330792f, 0.001642299466766417f, 0.0016092018922790885f, 0.0015774143394082785f, 0.0019623904954642057f, 0.0018786455038934946f, 0.00188386847730726f, 0.0019415507558733225f, 0.0017574353842064738f, 0.0016501572681590915f, 0.0015145797515287995f, 0.0018283865647390485f, 0.0014468457084149122f, 0.0018377691740170121f, 0.0015916021075099707f, 0.002164300298318267f, 0.0013654559152200818f, 0.0013529801508411765f, 0.00152070471085608f, 0.0016268910840153694f, 0.0013063704827800393f, 0.0017980696866288781f, 0.0015656250761821866f, 0.001388670178130269f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.040429335087537766f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015420956537127495f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001654455903917551f, 0.0016523173544555902f, 0.001599255483597517f, 0.0016889162361621857f, 0.0022393404506146908f, 0.001490569906309247f, 0.0017962430138140917f, 0.00146475771907717f, 0.0017090009059756994f, 0.0015690162545070052f, 0.0014638621360063553f, 0.0015858636470511556f, 0.001786517444998026f, 0.001534429844468832f, 0.0015940949087962508f, 0.0016067553078755736f, 0.0015258198836818337f, 0.0014108163304626942f, 0.0016991550801321864f, 0.0017483291449025273f, 0.001739394967444241f, 0.001715890597552061f, 0.0018628743709996343f, 0.0016373076941817999f, 0.0017791459104046226f, 0.0016123054083436728f, 0.0015801459085196257f, 0.0016410108655691147f, 0.0016450698021799326f, 0.0017990419873967767f, 0.0017176467226818204f, 0.001673585851676762f, 0.0015385467559099197f, 0.0016588533762842417f, 0.0017242648173123598f, 0.001616245019249618f, 0.001600685529410839f, 0.0015905844047665596f, 0.0017938207602128386f, 0.0018114590784534812f, 0.001536583062261343f, 0.0014741814229637384f, 0.0017915506614372134f, 0.0016070578712970018f, 0.0016406784998252988f, 0.001754704862833023f, 0.0017126896418631077f, 0.0017008965369313955f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.16668182611465454f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0031097522005438805f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03157194331288338f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.040429335087537766f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_pruned_model_input0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028399387374520302f),
    AI_PACK_INTQ_ZP(-19)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &conv2d_10_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 3, 1), AI_STRIDE_INIT(4, 1, 1, 48, 144),
  1, &conv2d_10_output_array, &conv2d_10_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 913, 1, 1), AI_STRIDE_INIT(4, 1, 1, 913, 913),
  1, &conv2d_10_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_weights, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 48, 3, 1, 1), AI_STRIDE_INIT(4, 1, 48, 48, 144),
  1, &conv2d_10_weights_array, &conv2d_10_weights_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_bias, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_11_bias_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 3, 1), AI_STRIDE_INIT(4, 1, 1, 64, 192),
  1, &conv2d_11_output_array, &conv2d_11_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output0, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 3), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_11_output_array, &conv2d_11_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 832, 1, 1), AI_STRIDE_INIT(4, 1, 1, 832, 832),
  1, &conv2d_11_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_weights, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 48, 1, 1, 64), AI_STRIDE_INIT(4, 1, 48, 3072, 3072),
  1, &conv2d_11_weights_array, &conv2d_11_weights_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 54, 1, 1), AI_STRIDE_INIT(4, 4, 4, 216, 216),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 54, 6, 1), AI_STRIDE_INIT(4, 1, 1, 54, 324),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 4676, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4676, 4676),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 7, 5, 1, 54), AI_STRIDE_INIT(4, 1, 7, 378, 1890),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 6, 1), AI_STRIDE_INIT(4, 1, 1, 48, 288),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_pad_before_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 54, 8, 1), AI_STRIDE_INIT(4, 1, 1, 54, 432),
  1, &conv2d_4_pad_before_output_array, &conv2d_4_pad_before_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 6440, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6440, 6440),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 54, 3, 1, 48), AI_STRIDE_INIT(4, 1, 54, 2592, 7776),
  1, &conv2d_4_weights_array, &conv2d_4_weights_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_14_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_14_output_array, &gemm_14_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 2, 2, 128, 128),
  1, &gemm_14_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_weights, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &gemm_14_weights_array, &gemm_14_weights_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  pool_13_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_13_output_array, &pool_13_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  pool_7_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 3, 1), AI_STRIDE_INIT(4, 1, 1, 48, 144),
  1, &pool_7_output_array, &pool_7_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_pruned_model_input0_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 7, 1, 6), AI_STRIDE_INIT(4, 1, 1, 7, 7),
  1, &serving_default_pruned_model_input0_output_array, &serving_default_pruned_model_input0_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_pruned_model_input0_output0, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 7, 6, 1), AI_STRIDE_INIT(4, 1, 1, 7, 42),
  1, &serving_default_pruned_model_input0_output_array, &serving_default_pruned_model_input0_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_14_weights, &gemm_14_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_14_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &gemm_14_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_13_layer, 13,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_13_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_11_weights, &conv2d_11_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_11_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_pw_sssa8_ch,
  &conv2d_11_chain,
  NULL, &pool_13_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_10_weights, &conv2d_10_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_10_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_10_layer, 10,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &conv2d_10_chain,
  NULL, &conv2d_11_layer, AI_STATIC, 
  .groups = 48, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_7_layer, 7,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_7_chain,
  NULL, &conv2d_10_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_4_chain,
  NULL, &pool_7_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_4_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_4_pad_before_value_data, conv2d_4_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_pad_before_layer, 4,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_4_pad_before_chain,
  NULL, &conv2d_4_layer, AI_STATIC, 
  .value = &conv2d_4_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_pruned_model_input0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_1_chain,
  NULL, &conv2d_4_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 13808, 1, 1),
    13808, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 7160, 1, 1),
    7160, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_pruned_model_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &gemm_14_output),
  &conv2d_1_layer, 0xaab8b9da, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 13808, 1, 1),
      13808, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 7160, 1, 1),
      7160, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_pruned_model_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &gemm_14_output),
  &conv2d_1_layer, 0xaab8b9da, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool forecast_temp_ml_model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_forecast_temp_ml_model_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_pruned_model_input0_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    serving_default_pruned_model_input0_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    conv2d_1_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 332);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 332);
    conv2d_1_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 5008);
    conv2d_1_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 5008);
    conv2d_4_pad_before_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    conv2d_4_pad_before_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    conv2d_4_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 720);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 720);
    conv2d_4_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_4_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    pool_7_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    pool_7_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 288);
    conv2d_10_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 432);
    conv2d_10_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 432);
    conv2d_10_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_10_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_11_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 144);
    conv2d_11_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 144);
    conv2d_11_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 976);
    conv2d_11_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 976);
    pool_13_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    pool_13_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_14_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_14_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 64);
    gemm_14_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
    gemm_14_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 192);
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
    
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 0);
    conv2d_1_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 0);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1892);
    conv2d_1_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1892);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2108);
    conv2d_4_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2108);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 9884);
    conv2d_4_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 9884);
    conv2d_10_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10076);
    conv2d_10_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10076);
    conv2d_10_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10220);
    conv2d_10_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10220);
    conv2d_11_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10412);
    conv2d_11_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10412);
    conv2d_11_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 13484);
    conv2d_11_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 13484);
    gemm_14_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_14_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 13740);
    gemm_14_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 13740);
    gemm_14_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_14_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 13804);
    gemm_14_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 13804);
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
      
      .n_macc            = 68403,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xaab8b9da,
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
      
      .n_macc            = 68403,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xaab8b9da,
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

