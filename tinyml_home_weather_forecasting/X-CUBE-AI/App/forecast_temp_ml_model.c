/**
  ******************************************************************************
  * @file    forecast_temp_ml_model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-02-26T20:23:29-0400
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
#define AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE     "0x4d409948577d37dc23849c60d21d66aa"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-02-26T20:23:29-0400"

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
  NULL, NULL, 192, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 624, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 650, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 528, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  pool_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 264, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 264, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  pool_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  pool_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  gemm_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  gemm_22_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 24, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 260, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 13, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 429, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 11, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 33, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 11, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 704, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  gemm_21_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  gemm_21_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  gemm_22_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  gemm_22_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 782, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1168, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 210, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 684, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1217, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  gemm_21_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  gemm_22_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 184, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11074721068143845f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_10_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 11,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005350487306714058f, 0.004732078872621059f, 0.003707330673933029f, 0.004054721910506487f, 0.003441560547798872f, 0.003127141622826457f, 0.00443303445354104f, 0.002631062176078558f, 0.004276633262634277f, 0.004094214178621769f, 0.0030000247061252594f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08164483308792114f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007832564413547516f, 0.009326611645519733f, 0.008832874707877636f, 0.009016583673655987f, 0.0066882651299238205f, 0.012092666700482368f, 0.008707882836461067f, 0.010380731895565987f, 0.009430364705622196f, 0.005874906200915575f, 0.014923305250704288f, 0.0067669996060431f, 0.011289691552519798f, 0.010254830121994019f, 0.010048164054751396f, 0.01273744460195303f, 0.00850208755582571f, 0.011245687492191792f, 0.0100189708173275f, 0.012059381231665611f, 0.008631853386759758f, 0.016135212033987045f, 0.007685053162276745f, 0.011396783404052258f, 0.008490379899740219f, 0.00584549643099308f, 0.007543782703578472f, 0.009188948199152946f, 0.008530424907803535f, 0.008579620160162449f, 0.013604247011244297f, 0.01538029219955206f, 0.016703525558114052f, 0.014602812938392162f, 0.006324004847556353f, 0.00818940531462431f, 0.010439186356961727f, 0.00962247047573328f, 0.010139639489352703f, 0.011417234316468239f, 0.009569092653691769f, 0.007699210662394762f, 0.007302328944206238f, 0.008270534686744213f, 0.011402669362723827f, 0.010877803899347782f, 0.010792126879096031f, 0.011088645085692406f, 0.009721213951706886f, 0.008333224803209305f, 0.01235564611852169f, 0.012693681754171848f, 0.010489857755601406f, 0.006439092103391886f, 0.00854296050965786f, 0.014771335758268833f, 0.008475203067064285f, 0.014101880602538586f, 0.015247585251927376f, 0.01237858459353447f, 0.011685388162732124f, 0.009673266671597958f, 0.013434866443276405f, 0.005696900188922882f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12723319232463837f),
    AI_PACK_INTQ_ZP(-24)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_14_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0029345573857426643f, 0.0027924953028559685f, 0.003174524288624525f, 0.002969128079712391f, 0.0010404505301266909f, 0.0025977755431085825f, 0.002781576244160533f, 0.002917308360338211f, 0.0030422587879002094f, 0.0030802846886217594f, 0.0025299463886767626f, 0.0030285280663520098f, 0.004254624247550964f, 0.0028665128629654646f, 0.0026359206531196833f, 0.0032942239195108414f, 0.005170287098735571f, 0.00492735393345356f, 0.0019332892261445522f, 0.004856501240283251f, 0.0037686447612941265f, 0.0033991967793554068f, 0.0038901835214346647f, 0.003107656491920352f, 0.003653700230643153f, 0.003163730027154088f, 0.0037188150454312563f, 0.003195978933945298f, 0.0044159796088933945f, 0.0015731474850326777f, 0.004584674257785082f, 0.0038113573100417852f, 0.003108748933300376f, 0.003904503071680665f, 0.003484743647277355f, 0.0027687866240739822f, 0.0035088255535811186f, 0.00399808632209897f, 0.004206059966236353f, 0.0019355526892468333f, 0.0034279506653547287f, 0.002105772728100419f, 0.0034347642213106155f, 0.002232255646958947f, 0.0032069056760519743f, 0.0046382625587284565f, 0.0033767393324524164f, 0.0030152813997119665f, 0.004956261720508337f, 0.002536000683903694f, 0.002708452520892024f, 0.004287658259272575f, 0.004215140361338854f, 0.0043002949096262455f, 0.0029567950405180454f, 0.005303190089762211f, 0.003308673156425357f, 0.003001856617629528f, 0.003765600500628352f, 0.00755431829020381f, 0.004979456774890423f, 0.0030560684390366077f, 0.004229707643389702f, 0.001995248720049858f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12164026498794556f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027441369369626045f, 0.02915017679333687f, 0.02874109148979187f, 0.021115263924002647f, 0.013952291570603848f, 0.013926341198384762f, 0.011057856492698193f, 0.008758842013776302f, 0.020012805238366127f, 0.009612045250833035f, 0.017341796308755875f, 0.02577490545809269f, 0.028877297416329384f, 0.01744714193046093f, 0.005777833517640829f, 0.012174825184047222f, 0.022116802632808685f, 0.011371340602636337f, 0.019146600738167763f, 0.025947565212845802f, 0.007372638676315546f, 0.013476237654685974f, 0.017766831442713737f, 0.029845785349607468f, 0.008445998653769493f, 0.017078371718525887f, 0.008048505522310734f, 0.014262856915593147f, 0.007585184648633003f, 0.022714685648679733f, 0.03480176255106926f, 0.015503454953432083f, 0.00610259547829628f, 0.015629079192876816f, 0.02004598081111908f, 0.00910228956490755f, 0.007150670979171991f, 0.02498669922351837f, 0.023741401731967926f, 0.02361186407506466f, 0.014320786111056805f, 0.029768671840429306f, 0.021632565185427666f, 0.008136986754834652f, 0.007070751860737801f, 0.021471180021762848f, 0.01115778461098671f, 0.012717650271952152f, 0.008924282155930996f, 0.00932248029857874f, 0.02406003326177597f, 0.007370090577751398f, 0.010802687145769596f, 0.01776856742799282f, 0.006863531190901995f, 0.023450413718819618f, 0.00862219836562872f, 0.01841871812939644f, 0.010415147058665752f, 0.016497604548931122f, 0.01004882249981165f, 0.015398752875626087f, 0.01982569694519043f, 0.039659321308135986f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07882823795080185f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 13,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008024413138628006f, 0.009032538160681725f, 0.011035154573619366f, 0.006180938333272934f, 0.009888973087072372f, 0.014338008128106594f, 0.010065250098705292f, 0.008125550113618374f, 0.014443640597164631f, 0.008617540821433067f, 0.009592438116669655f, 0.013333698734641075f, 0.012192869558930397f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1401149034500122f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07882823795080185f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 11,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018744932487607002f, 0.013631953857839108f, 0.010748306289315224f, 0.017913540825247765f, 0.014355354942381382f, 0.009306159801781178f, 0.011724761687219143f, 0.013976920396089554f, 0.01766660064458847f, 0.010645044967532158f, 0.012357193045318127f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_21_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023485718294978142f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_21_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002133660949766636f, 0.004299398511648178f, 0.0021679005585610867f, 0.020780116319656372f, 0.004030207637697458f, 0.013340690173208714f, 0.0021428822074085474f, 0.014028061181306839f, 0.011772979982197285f, 0.01467446330934763f, 0.0021682986989617348f, 0.0040622614324092865f, 0.012558030895888805f, 0.0041150543838739395f, 0.005025605671107769f, 0.002168704755604267f, 0.011426497250795364f, 0.004034600220620632f, 0.004764198325574398f, 0.004246591590344906f, 0.008591114543378353f, 0.0020555208902806044f, 0.016299191862344742f, 0.016022689640522003f, 0.005029439460486174f, 0.005896884482353926f, 0.004438425879925489f, 0.004217413254082203f, 0.0021512615494430065f, 0.0037915639113634825f, 0.004251337144523859f, 0.003937899135053158f, 0.002168054925277829f, 0.0021671978756785393f, 0.008925508707761765f, 0.004159180447459221f, 0.014172295108437538f, 0.0040927547961473465f, 0.01826845481991768f, 0.013992400839924812f, 0.00932511780411005f, 0.0019802129827439785f, 0.004111751448363066f, 0.005026127211749554f, 0.014191888272762299f, 0.0020633370149880648f, 0.01160283200442791f, 0.0021442812867462635f, 0.02187519147992134f, 0.00412346376106143f, 0.014518413692712784f, 0.013974335975944996f, 0.004347093869000673f, 0.004162176977843046f, 0.004505019634962082f, 0.0048457300290465355f, 0.01416077371686697f, 0.01128842867910862f, 0.014323638752102852f, 0.0021527153439819813f, 0.004084739834070206f, 0.0020993368234485388f, 0.013941486366093159f, 0.004462997894734144f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_22_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14232979714870453f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_22_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004829682409763336f, 0.003924019169062376f, 0.0042566293850541115f, 0.004775666631758213f, 0.004507454112172127f, 0.004483551252633333f, 0.003896888345479965f, 0.003490766743198037f, 0.003908114042133093f, 0.004437877796590328f, 0.0033007217571139336f, 0.005378402303904295f, 0.004239118192344904f, 0.005321887321770191f, 0.005302312783896923f, 0.004933464340865612f, 0.005082386080175638f, 0.0041804239153862f, 0.004737942945212126f, 0.0038519278168678284f, 0.004023613873869181f, 0.003809007816016674f, 0.004254661500453949f, 0.005213046446442604f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12164026498794556f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_20_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012342232279479504f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1401149034500122f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_pruned_model_input0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03731202334165573f),
    AI_PACK_INTQ_ZP(-29)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 11, 1, 1), AI_STRIDE_INIT(4, 4, 4, 44, 44),
  1, &conv2d_10_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 11, 24, 1), AI_STRIDE_INIT(4, 1, 1, 11, 264),
  1, &conv2d_10_output_array, &conv2d_10_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 210, 1, 1), AI_STRIDE_INIT(4, 1, 1, 210, 210),
  1, &conv2d_10_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_weights, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 11, 3, 1, 1), AI_STRIDE_INIT(4, 1, 11, 11, 33),
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
  AI_SHAPE_INIT(4, 1, 64, 24, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1536),
  1, &conv2d_11_output_array, &conv2d_11_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_scratch0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 684, 1, 1), AI_STRIDE_INIT(4, 1, 1, 684, 684),
  1, &conv2d_11_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_weights, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 11, 1, 1, 64), AI_STRIDE_INIT(4, 1, 11, 704, 704),
  1, &conv2d_11_weights_array, &conv2d_11_weights_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_14_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 24, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1536),
  1, &conv2d_14_output_array, &conv2d_14_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_scratch0, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 1217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1217, 1217),
  1, &conv2d_14_scratch0_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_weights, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 192),
  1, &conv2d_14_weights_array, &conv2d_14_weights_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_15_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 24, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1536),
  1, &conv2d_15_output_array, &conv2d_15_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_scratch0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 896, 1, 1), AI_STRIDE_INIT(4, 1, 1, 896, 896),
  1, &conv2d_15_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_weights, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 1, 64, 4096, 4096),
  1, &conv2d_15_weights_array, &conv2d_15_weights_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 13, 1, 1), AI_STRIDE_INIT(4, 4, 4, 52, 52),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 13, 48, 1), AI_STRIDE_INIT(4, 1, 1, 13, 624),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 782, 1, 1), AI_STRIDE_INIT(4, 1, 1, 782, 782),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 4, 5, 1, 13), AI_STRIDE_INIT(4, 1, 4, 52, 260),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 11, 1, 1), AI_STRIDE_INIT(4, 4, 4, 44, 44),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 11, 48, 1), AI_STRIDE_INIT(4, 1, 1, 11, 528),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_pad_before_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 13, 50, 1), AI_STRIDE_INIT(4, 1, 1, 13, 650),
  1, &conv2d_4_pad_before_output_array, &conv2d_4_pad_before_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 1168, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1168, 1168),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 13, 3, 1, 11), AI_STRIDE_INIT(4, 1, 13, 143, 429),
  1, &conv2d_4_weights_array, &conv2d_4_weights_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  gemm_21_bias, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_21_bias_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  gemm_21_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_21_output_array, &gemm_21_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  gemm_21_scratch0, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_21_scratch0_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  gemm_21_weights, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 64, 64, 1, 1), AI_STRIDE_INIT(4, 1, 64, 4096, 4096),
  1, &gemm_21_weights_array, &gemm_21_weights_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  gemm_22_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &gemm_22_bias_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  gemm_22_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 1, 1, 24, 24),
  1, &gemm_22_output_array, &gemm_22_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  gemm_22_scratch0, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 184, 1, 1), AI_STRIDE_INIT(4, 2, 2, 368, 368),
  1, &gemm_22_scratch0_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  gemm_22_weights, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 64, 24, 1, 1), AI_STRIDE_INIT(4, 1, 64, 1536, 1536),
  1, &gemm_22_weights_array, &gemm_22_weights_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  pool_18_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 12, 1), AI_STRIDE_INIT(4, 1, 1, 64, 768),
  1, &pool_18_output_array, &pool_18_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  pool_18_output0, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 12), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_18_output_array, &pool_18_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  pool_20_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_20_output_array, &pool_20_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  pool_7_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 11, 24, 1), AI_STRIDE_INIT(4, 1, 1, 11, 264),
  1, &pool_7_output_array, &pool_7_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_pruned_model_input0_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 1, 48), AI_STRIDE_INIT(4, 1, 1, 4, 4),
  1, &serving_default_pruned_model_input0_output_array, &serving_default_pruned_model_input0_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_pruned_model_input0_output0, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 48, 1), AI_STRIDE_INIT(4, 1, 1, 4, 192),
  1, &serving_default_pruned_model_input0_output_array, &serving_default_pruned_model_input0_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_22_weights, &gemm_22_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_22_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_22_layer, 22,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_22_chain,
  NULL, &gemm_22_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_21_weights, &gemm_21_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_21_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_21_layer, 21,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_21_chain,
  NULL, &gemm_22_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_18_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_20_layer, 20,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_20_chain,
  NULL, &gemm_21_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 12), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 12), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_18_layer, 18,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_18_chain,
  NULL, &pool_20_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_15_weights, &conv2d_15_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_15_layer, 15,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_pw_sssa8_ch,
  &conv2d_15_chain,
  NULL, &pool_18_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_14_weights, &conv2d_14_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_14_layer, 14,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_sssa8_ch,
  &conv2d_14_chain,
  NULL, &conv2d_15_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
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
  NULL, &conv2d_14_layer, AI_STATIC, 
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
  .groups = 11, 
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
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 1), 
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
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12612, 1, 1),
    12612, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 4292, 1, 1),
    4292, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_pruned_model_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &gemm_22_output),
  &conv2d_1_layer, 0xaa1dbc01, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12612, 1, 1),
      12612, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 4292, 1, 1),
      4292, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_pruned_model_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &gemm_22_output),
  &conv2d_1_layer, 0xaa1dbc01, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool forecast_temp_ml_model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_forecast_temp_ml_model_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_pruned_model_input0_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1140);
    serving_default_pruned_model_input0_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1140);
    conv2d_1_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1332);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1332);
    conv2d_1_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 44);
    conv2d_1_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 44);
    conv2d_4_pad_before_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    conv2d_4_pad_before_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 16);
    conv2d_4_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 668);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 668);
    conv2d_4_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1836);
    conv2d_4_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1836);
    pool_7_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    pool_7_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1560);
    conv2d_10_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2152);
    conv2d_10_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2152);
    conv2d_10_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1272);
    conv2d_10_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1272);
    conv2d_11_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 588);
    conv2d_11_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 588);
    conv2d_11_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    conv2d_11_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    conv2d_14_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3072);
    conv2d_14_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 3072);
    conv2d_14_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_14_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_15_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    conv2d_15_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 1536);
    conv2d_15_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2432);
    conv2d_15_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2432);
    pool_18_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    pool_18_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    pool_20_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    pool_20_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 768);
    gemm_21_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_21_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_21_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_21_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 832);
    gemm_22_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_22_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    gemm_22_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 368);
    gemm_22_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 368);
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
    conv2d_1_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 260);
    conv2d_1_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 260);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 312);
    conv2d_4_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 312);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 744);
    conv2d_4_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 744);
    conv2d_10_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 788);
    conv2d_10_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 788);
    conv2d_10_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 824);
    conv2d_10_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 824);
    conv2d_11_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 868);
    conv2d_11_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 868);
    conv2d_11_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1572);
    conv2d_11_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1572);
    conv2d_14_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1828);
    conv2d_14_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 1828);
    conv2d_14_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2020);
    conv2d_14_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2020);
    conv2d_15_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2276);
    conv2d_15_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 2276);
    conv2d_15_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 6372);
    conv2d_15_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 6372);
    gemm_21_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_21_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 6628);
    gemm_21_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 6628);
    gemm_21_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_21_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10724);
    gemm_21_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10724);
    gemm_22_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_22_weights_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10980);
    gemm_22_weights_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 10980);
    gemm_22_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_22_bias_array.data = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 12516);
    gemm_22_bias_array.data_start = AI_PTR(g_forecast_temp_ml_model_weights_map[0] + 12516);
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
      
      .n_macc            = 162451,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xaa1dbc01,
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
      
      .n_macc            = 162451,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xaa1dbc01,
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

