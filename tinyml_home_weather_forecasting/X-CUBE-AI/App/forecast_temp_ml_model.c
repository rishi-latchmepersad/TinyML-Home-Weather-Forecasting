/**
  ******************************************************************************
  * @file    forecast_temp_ml_model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-11-13T20:24:43-0400
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
#define AI_FORECAST_TEMP_ML_MODEL_MODEL_SIGNATURE     "0xfa21c5fcefa60d300b8a8336c3abb183"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-11-13T20:24:43-0400"

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
  NULL, NULL, 336, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2592, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2700, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  pool_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

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
    AI_PACK_INTQ_SCALE(0.06286109238862991f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_10_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001610324252396822f, 0.0013206690782681108f, 0.001474682125262916f, 0.0007813265547156334f, 0.0015687881968915462f, 0.001216310658492148f, 0.002050027484074235f, 0.001684420509263873f, 0.0022786089684814215f, 0.001167275127954781f, 0.0021886832546442747f, 0.0021059499122202396f, 0.0022690980695188046f, 0.0014147902838885784f, 0.001992434961721301f, 0.0007685531163588166f, 0.0015758706722408533f, 0.0017709742533043027f, 0.001569952699355781f, 0.0014834830071777105f, 0.0017367781838402152f, 0.0010873731225728989f, 0.001296790549531579f, 0.0014407394919544458f, 0.0017905213171616197f, 0.0016208565793931484f, 0.001175277167931199f, 0.0011512048076838255f, 0.0012538861483335495f, 0.002082499908283353f, 0.0018070628866553307f, 0.001704657101072371f, 0.0018828343600034714f, 0.0018566236831247807f, 0.0013843677006661892f, 0.0013411755207926035f, 0.001292628701776266f, 0.0017899235244840384f, 0.0022164040710777044f, 0.001332530751824379f, 0.0010728113120421767f, 0.0014598398702219129f, 0.000869181239977479f, 0.0009715876658447087f, 0.0016364381881430745f, 0.0018155615543946624f, 0.001594614703208208f, 0.00124964804854244f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0904446616768837f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022251561749726534f, 0.0021199050825089216f, 0.002171703614294529f, 0.0030970752704888582f, 0.002353289630264044f, 0.0025280779227614403f, 0.0025714512448757887f, 0.0023442429956048727f, 0.00228260294534266f, 0.0022248297464102507f, 0.002171391388401389f, 0.0025575694162398577f, 0.0025134384632110596f, 0.0021787642035633326f, 0.00228382577188313f, 0.0023473852779716253f, 0.0021598567254841328f, 0.0021413357462733984f, 0.0022334728855639696f, 0.002140078227967024f, 0.0025182298850268126f, 0.002569033997133374f, 0.0026711532846093178f, 0.0019214091589674354f, 0.002750878455117345f, 0.002589874668046832f, 0.001934422180056572f, 0.0023766877129673958f, 0.0026168590411543846f, 0.0021488452330231667f, 0.0024001169949769974f, 0.0024465867318212986f, 0.0021306064445525408f, 0.0022958521731197834f, 0.0023066136054694653f, 0.0022367278579622507f, 0.0021569987293332815f, 0.0026607480831444263f, 0.0024522957392036915f, 0.0021726512350142f, 0.002199584385380149f, 0.0022776536643505096f, 0.0024798561353236437f, 0.00215447717346251f, 0.002174391644075513f, 0.0030818218365311623f, 0.0024783415719866753f, 0.0023545478470623493f, 0.0021882851142436266f, 0.0022913573775440454f, 0.0021695701871067286f, 0.002152257366105914f, 0.0021009566262364388f, 0.0020694201812148094f, 0.0021102381870150566f, 0.002502650488168001f, 0.0026774138677865267f, 0.002100057667121291f, 0.002706455299630761f, 0.0023161780554801226f, 0.0028533164877444506f, 0.0024289730936288834f, 0.0022222266998142004f, 0.0021126132924109697f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02816382423043251f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 54,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006815995555371046f, 0.0020153354853391647f, 0.002265859628096223f, 0.002864051377400756f, 0.0076088919304311275f, 0.003425048664212227f, 0.0016803659964352846f, 0.001597003429196775f, 0.0033136059064418077f, 0.003752576420083642f, 0.0042176018469035625f, 0.006093797739595175f, 0.0030244910158216953f, 0.0029064868576824665f, 0.005307332146912813f, 0.005058306269347668f, 0.003103346796706319f, 0.005167544353753328f, 0.004334470257163048f, 0.0033887375611811876f, 0.004551227204501629f, 0.002998662879690528f, 0.006244204938411713f, 0.004903621971607208f, 0.0031099659390747547f, 0.0024346746504306793f, 0.005937632638961077f, 0.0037452203687280416f, 0.0020009137224406004f, 0.01205257698893547f, 0.004877152387052774f, 0.003867506980895996f, 0.0020166325848549604f, 0.002539342502132058f, 0.004542468581348658f, 0.002774043707177043f, 0.007952091284096241f, 0.0037902037147432566f, 0.004288212396204472f, 0.0027269460260868073f, 0.004862593486905098f, 0.005594425369054079f, 0.002604789799079299f, 0.0027944885659962893f, 0.0025553624145686626f, 0.0026440382935106754f, 0.0028603868559002876f, 0.002832240192219615f, 0.005014623515307903f, 0.0021520955488085747f, 0.006195145659148693f, 0.0014236135175451636f, 0.0032385268714278936f, 0.0025913496501743793f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.093657486140728f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02816382423043251f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0031755261588841677f, 0.004577514715492725f, 0.003272157395258546f, 0.00328246783465147f, 0.0032414377201348543f, 0.001454789424315095f, 0.0056453547440469265f, 0.0031804435420781374f, 0.005897053517401218f, 0.0014479989185929298f, 0.0034531268756836653f, 0.007034053094685078f, 0.005199828185141087f, 0.003835981246083975f, 0.0043050367385149f, 0.0033233577851206064f, 0.003784841625019908f, 0.0033964894246309996f, 0.0034032617695629597f, 0.003339516231790185f, 0.003075492801144719f, 0.0038209925405681133f, 0.0031254638452082872f, 0.0033225747756659985f, 0.0015033544041216373f, 0.003475983627140522f, 0.00384469679556787f, 0.0014571554493159056f, 0.0035255951806902885f, 0.005067729856818914f, 0.001445579226128757f, 0.0033601114992052317f, 0.003332603955641389f, 0.0038173324428498745f, 0.003005459439009428f, 0.002871585777029395f, 0.003488613059744239f, 0.004116932395845652f, 0.0047901771031320095f, 0.0028314620722085238f, 0.0014428931754082441f, 0.0035865711979568005f, 0.0014552214415743947f, 0.002694352064281702f, 0.004130541812628508f, 0.001789393019862473f, 0.0030633003916591406f, 0.00162739516235888f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.150478333234787f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0029172226786613464f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.041906002908945084f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.093657486140728f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_pruned_model_input0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.043063752353191376f),
    AI_PACK_INTQ_ZP(-8)))

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
  AI_SHAPE_INIT(4, 1, 48, 24, 1), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
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
  AI_SHAPE_INIT(4, 1, 64, 24, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1536),
  1, &conv2d_11_output_array, &conv2d_11_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output0, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 24), AI_STRIDE_INIT(4, 1, 1, 64, 64),
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
  AI_SHAPE_INIT(4, 1, 54, 48, 1), AI_STRIDE_INIT(4, 1, 1, 54, 2592),
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
  AI_SHAPE_INIT(4, 1, 48, 48, 1), AI_STRIDE_INIT(4, 1, 1, 48, 2304),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_pad_before_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 54, 50, 1), AI_STRIDE_INIT(4, 1, 1, 54, 2700),
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
  AI_SHAPE_INIT(4, 1, 48, 24, 1), AI_STRIDE_INIT(4, 1, 1, 48, 1152),
  1, &pool_7_output_array, &pool_7_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_pruned_model_input0_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 7, 1, 48), AI_STRIDE_INIT(4, 1, 1, 7, 7),
  1, &serving_default_pruned_model_input0_output_array, &serving_default_pruned_model_input0_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_pruned_model_input0_output0, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 7, 48, 1), AI_STRIDE_INIT(4, 1, 1, 7, 336),
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
  .pool_size = AI_SHAPE_2D_INIT(1, 24), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 24), 
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
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 13808, 1, 1),
    13808, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 11444, 1, 1),
    11444, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_pruned_model_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &gemm_14_output),
  &conv2d_1_layer, 0x092709d7, NULL)

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
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 11444, 1, 1),
      11444, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_IN_NUM, &serving_default_pruned_model_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_FORECAST_TEMP_ML_MODEL_OUT_NUM, &gemm_14_output),
  &conv2d_1_layer, 0x092709d7, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool forecast_temp_ml_model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_forecast_temp_ml_model_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_pruned_model_input0_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2304);
    serving_default_pruned_model_input0_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2304);
    conv2d_1_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2640);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2640);
    conv2d_1_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 7316);
    conv2d_1_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 7316);
    conv2d_4_pad_before_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2304);
    conv2d_4_pad_before_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2304);
    conv2d_4_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 5004);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 5004);
    conv2d_4_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_4_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    pool_7_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2304);
    pool_7_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2304);
    conv2d_10_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_10_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_10_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 916);
    conv2d_10_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 916);
    conv2d_11_scratch0_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_11_scratch0_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 0);
    conv2d_11_output_array.data = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2068);
    conv2d_11_output_array.data_start = AI_PTR(g_forecast_temp_ml_model_activations_map[0] + 2068);
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
      
      .n_macc            = 545271,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x092709d7,
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
      
      .n_macc            = 545271,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x092709d7,
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

