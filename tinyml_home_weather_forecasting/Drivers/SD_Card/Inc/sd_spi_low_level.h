#ifndef SD_SPI_LOW_LEVEL_H
#define SD_SPI_LOW_LEVEL_H

#include <stdint.h>

/* ===== SD Card SPI Commands ===== */
#define SD_CMD0     0       /* GO_IDLE_STATE */
#define SD_CMD1     1       /* SEND_OP_COND (MMC) */
#define SD_CMD8     8       /* SEND_IF_COND */
#define SD_CMD9     9       /* SEND_CSD */
#define SD_CMD10    10      /* SEND_CID */
#define SD_CMD12    12      /* STOP_TRANSMISSION */
#define SD_CMD16    16      /* SET_BLOCKLEN */
#define SD_CMD17    17      /* READ_SINGLE_BLOCK */
#define SD_CMD18    18      /* READ_MULTIPLE_BLOCK */
#define SD_CMD23    23      /* SET_BLOCK_COUNT (MMC) */
#define SD_CMD24    24      /* WRITE_BLOCK */
#define SD_CMD25    25      /* WRITE_MULTIPLE_BLOCK */
#define SD_CMD27    27      /* PROGRAM_CSD */
#define SD_CMD28    28      /* SET_WRITE_PROT */
#define SD_CMD29    29      /* CLR_WRITE_PROT */
#define SD_CMD30    30      /* SEND_WRITE_PROT */
#define SD_CMD32    32      /* ERASE_WR_BLK_START */
#define SD_CMD33    33      /* ERASE_WR_BLK_END */
#define SD_CMD38    38      /* ERASE */
#define SD_CMD41    41      /* SEND_OP_COND (SDC) */
#define SD_CMD55    55      /* APP_CMD */
#define SD_CMD58    58      /* READ_OCR */

/* ===== SD Card Types ===== */
#define CT_MMC      0x01    /* MMC ver 3 */
#define CT_SD1      0x02    /* SD ver 1 */
#define CT_SD2      0x04    /* SD ver 2 */
#define CT_SDC      (CT_SD1|CT_SD2)
#define CT_BLOCK    0x08    /* Block addressing */

/* ===== SD Card Response Tokens ===== */
#define SD_RESPONSE_NO_ERROR        0x00
#define SD_IN_IDLE_STATE            0x01
#define SD_ERASE_RESET              0x02
#define SD_ILLEGAL_COMMAND          0x04
#define SD_COM_CRC_ERROR            0x08
#define SD_ERASE_SEQUENCE_ERROR     0x10
#define SD_ADDRESS_ERROR            0x20
#define SD_PARAMETER_ERROR          0x40

/* ===== Data Start/Stop Tokens ===== */
#define SD_START_DATA_SINGLE_BLOCK_READ    0xFE
#define SD_START_DATA_MULTIPLE_BLOCK_READ  0xFE
#define SD_START_DATA_SINGLE_BLOCK_WRITE   0xFE
#define SD_START_DATA_MULTIPLE_BLOCK_WRITE 0xFC
#define SD_STOP_DATA_MULTIPLE_BLOCK_WRITE  0xFD

#ifdef __cplusplus
extern "C" {
#endif

/* Low-level SPI helpers */
void SDLL_SendByte(uint8_t data);
uint8_t SDLL_ReadByte(void);
void SDLL_CS_Low(void);
void SDLL_CS_High(void);
uint8_t SDLL_WaitReady(uint32_t timeout_ms);

/* SD protocol helpers (SPI mode) */
uint8_t SDLL_SendCommand(uint8_t cmd, uint32_t arg);
uint8_t SDLL_ReadDataBlock(uint8_t *buff, uint32_t btr);
uint8_t SDLL_WriteDataBlock(const uint8_t *buff, uint8_t token);

#ifdef __cplusplus
}
#endif

#endif /* SD_SPI_LOW_LEVEL_H */
