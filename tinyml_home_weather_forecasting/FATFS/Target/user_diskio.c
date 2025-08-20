/* USER CODE BEGIN Header */
/**
 ******************************************************************************
  * @file    user_diskio.c
  * @brief   This file includes a diskio driver skeleton to be completed by the user.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
 /* USER CODE END Header */

#ifdef USE_OBSOLETE_USER_CODE_SECTION_0
/*
 * Warning: the user section 0 is no more in use (starting from CubeMx version 4.16.0)
 * To be suppressed in the future.
 * Kept to ensure backward compatibility with previous CubeMx versions when
 * migrating projects.
 * User code previously added there should be copied in the new user sections before
 * the section contents can be deleted.
 */
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */
#endif

/* USER CODE BEGIN DECL */

/* Includes ------------------------------------------------------------------*/
#include <string.h>
#include <stdio.h>
#include "ff_gen_drv.h"
#include "main.h"

/* SD Card SPI Commands */
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

/* SD Card Types */
#define CT_MMC      0x01    /* MMC ver 3 */
#define CT_SD1      0x02    /* SD ver 1 */
#define CT_SD2      0x04    /* SD ver 2 */
#define CT_SDC      (CT_SD1|CT_SD2)
#define CT_BLOCK    0x08    /* Block addressing */

/* SD Card Response Tokens */
#define SD_RESPONSE_NO_ERROR        0x00
#define SD_IN_IDLE_STATE            0x01
#define SD_ERASE_RESET              0x02
#define SD_ILLEGAL_COMMAND          0x04
#define SD_COM_CRC_ERROR            0x08
#define SD_ERASE_SEQUENCE_ERROR     0x10
#define SD_ADDRESS_ERROR            0x20
#define SD_PARAMETER_ERROR          0x40

/* Data Start Tokens */
#define SD_START_DATA_SINGLE_BLOCK_READ    0xFE
#define SD_START_DATA_MULTIPLE_BLOCK_READ  0xFE
#define SD_START_DATA_SINGLE_BLOCK_WRITE   0xFE
#define SD_START_DATA_MULTIPLE_BLOCK_WRITE 0xFC

#define SD_STOP_DATA_MULTIPLE_BLOCK_WRITE  0xFD

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define SD_TIMEOUT          1000
#define SD_BLOCK_SIZE       512

// SPI handle and CS pin definitions - matching your main.c
extern SPI_HandleTypeDef hspi1;
#define SD_SPI_HANDLE       hspi1
#define SD_CS_PORT          SD_Card_CS_GPIO_Port
#define SD_CS_PIN           SD_Card_CS_Pin

/* Private variables ---------------------------------------------------------*/
static volatile DSTATUS Stat = STA_NOINIT;
static uint8_t CardType = 0;

/* USER CODE END DECL */

/* Private function prototypes -----------------------------------------------*/
DSTATUS USER_initialize (BYTE pdrv);
DSTATUS USER_status (BYTE pdrv);
DRESULT USER_read (BYTE pdrv, BYTE *buff, DWORD sector, UINT count);
#if _USE_WRITE == 1
  DRESULT USER_write (BYTE pdrv, const BYTE *buff, DWORD sector, UINT count);
#endif /* _USE_WRITE == 1 */
#if _USE_IOCTL == 1
  DRESULT USER_ioctl (BYTE pdrv, BYTE cmd, void *buff);
#endif /* _USE_IOCTL == 1 */

/* SPI SD Card Functions */
static void SD_SPI_SendByte(uint8_t data);
static uint8_t SD_SPI_ReadByte(void);
static void SD_CS_LOW(void);
static void SD_CS_HIGH(void);
static uint8_t SD_SendCommand(uint8_t cmd, uint32_t arg);
static uint8_t SD_WaitReady(void);
static uint8_t SD_ReadDataBlock(uint8_t *buff, uint32_t btr);
static uint8_t SD_WriteDataBlock(const uint8_t *buff, uint8_t token);

Diskio_drvTypeDef  USER_Driver =
{
  USER_initialize,
  USER_status,
  USER_read,
#if  _USE_WRITE
  USER_write,
#endif  /* _USE_WRITE == 1 */
#if  _USE_IOCTL == 1
  USER_ioctl,
#endif /* _USE_IOCTL == 1 */
};

/* Private functions ---------------------------------------------------------*/

/**
 * @brief  Send a byte via SPI
 */
static void SD_SPI_SendByte(uint8_t data)
{
  HAL_SPI_Transmit(&SD_SPI_HANDLE, &data, 1, SD_TIMEOUT);
}

/**
 * @brief  Read a byte via SPI
 */
static uint8_t SD_SPI_ReadByte(void)
{
  uint8_t data = 0xFF;
  HAL_SPI_TransmitReceive(&SD_SPI_HANDLE, &data, &data, 1, SD_TIMEOUT);
  return data;
}

/**
 * @brief  Set CS low
 */
static void SD_CS_LOW(void)
{
  HAL_GPIO_WritePin(SD_CS_PORT, SD_CS_PIN, GPIO_PIN_RESET);
}

/**
 * @brief  Set CS high
 */
static void SD_CS_HIGH(void)
{
  HAL_GPIO_WritePin(SD_CS_PORT, SD_CS_PIN, GPIO_PIN_SET);
}

/**
 * @brief  Wait for SD card ready
 */
static uint8_t SD_WaitReady(void)
{
  uint32_t timeout = HAL_GetTick() + SD_TIMEOUT;

  do {
    if (SD_SPI_ReadByte() == 0xFF) return 1;
  } while (HAL_GetTick() < timeout);

  return 0;
}

/**
 * @brief  Send command to SD card
 */
static uint8_t SD_SendCommand(uint8_t cmd, uint32_t arg)
{
  uint8_t n, res;

  if (cmd & 0x80) {
    cmd &= 0x7F;
    res = SD_SendCommand(SD_CMD55, 0);
    if (res > 1) return res;
  }

  /* Select the card and wait for ready */
  SD_CS_HIGH();
  SD_SPI_SendByte(0xFF);
  SD_CS_LOW();

  if (!SD_WaitReady()) return 0xFF;

  /* Send command packet */
  SD_SPI_SendByte(0x40 | cmd);           /* Start + Command index */
  SD_SPI_SendByte((uint8_t)(arg >> 24)); /* Argument[31..24] */
  SD_SPI_SendByte((uint8_t)(arg >> 16)); /* Argument[23..16] */
  SD_SPI_SendByte((uint8_t)(arg >> 8));  /* Argument[15..8] */
  SD_SPI_SendByte((uint8_t)arg);         /* Argument[7..0] */

  /* CRC */
  n = 0x01; /* Dummy CRC + Stop */
  if (cmd == SD_CMD0) n = 0x95; /* Valid CRC for CMD0(0) */
  if (cmd == SD_CMD8) n = 0x87; /* Valid CRC for CMD8(0x1AA) */
  SD_SPI_SendByte(n);

  /* Receive command response */
  if (cmd == SD_CMD12) SD_SPI_ReadByte(); /* Skip a stuff byte when stop reading */

  n = 10; /* Wait for a valid response in timeout of 10 attempts */
  do {
    res = SD_SPI_ReadByte();
  } while ((res & 0x80) && --n);

  return res; /* Return with the response value */
}

/**
 * @brief  Read data block from SD card
 */
static uint8_t SD_ReadDataBlock(uint8_t *buff, uint32_t btr)
{
  uint8_t token;
  uint32_t timeout = HAL_GetTick() + SD_TIMEOUT;

  /* Wait for data packet in timeout of 200ms */
  do {
    token = SD_SPI_ReadByte();
  } while ((token == 0xFF) && (HAL_GetTick() < timeout));

  if (token != 0xFE) return 0; /* If not valid data token, return with error */

  /* Receive the data block into buffer */
  do {
    *buff++ = SD_SPI_ReadByte();
  } while (--btr);

  SD_SPI_ReadByte(); /* Discard CRC */
  SD_SPI_ReadByte();

  return 1; /* Return with success */
}

/**
 * @brief  Write data block to SD card
 */
#if _USE_WRITE == 1
static uint8_t SD_WriteDataBlock(const uint8_t *buff, uint8_t token)
{
  uint8_t resp;
  uint32_t bc = SD_BLOCK_SIZE;

  if (!SD_WaitReady()) return 0;

  SD_SPI_SendByte(token); /* Send token */

  if (token != 0xFD) { /* Is data token */
    do { /* Send the data block to the MMC */
      SD_SPI_SendByte(*buff++);
    } while (--bc);

    SD_SPI_SendByte(0xFF); /* CRC (Dummy) */
    SD_SPI_SendByte(0xFF);

    resp = SD_SPI_ReadByte(); /* Receive data response */
    if ((resp & 0x1F) != 0x05) /* If not accepted, return with error */
      return 0;
  }

  return 1;
}
#endif

/**
  * @brief  Initializes a Drive
  * @param  pdrv: Physical drive number (0..)
  * @retval DSTATUS: Operation status
  */
DSTATUS USER_initialize (
	BYTE pdrv           /* Physical drive nmuber to identify the drive */
)
{
  /* USER CODE BEGIN INIT */
  uint8_t n, cmd, ty, ocr[4];
  uint32_t timeout;

  if (pdrv) return STA_NOINIT; /* Supports only single drive */

  /* Initialize SPI */
  SD_CS_HIGH();

  /* Send 80 dummy clocks */
  for (n = 0; n < 10; n++) SD_SPI_SendByte(0xFF);

  ty = 0;
  if (SD_SendCommand(SD_CMD0, 0) == 1) { /* Enter Idle state */
    timeout = HAL_GetTick() + 1000; /* Initialization timeout of 1000 msec */

    if (SD_SendCommand(SD_CMD8, 0x1AA) == 1) { /* SDv2? */
      /* Get trailing return value of R7 resp */
      for (n = 0; n < 4; n++) ocr[n] = SD_SPI_ReadByte();

      if (ocr[2] == 0x01 && ocr[3] == 0xAA) { /* The card can work at vdd range of 2.7-3.6V */
        /* Wait for leaving idle state (ACMD41 with HCS bit) */
        while ((HAL_GetTick() < timeout) && SD_SendCommand(SD_CMD41 | 0x80, 1UL << 30));

        if ((HAL_GetTick() < timeout) && SD_SendCommand(SD_CMD58, 0) == 0) { /* Check CCS bit in the OCR */
          for (n = 0; n < 4; n++) ocr[n] = SD_SPI_ReadByte();
          ty = (ocr[0] & 0x40) ? CT_SD2 | CT_BLOCK : CT_SD2; /* SDv2 */
        }
      }
    } else { /* SDv1 or MMCv3 */
      if (SD_SendCommand(SD_CMD41 | 0x80, 0) <= 1) {
        ty = CT_SD1; cmd = SD_CMD41 | 0x80; /* SDv1 */
      } else {
        ty = CT_MMC; cmd = SD_CMD1; /* MMCv3 */
      }

      /* Wait for leaving idle state */
      while ((HAL_GetTick() < timeout) && SD_SendCommand(cmd, 0));

      /* Set R/W block length to 512 */
      if ((HAL_GetTick() >= timeout) || SD_SendCommand(SD_CMD16, 512) != 0) ty = 0;
    }
  }

  CardType = ty;
  SD_CS_HIGH();
  SD_SPI_SendByte(0xFF);

  if (ty) { /* Initialization succeeded */
    Stat &= ~STA_NOINIT; /* Clear STA_NOINIT */
    //scale up the SPI speed
    extern SPI_HandleTypeDef hspi1;
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_64;
    HAL_SPI_Init(&hspi1);
  } else {
    Stat = STA_NOINIT;
  }

  return Stat;
  /* USER CODE END INIT */
}

/**
  * @brief  Gets Disk Status
  * @param  pdrv: Physical drive number (0..)
  * @retval DSTATUS: Operation status
  */
DSTATUS USER_status (
	BYTE pdrv       /* Physical drive number to identify the drive */
)
{
  /* USER CODE BEGIN STATUS */
  if (pdrv) return STA_NOINIT; /* Supports only single drive */
  return Stat;
  /* USER CODE END STATUS */
}

/**
  * @brief  Reads Sector(s)
  * @param  pdrv: Physical drive number (0..)
  * @param  *buff: Data buffer to store read data
  * @param  sector: Sector address (LBA)
  * @param  count: Number of sectors to read (1..128)
  * @retval DRESULT: Operation result
  */
DRESULT USER_read (
	BYTE pdrv,      /* Physical drive nmuber to identify the drive */
	BYTE *buff,     /* Data buffer to store read data */
	DWORD sector,   /* Sector address in LBA */
	UINT count      /* Number of sectors to read */
)
{
  /* USER CODE BEGIN READ */
  if (pdrv || !count) return RES_PARERR;
  if (Stat & STA_NOINIT) return RES_NOTRDY;

  if (!(CardType & CT_BLOCK)) sector *= 512; /* Convert to byte address if needed */

  if (count == 1) { /* Single block read */
    if ((SD_SendCommand(SD_CMD17, sector) == 0) && SD_ReadDataBlock(buff, 512))
      count = 0;
  } else { /* Multiple block read */
    if (SD_SendCommand(SD_CMD18, sector) == 0) {
      do {
        if (!SD_ReadDataBlock(buff, 512)) break;
        buff += 512;
      } while (--count);
      SD_SendCommand(SD_CMD12, 0); /* STOP_TRANSMISSION */
    }
  }

  SD_CS_HIGH();
  SD_SPI_SendByte(0xFF);

  return count ? RES_ERROR : RES_OK;
  /* USER CODE END READ */
}

/**
  * @brief  Writes Sector(s)
  * @param  pdrv: Physical drive number (0..)
  * @param  *buff: Data to be written
  * @param  sector: Sector address (LBA)
  * @param  count: Number of sectors to write (1..128)
  * @retval DRESULT: Operation result
  */
#if _USE_WRITE == 1
DRESULT USER_write (
	BYTE pdrv,          /* Physical drive nmuber to identify the drive */
	const BYTE *buff,   /* Data to be written */
	DWORD sector,       /* Sector address in LBA */
	UINT count          /* Number of sectors to write */
)
{
  /* USER CODE BEGIN WRITE */
  if (pdrv || !count) return RES_PARERR;
  if (Stat & STA_NOINIT) return RES_NOTRDY;
  if (Stat & STA_PROTECT) return RES_WRPRT;

  if (!(CardType & CT_BLOCK)) sector *= 512; /* Convert to byte address if needed */

  if (count == 1) { /* Single block write */
    if ((SD_SendCommand(SD_CMD24, sector) == 0) && SD_WriteDataBlock(buff, 0xFE))
      count = 0;
  } else { /* Multiple block write */
    if (CardType & CT_SDC) SD_SendCommand(SD_CMD23 | 0x80, count);
    if (SD_SendCommand(SD_CMD25, sector) == 0) {
      do {
        if (!SD_WriteDataBlock(buff, 0xFC)) break;
        buff += 512;
      } while (--count);
      if (!SD_WriteDataBlock(0, 0xFD)) count = 1; /* STOP_TRAN token */
    }
  }

  SD_CS_HIGH();
  SD_SPI_SendByte(0xFF);

  return count ? RES_ERROR : RES_OK;
  /* USER CODE END WRITE */
}
#endif /* _USE_WRITE == 1 */

/**
  * @brief  I/O control operation
  * @param  pdrv: Physical drive number (0..)
  * @param  cmd: Control code
  * @param  *buff: Buffer to send/receive control data
  * @retval DRESULT: Operation result
  */
#if _USE_IOCTL == 1
DRESULT USER_ioctl (
	BYTE pdrv,      /* Physical drive nmuber (0..) */
	BYTE cmd,       /* Control code */
	void *buff      /* Buffer to send/receive control data */
)
{
  /* USER CODE BEGIN IOCTL */
  DRESULT res = RES_ERROR;
  uint8_t n, csd[16];
  DWORD cs;

  if (pdrv) return RES_PARERR;
  if (Stat & STA_NOINIT) return RES_NOTRDY;

  switch (cmd) {
    case CTRL_SYNC: /* Make sure that no pending write process */
      SD_CS_LOW();
      if (SD_WaitReady()) res = RES_OK;
      SD_CS_HIGH();
      break;

    case GET_SECTOR_COUNT: /* Get number of sectors on the disk (DWORD) */
      if ((SD_SendCommand(SD_CMD9, 0) == 0) && SD_ReadDataBlock(csd, 16)) {
        if ((csd[0] >> 6) == 1) { /* SDC ver 2.00 */
          cs = csd[9] + ((DWORD)csd[8] << 8) + ((DWORD)(csd[7] & 63) << 16) + 1;
          *(DWORD*)buff = cs << 10;
        } else { /* SDC ver 1.XX or MMC */
          n = (csd[5] & 15) + ((csd[10] & 128) >> 7) + ((csd[9] & 3) << 1) + 2;
          cs = (csd[8] >> 6) + ((DWORD)csd[7] << 2) + ((DWORD)(csd[6] & 3) << 10) + 1;
          *(DWORD*)buff = cs << (n - 9);
        }
        res = RES_OK;
      }
      SD_CS_HIGH();
      SD_SPI_SendByte(0xFF);
      break;

    case GET_SECTOR_SIZE: /* Get R/W sector size (WORD) */
      *(WORD*)buff = 512;
      res = RES_OK;
      break;

    case GET_BLOCK_SIZE: /* Get erase block size in unit of sector (DWORD) */
      *(DWORD*)buff = 128;
      res = RES_OK;
      break;

    default:
      res = RES_PARERR;
  }

  return res;
  /* USER CODE END IOCTL */
}
#endif /* _USE_IOCTL == 1 */
