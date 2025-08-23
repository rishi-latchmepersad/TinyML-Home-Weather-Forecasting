/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    user_diskio.c
 * @brief   FatFs Disk I/O driver (SPI SD card) with LL split-out.
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
#include "diskio.h"
#include "main.h"
#include "sd_spi_low_level.h"   /* Low-level SPI/SD helpers (kept outside CubeMX) */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define SD_BLOCK_SIZE  512

/* Private variables ---------------------------------------------------------*/
/* Disk status */
static volatile DSTATUS Stat = STA_NOINIT;
/* Card type flags (CT_SD1/CT_SD2/CT_BLOCK ...) filled during init */
static uint8_t CardType = 0;

/* SPI handle used by the SD */
extern SPI_HandleTypeDef hspi1;

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
  * @brief  Initializes a Drive
  * @param  pdrv: Physical drive number (0..)
  * @retval DSTATUS: Operation status
  */
DSTATUS USER_initialize (
	BYTE pdrv           /* Physical drive nmuber to identify the drive */
)
{
  /* USER CODE BEGIN INIT */
	uint8_t n, ty = 0U, ocr[4];
	uint32_t timeout;
	uint8_t r1;

	/* 1) Power-on SPI handshake before ANY command */
	sd_spi_power_on_sequence();

	/* 2) Now try CMD0 (go idle) with proper select/CRC inside SDLL_SendCommand */
	r1 = SDLL_SendCommand(SD_CMD0, 0UL);
	printf("CMD0(after handshake) R1=0x%02X\r\n", r1);

	if (r1 == 1U)  /* Card entered Idle state */
	{
	    timeout = HAL_GetTick() + 1000U;

	    /* 3) Probe SDv2 with CMD8 */
	    r1 = SDLL_SendCommand(SD_CMD8, 0x1AAUL);
	    printf("CMD8 R1=0x%02X\r\n", r1);

	    if (r1 == 1U)
	    {
	        for (n = 0U; n < 4U; n++) { ocr[n] = SDLL_ReadByte(); }
	        if ((ocr[2] == 0x01U) && (ocr[3] == 0xAAU))
	        {
	            while ((HAL_GetTick() < timeout) &&
	                   SDLL_SendCommand(SD_CMD41 | 0x80U, 1UL << 30)) { /* ACMD41(HCS) */ }
	            if ((HAL_GetTick() < timeout) && (SDLL_SendCommand(SD_CMD58, 0UL) == 0U))
	            {
	                for (n = 0U; n < 4U; n++) { ocr[n] = SDLL_ReadByte(); }
	                ty = (ocr[0] & 0x40U) ? (CT_SD2 | CT_BLOCK) : CT_SD2;
	            }
	        }
	    }
	    else  /* SDv1/MMC path */
	    {
	        uint8_t cmd;
	        if (SDLL_SendCommand(SD_CMD41 | 0x80U, 0UL) <= 1U) { ty = CT_SD1; cmd = SD_CMD41 | 0x80U; }
	        else                                               { ty = CT_MMC; cmd = SD_CMD1;        }

	        while ((HAL_GetTick() < timeout) && SDLL_SendCommand(cmd, 0UL)) { /* wait ready */ }
	        if ((HAL_GetTick() >= timeout) || (SDLL_SendCommand(SD_CMD16, 512U) != 0U)) { ty = 0U; }
	    }
	}

	/* 4) Finish up */
	CardType = ty;
	SDLL_CS_High();
	SDLL_SendByte(0xFF);

	Stat = (ty != 0U) ? (Stat & (DSTATUS)~STA_NOINIT) : STA_NOINIT;
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
  if (pdrv) return STA_NOINIT;
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

  if (!(CardType & CT_BLOCK)) sector *= 512; /* SDSC byte addressing */

  if (count == 1) { /* Single block */
    if ((SDLL_SendCommand(SD_CMD17, sector) == 0) &&
         SDLL_ReadDataBlock(buff, SD_BLOCK_SIZE))
      count = 0;
  } else {          /* Multi-block as repeated singles (robust fallback) */
    do {
      if (SDLL_SendCommand(SD_CMD17, sector) != 0x00 ||
          !SDLL_ReadDataBlock(buff, SD_BLOCK_SIZE)) {
        SDLL_CS_High();
        (void)SDLL_ReadByte(); /* trailing dummy */
        return RES_ERROR;
      }
      sector++;
      buff += SD_BLOCK_SIZE;
    } while (--count);
  }

  SDLL_CS_High();
  SDLL_SendByte(0xFF);

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

  if (!(CardType & CT_BLOCK)) sector *= 512;

  if (count == 1) { /* Single block */
    if ((SDLL_SendCommand(SD_CMD24, sector) == 0) &&
         SDLL_WriteDataBlock(buff, SD_START_DATA_SINGLE_BLOCK_WRITE))
      count = 0;
  } else {
    if (CardType & CT_SDC) SDLL_SendCommand(SD_CMD23 | 0x80, count);
    if (SDLL_SendCommand(SD_CMD25, sector) == 0) {
      do {
        if (!SDLL_WriteDataBlock(buff, SD_START_DATA_MULTIPLE_BLOCK_WRITE))
          break;
        buff += SD_BLOCK_SIZE;
      } while (--count);
      if (!SDLL_WriteDataBlock(0, SD_STOP_DATA_MULTIPLE_BLOCK_WRITE))
        count = 1; /* STOP_TRAN token */
    }
  }

  SDLL_CS_High();
  SDLL_SendByte(0xFF);

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
    case CTRL_SYNC:
      SDLL_CS_Low();
      (void)SDLL_ReadByte();                 /* prime the clock */
      if (SDLL_WaitReady(200)) res = RES_OK; /* bounded wait */
      SDLL_CS_High();
      (void)SDLL_ReadByte();                 /* trailing dummy */
      break;

    case GET_SECTOR_COUNT:
      if ((SDLL_SendCommand(SD_CMD9, 0) == 0) && SDLL_ReadDataBlock(csd, 16)) {
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
      SDLL_CS_High();
      SDLL_SendByte(0xFF);
      break;

    case GET_SECTOR_SIZE:
      *(WORD*)buff = 512;
      res = RES_OK;
      break;

    case GET_BLOCK_SIZE:
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

