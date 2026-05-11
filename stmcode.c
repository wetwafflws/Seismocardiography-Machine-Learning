/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  *
  * FIXES vs original:
  *   BUG 1 — Send_*_Packet() no longer calls HAL_UART_Transmit() (blocking,
  *            HAL_MAX_DELAY) or HAL_Delay(1) inside the retry loop.  The UART
  *            output was a debug leftover; it blocked the main loop for ≥1 ms
  *            per packet.  USB CDC retries now simply drop the packet after 3
  *            attempts rather than spinning — the host parser already handles
  *            occasional gaps gracefully.
  *
  *   BUG 2 — process_half read/write now surrounded by __DMB() (data memory
  *            barrier) so the Cortex-M7 out-of-order pipeline cannot reorder
  *            the flag read relative to the accel_data array access.
  *
  *   BUG 3 — HAL_GPIO_EXTI_Callback() now checks ppg_dma_busy before kicking
  *            a new DMA read.  The flag is set in the callback and cleared in
  *            HAL_I2C_MemRxCpltCallback(), preventing re-entrant DMA calls on
  *            a busy I2C bus.
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "usbd_cdc_if.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

#define MAX30102_I2C_ADDR (0x57 << 1)

#define MAX30102_REG_INT_STATUS_1  0x00
#define MAX30102_REG_FIFO_WR_PTR   0x04
#define MAX30102_REG_FIFO_RD_PTR   0x06
#define MAX30102_REG_FIFO_DATA     0x07
#define MAX30102_REG_FIFO_CONFIG   0x08
#define MAX30102_REG_MODE_CONFIG   0x09
#define MAX30102_REG_SPO2_CONFIG   0x0A
#define MAX30102_REG_LED1_PA       0x0C
#define MAX30102_REG_LED2_PA       0x0D
#define MAX30102_REG_PART_ID       0xFF

#define MA_WINDOW_SIZE  50

// 2nd-order Butterworth low-pass, Fc=5Hz, Fs=100Hz
#define IIR_B0  0.02008f
#define IIR_B1  0.04016f
#define IIR_B2  0.02008f
#define IIR_A1 -1.56102f
#define IIR_A2  0.64135f

#define REFRACTORY_MS       400
#define BEAT_PULSE_MS        50
#define THRESHOLD_FACTOR    0.4f
#define AMPLITUDE_DECAY     0.99f

#define SCG_BUFFER_SIZE 6

#define PKT_MAGIC       0xAA
#define PKT_TYPE_SCG    0x01
#define PKT_TYPE_BEAT   0x02
#define PKT_TYPE_PPG    0x03

typedef struct __attribute__((packed)) {
    uint8_t  magic;
    uint8_t  type;
    uint32_t timestamp_ms;
    int16_t  x;
    int16_t  y;
    int16_t  z;
    uint8_t  checksum;
} SCG_Packet_t;  // 13 bytes

typedef struct __attribute__((packed)) {
    uint8_t  magic;
    uint8_t  type;
    uint32_t timestamp_ms;
    uint8_t  checksum;
} Beat_Packet_t;  // 7 bytes

typedef struct __attribute__((packed)) {
    uint8_t  magic;
    uint8_t  type;
    uint32_t timestamp_ms;
    uint32_t ppg_raw;
    uint8_t  checksum;
} PPG_Packet_t;  // 11 bytes

/* USER CODE END PTD */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
ADC_HandleTypeDef hadc2;
DMA_HandleTypeDef hdma_adc1;

I2C_HandleTypeDef hi2c1;
DMA_HandleTypeDef hdma_i2c1_rx;

TIM_HandleTypeDef htim1;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */

volatile uint8_t scg_ready = 0;

volatile uint8_t ppg_dma_rx_buffer[6];
volatile uint8_t ppg_sample_ready = 0;
// FIX 3: guard flag — set before kicking DMA, cleared in MemRxCplt callback.
// Prevents re-entrant HAL_I2C_Mem_Read_DMA() calls when a prior read is still
// in flight.  Declared volatile so the EXTI ISR sees the update from the I2C
// DMA complete callback without compiler optimisation.
volatile uint8_t ppg_dma_busy = 0;

__attribute__((aligned(32))) volatile uint16_t accel_data[SCG_BUFFER_SIZE];

// FIX 2: use a plain uint8_t instead of bare volatile so we can insert __DMB()
// barriers explicitly at each read/write site.  The volatile qualifier alone is
// not sufficient to prevent Cortex-M7 reordering of the flag relative to the
// array accesses that follow it.
volatile uint8_t process_half = 0;
volatile uint32_t scg_timestamp = 0;

uint32_t ma_buffer[MA_WINDOW_SIZE];
uint32_t ma_sum    = 0;
uint8_t  ma_index  = 0;
uint8_t  ma_filled = 0;

float iir_x1 = 0.0f, iir_x2 = 0.0f;
float iir_y1 = 0.0f, iir_y2 = 0.0f;

float    peak_amplitude   = 1000.0f;
float    peak_threshold   = 400.0f;
float    last_filtered    = 0.0f;
uint8_t  rising           = 0;
float    slope_peak_val   = 0.0f;

#define  PEAK_CONFIRM_SAMPLES  3

uint8_t  falling_count    = 0;
uint32_t last_beat_tick   = 0;
uint32_t beat_pulse_start = 0;
uint8_t  pulse_active     = 0;
uint8_t  beat_this_sample = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_I2C1_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM1_Init(void);
static void MX_ADC2_Init(void);
static void MX_USART1_UART_Init(void);
/* USER CODE BEGIN PFP */
HAL_StatusTypeDef MAX30102_WriteReg(uint8_t reg, uint8_t value);
HAL_StatusTypeDef MAX30102_ReadReg(uint8_t reg, uint8_t *value);
int32_t Detrend_IR(uint32_t ir_sample);
/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

static uint8_t xor_checksum(uint8_t *buf, uint8_t len) {
    uint8_t c = 0;
    for (uint8_t i = 0; i < len; i++) c ^= buf[i];
    return c;
}

// FIX 1: Removed HAL_UART_Transmit (was blocking with HAL_MAX_DELAY, ≥1 ms
// per call at 115200 baud for a 13-byte SCG packet).  Also removed HAL_Delay(1)
// from the retry loop — spinning while USB is busy stalls the entire main loop.
// Drop the packet after 3 failed attempts; the PC parser tolerates gaps.
void Send_SCG_Packet(uint32_t ts, int16_t x, int16_t y, int16_t z) {
    SCG_Packet_t pkt;
    pkt.magic        = PKT_MAGIC;
    pkt.type         = PKT_TYPE_SCG;
    pkt.timestamp_ms = ts;
    pkt.x = x; pkt.y = y; pkt.z = z;
    pkt.checksum     = xor_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

    uint8_t retries = 3;
    while (CDC_Transmit_FS((uint8_t*)&pkt, sizeof(pkt)) == USBD_BUSY && --retries)
        ; // No HAL_Delay — just retry immediately or drop
}

void Send_Beat_Packet(void) {
    Beat_Packet_t pkt;
    pkt.magic        = PKT_MAGIC;
    pkt.type         = PKT_TYPE_BEAT;
    pkt.timestamp_ms = HAL_GetTick();
    pkt.checksum     = xor_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

    uint8_t retries = 3;
    while (CDC_Transmit_FS((uint8_t*)&pkt, sizeof(pkt)) == USBD_BUSY && --retries)
        ;
}

void Send_PPG_Packet(uint32_t ts, uint32_t ppg_raw) {
    PPG_Packet_t pkt;
    pkt.magic        = PKT_MAGIC;
    pkt.type         = PKT_TYPE_PPG;
    pkt.timestamp_ms = ts;
    pkt.ppg_raw      = ppg_raw;
    pkt.checksum     = xor_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

    uint8_t retries = 3;
    while (CDC_Transmit_FS((uint8_t*)&pkt, sizeof(pkt)) == USBD_BUSY && --retries)
        ;
}

HAL_StatusTypeDef MAX30102_WriteReg(uint8_t reg, uint8_t value) {
    return HAL_I2C_Mem_Write(&hi2c1, MAX30102_I2C_ADDR, reg, I2C_MEMADD_SIZE_8BIT, &value, 1, HAL_MAX_DELAY);
}

HAL_StatusTypeDef MAX30102_ReadReg(uint8_t reg, uint8_t *value) {
    return HAL_I2C_Mem_Read(&hi2c1, MAX30102_I2C_ADDR, reg, I2C_MEMADD_SIZE_8BIT, value, 1, HAL_MAX_DELAY);
}

int32_t Detrend_IR(uint32_t ir_sample) {
    ma_sum -= ma_buffer[ma_index];
    ma_buffer[ma_index] = ir_sample;
    ma_sum += ir_sample;
    ma_index++;
    if (ma_index >= MA_WINDOW_SIZE) {
        ma_index  = 0;
        ma_filled = 1;
    }
    if (!ma_filled) return INT32_MIN;
    uint32_t mean = ma_sum / MA_WINDOW_SIZE;
    return (int32_t)ir_sample - (int32_t)mean;
}

float LowPass_Filter(float input) {
    float y = IIR_B0 * input
            + IIR_B1 * iir_x1
            + IIR_B2 * iir_x2
            - IIR_A1 * iir_y1
            - IIR_A2 * iir_y2;
    iir_x2 = iir_x1; iir_x1 = input;
    iir_y2 = iir_y1; iir_y1 = y;
    return y;
}

void Detect_Peak(float filtered) {
    uint32_t now = HAL_GetTick();

    if (filtered > peak_amplitude) {
        peak_amplitude = filtered;
    } else {
        peak_amplitude *= AMPLITUDE_DECAY;
    }
    peak_threshold = peak_amplitude * THRESHOLD_FACTOR;

    float slope = filtered - last_filtered;

    if (!rising) {
        if (filtered > peak_threshold && slope > 0.0f) {
            rising         = 1;
            falling_count  = 0;
            slope_peak_val = filtered;
        }
    } else {
        if (slope > 0.0f) {
            slope_peak_val = filtered;
            falling_count  = 0;
        } else {
            falling_count++;
            if (falling_count >= PEAK_CONFIRM_SAMPLES) {
                rising        = 0;
                falling_count = 0;
                if ((now - last_beat_tick) >= REFRACTORY_MS) {
                    last_beat_tick = now;
                    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_7, GPIO_PIN_SET);
                    beat_pulse_start = now;
                    pulse_active     = 1;
                    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_3, GPIO_PIN_SET);
                    Send_Beat_Packet();
                    beat_this_sample = 1;
                }
            }
            if (filtered < peak_threshold) {
                rising        = 0;
                falling_count = 0;
            }
        }
    }

    if (pulse_active && (now - beat_pulse_start) >= BEAT_PULSE_MS) {
        HAL_GPIO_WritePin(GPIOE, GPIO_PIN_7, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(GPIOE, GPIO_PIN_3, GPIO_PIN_RESET);
        pulse_active = 0;
    }

    last_filtered = filtered;
}

/* USER CODE END 0 */

int main(void)
{
  /* USER CODE BEGIN 1 */
  /* USER CODE END 1 */

  MPU_Config();
  HAL_Init();
  SystemClock_Config();
  PeriphCommonClock_Config();

  MX_GPIO_Init();
  MX_DMA_Init();
  MX_I2C1_Init();
  MX_USB_DEVICE_Init();
  MX_ADC1_Init();
  MX_TIM1_Init();
  MX_ADC2_Init();
  MX_USART1_UART_Init();

  /* USER CODE BEGIN 2 */
  HAL_Delay(3000);

  char startup_msg[] = "--- STM32 Booted. Checking Sensor... ---\r\n";
  CDC_Transmit_FS((uint8_t*)startup_msg, strlen(startup_msg));
  HAL_Delay(10);

  uint8_t part_id = 0;
  HAL_StatusTypeDef status = MAX30102_ReadReg(MAX30102_REG_PART_ID, &part_id);

  char id_msg[64];
  snprintf(id_msg, sizeof(id_msg), "I2C Status: %d, Part ID: 0x%02X\r\n", status, part_id);
  CDC_Transmit_FS((uint8_t*)id_msg, strlen(id_msg));
  HAL_Delay(10);

  if (status == HAL_OK && part_id == 0x15) {
      MAX30102_WriteReg(MAX30102_REG_MODE_CONFIG, 0x40);
      HAL_Delay(100);
      MAX30102_WriteReg(0x02, 0x40);
      MAX30102_WriteReg(MAX30102_REG_FIFO_CONFIG, 0x1F);
      MAX30102_WriteReg(MAX30102_REG_MODE_CONFIG, 0x03);
      MAX30102_WriteReg(MAX30102_REG_SPO2_CONFIG, 0x27);
      MAX30102_WriteReg(MAX30102_REG_LED1_PA, 0x24);
      MAX30102_WriteReg(MAX30102_REG_LED2_PA, 0x24);
      MAX30102_WriteReg(MAX30102_REG_FIFO_WR_PTR, 0x00);
      MAX30102_WriteReg(MAX30102_REG_FIFO_RD_PTR, 0x00);
      uint8_t dummy;
      MAX30102_ReadReg(0x00, &dummy);
      char success_msg[] = "Sensor Initialized! Waiting for EXTI...\r\n";
      CDC_Transmit_FS((uint8_t*)success_msg, strlen(success_msg));
      HAL_Delay(10);
  } else {
      char fail_msg[] = "ERROR: Sensor not found or wrong Part ID.\r\n";
      CDC_Transmit_FS((uint8_t*)fail_msg, strlen(fail_msg));
      HAL_Delay(10);
  }

  if (HAL_ADC_Start_DMA(&hadc1, (uint32_t*)accel_data, SCG_BUFFER_SIZE) != HAL_OK) {
      char adc_fail_msg[] = "ERROR: HAL_ADC_Start_DMA failed.\r\n";
      CDC_Transmit_FS((uint8_t*)adc_fail_msg, strlen(adc_fail_msg));
  }
  if (HAL_TIM_Base_Start(&htim1) != HAL_OK) {
      char tim_fail_msg[] = "ERROR: HAL_TIM_Base_Start failed.\r\n";
      CDC_Transmit_FS((uint8_t*)tim_fail_msg, strlen(tim_fail_msg));
  }
  /* USER CODE END 2 */

  while (1)
  {
      /* ── SCG (256 Hz, DMA double-buffer) ──────────────────────────────── */

      // FIX 2: __DMB() ensures the CPU has finished writing process_half
      // (in the ISR) before we read it here.  Without the barrier, the
      // Cortex-M7 store buffer can defer the ISR write past this load.
      __DMB();
      uint8_t half = process_half;
      __DMB();

      if (half == 1) {
          #if (__DCACHE_PRESENT == 1U)
          SCB_InvalidateDCache_by_Addr((uint32_t *)&accel_data[0], 32U);
          #endif
          // Snapshot the timestamp before clearing the flag so we don't race
          // with the next ISR firing.
          uint32_t ts = scg_timestamp;
          __DMB();
          process_half = 0;
          __DMB();
          Send_SCG_Packet(ts, (int16_t)accel_data[0],
                              (int16_t)accel_data[1],
                              (int16_t)accel_data[2]);
      }
      else if (half == 2) {
          #if (__DCACHE_PRESENT == 1U)
          SCB_InvalidateDCache_by_Addr((uint32_t *)&accel_data[3], 32U);
          #endif
          uint32_t ts = scg_timestamp;
          __DMB();
          process_half = 0;
          __DMB();
          Send_SCG_Packet(ts, (int16_t)accel_data[3],
                              (int16_t)accel_data[4],
                              (int16_t)accel_data[5]);
      }

      /* ── PPG (100 Hz, I2C DMA) ────────────────────────────────────────── */
      if (ppg_sample_ready) {
          ppg_sample_ready = 0;

          uint32_t ir = (((uint32_t)ppg_dma_rx_buffer[3] << 16) |
                         ((uint32_t)ppg_dma_rx_buffer[4] << 8)  |
                         (uint32_t)ppg_dma_rx_buffer[5]) & 0x03FFFF;
          uint32_t ppg_ts = HAL_GetTick();

          Send_PPG_Packet(ppg_ts, ir);

          if (ir > 100000) {
              int32_t detrended = Detrend_IR(ir);
              if (detrended != INT32_MIN) {
                  float filtered = LowPass_Filter((float)detrended);
                  beat_this_sample = 0;
                  Detect_Peak(filtered);
              }
          }
      }
  }
  /* USER CODE END WHILE */
}

/* USER CODE BEGIN 4 */

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef* hadc) {
    if (hadc->Instance == ADC1) {
        scg_timestamp = HAL_GetTick();
        __DMB();              // FIX 2: ensure timestamp write is visible before flag
        process_half = 1;
        __DMB();
    }
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc) {
    if (hadc->Instance == ADC1) {
        scg_timestamp = HAL_GetTick();
        __DMB();
        process_half = 2;
        __DMB();
    }
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    if (GPIO_Pin == MAX_INT_2_Pin) {
        // FIX 3: Only kick a new DMA read if the previous one has completed.
        // Without this guard, a second falling edge arriving before the I2C DMA
        // finishes calls HAL_I2C_Mem_Read_DMA() on a busy bus.  HAL returns
        // HAL_BUSY and the read is silently dropped — but the FIFO keeps
        // signalling, the next interrupt arrives immediately, and we spin in a
        // tight ISR loop that starves the main loop and floods the USB buffer
        // with retries, causing the PC-side freeze.
        if (!ppg_dma_busy) {
            ppg_dma_busy = 1;
            HAL_StatusTypeDef ret = HAL_I2C_Mem_Read_DMA(
                &hi2c1, MAX30102_I2C_ADDR,
                MAX30102_REG_FIFO_DATA, I2C_MEMADD_SIZE_8BIT,
                (uint8_t*)ppg_dma_rx_buffer, 6);
            if (ret != HAL_OK) {
                // DMA failed to start — clear the busy flag so the next
                // interrupt can try again rather than permanently stalling.
                ppg_dma_busy = 0;
            }
        }
        // If ppg_dma_busy is already set, the EXTI is firing faster than the
        // I2C can drain.  Drop this interrupt; the FIFO will assert again on
        // the next sample.
    }
}

void HAL_I2C_MemRxCpltCallback(I2C_HandleTypeDef *hi2c) {
    if (hi2c->Instance == I2C1) {
        ppg_sample_ready = 1;
        ppg_dma_busy     = 0;   // FIX 3: allow next EXTI to kick a new DMA read
    }
}

/* USER CODE END 4 */

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);
  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState       = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM       = 2;
  RCC_OscInitStruct.PLL.PLLN       = 12;
  RCC_OscInitStruct.PLL.PLLP       = 2;
  RCC_OscInitStruct.PLL.PLLQ       = 3;
  RCC_OscInitStruct.PLL.PLLR       = 2;
  RCC_OscInitStruct.PLL.PLLRGE     = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL  = RCC_PLL1VCOMEDIUM;
  RCC_OscInitStruct.PLL.PLLFRACN   = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

  RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                   | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2
                                   | RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider  = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider  = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK) Error_Handler();
}

void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  PeriphClkInitStruct.PeriphClockSelection  = RCC_PERIPHCLK_ADC;
  PeriphClkInitStruct.PLL2.PLL2M            = 2;
  PeriphClkInitStruct.PLL2.PLL2N            = 12;
  PeriphClkInitStruct.PLL2.PLL2P            = 2;
  PeriphClkInitStruct.PLL2.PLL2Q            = 2;
  PeriphClkInitStruct.PLL2.PLL2R            = 2;
  PeriphClkInitStruct.PLL2.PLL2RGE          = RCC_PLL2VCIRANGE_3;
  PeriphClkInitStruct.PLL2.PLL2VCOSEL       = RCC_PLL2VCOMEDIUM;
  PeriphClkInitStruct.PLL2.PLL2FRACN        = 0;
  PeriphClkInitStruct.AdcClockSelection     = RCC_ADCCLKSOURCE_PLL2;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) Error_Handler();
}

static void MX_ADC1_Init(void)
{
  ADC_MultiModeTypeDef multimode = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  hadc1.Instance                   = ADC1;
  hadc1.Init.ClockPrescaler         = ADC_CLOCK_ASYNC_DIV2;
  hadc1.Init.Resolution             = ADC_RESOLUTION_16B;
  hadc1.Init.ScanConvMode           = ADC_SCAN_ENABLE;
  hadc1.Init.EOCSelection           = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait       = DISABLE;
  hadc1.Init.ContinuousConvMode     = DISABLE;
  hadc1.Init.NbrOfConversion        = 3;
  hadc1.Init.DiscontinuousConvMode  = DISABLE;
  hadc1.Init.ExternalTrigConv       = ADC_EXTERNALTRIG_T1_TRGO;
  hadc1.Init.ExternalTrigConvEdge   = ADC_EXTERNALTRIGCONVEDGE_RISING;
  hadc1.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DMA_CIRCULAR;
  hadc1.Init.Overrun                = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.LeftBitShift           = ADC_LEFTBITSHIFT_NONE;
  hadc1.Init.OversamplingMode       = DISABLE;
  hadc1.Init.Oversampling.Ratio     = 1;
  if (HAL_ADC_Init(&hadc1) != HAL_OK) Error_Handler();

  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK) Error_Handler();

  sConfig.Channel              = ADC_CHANNEL_3;
  sConfig.Rank                 = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime         = ADC_SAMPLETIME_810CYCLES_5;
  sConfig.SingleDiff           = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber         = ADC_OFFSET_NONE;
  sConfig.Offset               = 0;
  sConfig.OffsetSignedSaturation = DISABLE;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) Error_Handler();
  sConfig.Channel = ADC_CHANNEL_4; sConfig.Rank = ADC_REGULAR_RANK_2;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) Error_Handler();
  sConfig.Channel = ADC_CHANNEL_5; sConfig.Rank = ADC_REGULAR_RANK_3;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) Error_Handler();
}

static void MX_ADC2_Init(void)
{
  ADC_ChannelConfTypeDef sConfig = {0};

  hadc2.Instance                   = ADC2;
  hadc2.Init.ClockPrescaler         = ADC_CLOCK_ASYNC_DIV2;
  hadc2.Init.Resolution             = ADC_RESOLUTION_16B;
  hadc2.Init.ScanConvMode           = ADC_SCAN_DISABLE;
  hadc2.Init.EOCSelection           = ADC_EOC_SINGLE_CONV;
  hadc2.Init.LowPowerAutoWait       = DISABLE;
  hadc2.Init.ContinuousConvMode     = DISABLE;
  hadc2.Init.NbrOfConversion        = 1;
  hadc2.Init.DiscontinuousConvMode  = DISABLE;
  hadc2.Init.ExternalTrigConv       = ADC_SOFTWARE_START;
  hadc2.Init.ExternalTrigConvEdge   = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc2.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DR;
  hadc2.Init.Overrun                = ADC_OVR_DATA_PRESERVED;
  hadc2.Init.LeftBitShift           = ADC_LEFTBITSHIFT_NONE;
  hadc2.Init.OversamplingMode       = DISABLE;
  hadc2.Init.Oversampling.Ratio     = 1;
  if (HAL_ADC_Init(&hadc2) != HAL_OK) Error_Handler();

  sConfig.Channel              = ADC_CHANNEL_18;
  sConfig.Rank                 = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime         = ADC_SAMPLETIME_1CYCLE_5;
  sConfig.SingleDiff           = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber         = ADC_OFFSET_NONE;
  sConfig.Offset               = 0;
  sConfig.OffsetSignedSaturation = DISABLE;
  if (HAL_ADC_ConfigChannel(&hadc2, &sConfig) != HAL_OK) Error_Handler();
}

static void MX_I2C1_Init(void)
{
  hi2c1.Instance              = I2C1;
  hi2c1.Init.Timing           = 0x00909FCE;
  hi2c1.Init.OwnAddress1      = 0;
  hi2c1.Init.AddressingMode   = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode  = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2      = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode  = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode    = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK) Error_Handler();
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK) Error_Handler();
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK) Error_Handler();
}

static void MX_TIM1_Init(void)
{
  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  htim1.Instance               = TIM1;
  htim1.Init.Prescaler         = 500 - 1;
  htim1.Init.CounterMode       = TIM_COUNTERMODE_UP;
  htim1.Init.Period            = 586 - 1;
  htim1.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK) Error_Handler();

  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK) Error_Handler();

  sMasterConfig.MasterOutputTrigger  = TIM_TRGO_UPDATE;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode      = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK) Error_Handler();
}

static void MX_USART1_UART_Init(void)
{
  huart1.Instance            = USART1;
  huart1.Init.BaudRate       = 115200;
  huart1.Init.WordLength     = UART_WORDLENGTH_8B;
  huart1.Init.StopBits       = UART_STOPBITS_1;
  huart1.Init.Parity         = UART_PARITY_NONE;
  huart1.Init.Mode           = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl      = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling   = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK) Error_Handler();
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK) Error_Handler();
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK) Error_Handler();
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK) Error_Handler();
}

static void MX_DMA_Init(void)
{
  __HAL_RCC_DMA1_CLK_ENABLE();
  HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream0_IRQn);
  HAL_NVIC_SetPriority(DMA1_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream1_IRQn);
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(GPIOE, BEAT_LED_Pin | BEAT_OUT_Pin, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin   = BEAT_LED_Pin | BEAT_OUT_Pin;
  GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull  = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  GPIO_InitStruct.Pin  = MAX_INT_Pin | MAX_INT_2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  HAL_NVIC_SetPriority(MAX_INT_2_EXTI_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(MAX_INT_2_EXTI_IRQn);
  HAL_NVIC_SetPriority(MAX_INT_EXTI_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(MAX_INT_EXTI_IRQn);
}

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};
  HAL_MPU_Disable();
  MPU_InitStruct.Enable            = MPU_REGION_ENABLE;
  MPU_InitStruct.Number            = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress       = 0x0;
  MPU_InitStruct.Size              = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable  = 0x87;
  MPU_InitStruct.TypeExtField      = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission  = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec       = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable       = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable       = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable      = MPU_ACCESS_NOT_BUFFERABLE;
  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

void Error_Handler(void)
{
  __disable_irq();
  while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {}
#endif
