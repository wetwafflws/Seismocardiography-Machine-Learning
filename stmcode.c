/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "usbd_cdc_if.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
// shift by 1 krn stm32 8 bit addr sedangkan maxnya 7 bit
#define MAX30102_I2C_ADDR (0x57 << 1) // 0xAE for writing, 0xAF for reading

// MAX30102 Registers
#define MAX30102_REG_INT_STATUS_1  0x00
#define MAX30102_REG_FIFO_WR_PTR   0x04
#define MAX30102_REG_FIFO_RD_PTR   0x06
#define MAX30102_REG_FIFO_DATA     0x07
#define MAX30102_REG_FIFO_CONFIG   0x08
#define MAX30102_REG_MODE_CONFIG   0x09
#define MAX30102_REG_SPO2_CONFIG   0x0A
#define MAX30102_REG_LED1_PA       0x0C // RED LED Amplitude
#define MAX30102_REG_LED2_PA       0x0D // IR LED Amplitude
#define MAX30102_REG_PART_ID       0xFF

#define MA_WINDOW_SIZE  50

// 2nd-order Butterworth low-pass, Fc=5Hz, Fs=100Hz
// Generated via bilinear transform:
//   b = [0.02008, 0.04016, 0.02008]
//   a = [1.0,    -1.56102, 0.64135]
#define IIR_B0  0.02008f
#define IIR_B1  0.04016f
#define IIR_B2  0.02008f
#define IIR_A1 -1.56102f  // coefficient on y[n-1]
#define IIR_A2  0.64135f  // coefficient on y[n-2]

#define REFRACTORY_MS       400   // Minimum ms between beats (~150 BPM max)
#define BEAT_PULSE_MS        50   // Duration of PB1 output pulse
#define THRESHOLD_FACTOR    0.4f  // Peak threshold = 40% of tracked amplitude
#define AMPLITUDE_DECAY     0.99f // How quickly amplitude tracking follows signal down

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;

I2C_HandleTypeDef hi2c1;

TIM_HandleTypeDef htim1;

/* USER CODE BEGIN PV */

volatile uint8_t scg_ready = 0;

__attribute__((aligned(32))) volatile uint16_t accel_data[3];

// Circular buffer stores recent IR samples for the DC baseline estimate.
uint32_t ma_buffer[MA_WINDOW_SIZE];
uint32_t ma_sum    = 0;
uint8_t  ma_index  = 0;
uint8_t  ma_filled = 0;

float iir_x1 = 0.0f, iir_x2 = 0.0f;
float iir_y1 = 0.0f, iir_y2 = 0.0f;

float    peak_amplitude   = 1000.0f; // Tracked signal amplitude (adaptive)
float    peak_threshold   = 400.0f;  // Recomputed each sample from amplitude
float    last_filtered    = 0.0f;    // Previous sample (for slope detection)
// Slope-based detection state
uint8_t  rising            = 0;      // 1 = signal is currently in a rising slope
float    slope_peak_val    = 0.0f;   // Highest value seen during current rise
// Minimum number of consecutive falling samples required to confirm a peak.
// Prevents triggering on a tiny wobble at the top of the waveform.
#define  PEAK_CONFIRM_SAMPLES  3
#define PKT_MAGIC       0xAA
#define PKT_TYPE_SCG    0x01
#define PKT_TYPE_BEAT   0x02

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

uint8_t  falling_count     = 0;
uint32_t last_beat_tick    = 0;
uint32_t beat_pulse_start  = 0;
uint8_t  pulse_active      = 0;
uint8_t  beat_this_sample  = 0;


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_I2C1_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM1_Init(void);
/* USER CODE BEGIN PFP */
HAL_StatusTypeDef MAX30102_WriteReg(uint8_t reg, uint8_t value);
HAL_StatusTypeDef MAX30102_ReadReg(uint8_t reg, uint8_t *value);
int32_t Detrend_IR(uint32_t ir_sample);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static uint8_t xor_checksum(uint8_t *buf, uint8_t len) {
    uint8_t c = 0;
    for (uint8_t i = 0; i < len; i++) c ^= buf[i];
    return c;
}

void Send_SCG_Packet(int16_t x, int16_t y, int16_t z) {
    SCG_Packet_t pkt;
    pkt.magic        = PKT_MAGIC;
    pkt.type         = PKT_TYPE_SCG;
    pkt.timestamp_ms = HAL_GetTick();
    pkt.x = x; pkt.y = y; pkt.z = z;
    pkt.checksum     = xor_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);

  uint8_t retries = 3;
  while (CDC_Transmit_FS((uint8_t*)&pkt, sizeof(pkt)) == USBD_BUSY && --retries)
    HAL_Delay(1);
}

void Send_Beat_Packet(void) {
    Beat_Packet_t pkt;
    pkt.magic        = PKT_MAGIC;
    pkt.type         = PKT_TYPE_BEAT;
    pkt.timestamp_ms = HAL_GetTick();
    pkt.checksum     = xor_checksum((uint8_t*)&pkt, sizeof(pkt) - 1);
    uint8_t retries = 3;
    while (CDC_Transmit_FS((uint8_t*)&pkt, sizeof(pkt)) == USBD_BUSY && --retries)
        HAL_Delay(1);
}

HAL_StatusTypeDef MAX30102_WriteReg(uint8_t reg, uint8_t value) {
    return HAL_I2C_Mem_Write(&hi2c1, MAX30102_I2C_ADDR, reg, I2C_MEMADD_SIZE_8BIT, &value, 1, HAL_MAX_DELAY);
}

HAL_StatusTypeDef MAX30102_ReadReg(uint8_t reg, uint8_t *value) {
    return HAL_I2C_Mem_Read(&hi2c1, MAX30102_I2C_ADDR, reg, I2C_MEMADD_SIZE_8BIT, value, 1, HAL_MAX_DELAY);
}
int32_t Detrend_IR(uint32_t ir_sample) {
    // Update circular buffer: subtract old value, add new value
    ma_sum -= ma_buffer[ma_index];
    ma_buffer[ma_index] = ir_sample;
    ma_sum += ir_sample;

    ma_index++;
    if (ma_index >= MA_WINDOW_SIZE) {
        ma_index  = 0;
        ma_filled = 1;  // Buffer is fully populated from this point on
    }

    // Don't return data until the window is full — the mean would be wrong
    if (!ma_filled) {
        return INT32_MIN;  // Sentinel: caller should skip this sample
    }

    uint32_t mean = ma_sum / MA_WINDOW_SIZE;
    return (int32_t)ir_sample - (int32_t)mean;
}

float LowPass_Filter(float input) {
    float y = IIR_B0 * input
            + IIR_B1 * iir_x1
            + IIR_B2 * iir_x2
            - IIR_A1 * iir_y1
            - IIR_A2 * iir_y2;

    iir_x2 = iir_x1;
    iir_x1 = input;
    iir_y2 = iir_y1;
    iir_y1 = y;

    return y;
}

void Detect_Peak(float filtered) {
    uint32_t now = HAL_GetTick();

    // --- Adaptive amplitude tracking ---
    if (filtered > peak_amplitude) {
        peak_amplitude = filtered;
    } else {
        peak_amplitude *= AMPLITUDE_DECAY;
    }
    peak_threshold = peak_amplitude * THRESHOLD_FACTOR;

    float slope = filtered - last_filtered;

    if (!rising) {
        // Waiting for a rise above threshold to begin a candidate peak
        if (filtered > peak_threshold && slope > 0.0f) {
            rising        = 1;
            falling_count = 0;
            slope_peak_val = filtered;
        }
    } else {
        // Currently tracking a rising slope
        if (slope > 0.0f) {
            // Still climbing — update the tracked peak value
            slope_peak_val = filtered;
            falling_count  = 0;
        } else {
            // Slope has gone negative — count consecutive falling samples
            falling_count++;

            if (falling_count >= PEAK_CONFIRM_SAMPLES) {
                // Enough falling samples — this is a real peak, not a wobble
                rising = 0;
                falling_count = 0;

                // Only fire if refractory period has passed
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

            // If signal drops back below threshold mid-fall, abort this candidate
            // (wasn't a real peak, just noise on the rising slope)
            if (filtered < peak_threshold) {
                rising        = 0;
                falling_count = 0;
            }
        }
    }

    // Non-blocking pulse timeout
    if (pulse_active && (now - beat_pulse_start) >= BEAT_PULSE_MS) {
        HAL_GPIO_WritePin(GPIOE, GPIO_PIN_7, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(GPIOE, GPIO_PIN_3, GPIO_PIN_RESET);
        pulse_active = 0;
    }

    last_filtered = filtered;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_I2C1_Init();
  MX_USB_DEVICE_Init();
  MX_ADC1_Init();
  MX_TIM1_Init();
  /* USER CODE BEGIN 2 */

    // 1. MANDATORY DELAY: Wait 3 seconds for the PC's USB port to connect
    HAL_Delay(3000);

    char startup_msg[] = "--- STM32 Booted. Checking Sensor... ---\r\n";
    CDC_Transmit_FS((uint8_t*)startup_msg, strlen(startup_msg));
    HAL_Delay(10);

    // 2. Read the Part ID
    uint8_t part_id = 0;
    HAL_StatusTypeDef status = MAX30102_ReadReg(MAX30102_REG_PART_ID, &part_id);

    // 3. Print the actual I2C status and Part ID to the screen
    char id_msg[64];
    // Note: status 0 = HAL_OK, 1 = HAL_ERROR, 2 = HAL_BUSY, 3 = HAL_TIMEOUT
    snprintf(id_msg, sizeof(id_msg), "I2C Status: %d, Part ID: 0x%02X\r\n", status, part_id);
    CDC_Transmit_FS((uint8_t*)id_msg, strlen(id_msg));
    HAL_Delay(10);

    // 4. Proceed only if successful
    if (status == HAL_OK && part_id == 0x15) {

        MAX30102_WriteReg(MAX30102_REG_MODE_CONFIG, 0x40); // Reset
        HAL_Delay(100);

        MAX30102_WriteReg(0x02, 0x80); // Enable A_FULL Interrupt
        MAX30102_WriteReg(MAX30102_REG_FIFO_CONFIG, 0x1F); // Rollover + 15 empty samples
        MAX30102_WriteReg(MAX30102_REG_MODE_CONFIG, 0x03); // SpO2 Mode
        MAX30102_WriteReg(MAX30102_REG_SPO2_CONFIG, 0x27); // 100Hz, 411us
        MAX30102_WriteReg(MAX30102_REG_LED1_PA, 0x24); // Red LED
        MAX30102_WriteReg(MAX30102_REG_LED2_PA, 0x24); // IR LED

        MAX30102_WriteReg(MAX30102_REG_FIFO_WR_PTR, 0x00);
        MAX30102_WriteReg(MAX30102_REG_FIFO_RD_PTR, 0x00);

        // Clear leftover interrupts
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

    if (HAL_ADC_Start_DMA(&hadc1, (uint32_t*)accel_data, 3) != HAL_OK) {
      char adc_fail_msg[] = "ERROR: HAL_ADC_Start_DMA failed.\r\n";
      CDC_Transmit_FS((uint8_t*)adc_fail_msg, strlen(adc_fail_msg));
      HAL_Delay(10);
    }
    if (HAL_TIM_Base_Start(&htim1) != HAL_OK) {
      char tim_fail_msg[] = "ERROR: HAL_TIM_Base_Start failed.\r\n";
      CDC_Transmit_FS((uint8_t*)tim_fail_msg, strlen(tim_fail_msg));
      HAL_Delay(10);
    }
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
    uint32_t last_scg_tx = HAL_GetTick();
    while (1) {

      uint32_t now = HAL_GetTick();
      if ((now - last_scg_tx) >= 10U) {
        last_scg_tx = now;
  #if (__DCACHE_PRESENT == 1U)
        // Ensure CPU reads fresh DMA-updated ADC samples on cache-enabled MCUs.
        SCB_InvalidateDCache_by_Addr((uint32_t *)(void *)accel_data, 32U);
  #endif
        Send_SCG_Packet((int16_t)accel_data[0],
                (int16_t)accel_data[1],
                (int16_t)accel_data[2]);
        scg_ready = 0;
      }

        uint8_t wr_ptr = 0, rd_ptr = 0;
        MAX30102_ReadReg(MAX30102_REG_FIFO_WR_PTR, &wr_ptr);
        MAX30102_ReadReg(MAX30102_REG_FIFO_RD_PTR, &rd_ptr);

        int8_t num_samples = (int8_t)wr_ptr - (int8_t)rd_ptr;
        if (num_samples < 0) num_samples += 32;

        if (num_samples > 0) {
            for (int i = 0; i < num_samples; i++) {
                uint8_t sample_buffer[6];
                HAL_I2C_Mem_Read(&hi2c1, MAX30102_I2C_ADDR, MAX30102_REG_FIFO_DATA,
                                 I2C_MEMADD_SIZE_8BIT, sample_buffer, 6, HAL_MAX_DELAY);

                uint32_t ir = (((uint32_t)sample_buffer[3] << 16) |
                               ((uint32_t)sample_buffer[4] << 8)  |
                                (uint32_t)sample_buffer[5]) & 0x03FFFF;

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
        // No delay — loop runs as fast as I2C allows,
        // which is naturally rate-limited by the blocking Mem_Read calls
    }

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 5;
  RCC_OscInitStruct.PLL.PLLN = 48;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 5;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_MultiModeTypeDef multimode = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV2;
  hadc1.Init.Resolution = ADC_RESOLUTION_16B;
  hadc1.Init.ScanConvMode = ADC_SCAN_ENABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 3;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_EXTERNALTRIG_T1_TRGO;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_RISING;
  hadc1.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DMA_CIRCULAR;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.LeftBitShift = ADC_LEFTBITSHIFT_NONE;
  hadc1.Init.OversamplingMode = DISABLE;
  hadc1.Init.Oversampling.Ratio = 1;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the ADC multi-mode
  */
  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_3;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_810CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  sConfig.OffsetSignedSaturation = DISABLE;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_4;
  sConfig.Rank = ADC_REGULAR_RANK_2;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_5;
  sConfig.Rank = ADC_REGULAR_RANK_3;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x007074AF;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 1250-1;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 375-1;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream0_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOE, BEAT_LED_Pin|BEAT_OUT_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pins : BEAT_LED_Pin BEAT_OUT_Pin */
  GPIO_InitStruct.Pin = BEAT_LED_Pin|BEAT_OUT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pin : MAX_INT_Pin */
  GPIO_InitStruct.Pin = MAX_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(MAX_INT_GPIO_Port, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(MAX_INT_EXTI_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(MAX_INT_EXTI_IRQn);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
    if (hadc->Instance == ADC1) {
        scg_ready = 1;  // just set flag, nothing else
    }
}


/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
