################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../TFLite/tensorflow/lite/micro/cortex_m_generic/debug_log.cc 

CC_DEPS += \
./TFLite/tensorflow/lite/micro/cortex_m_generic/debug_log.d 

OBJS += \
./TFLite/tensorflow/lite/micro/cortex_m_generic/debug_log.o 


# Each subdirectory must supply rules for building sources it contributes
TFLite/tensorflow/lite/micro/cortex_m_generic/%.o TFLite/tensorflow/lite/micro/cortex_m_generic/%.su TFLite/tensorflow/lite/micro/cortex_m_generic/%.cyclo: ../TFLite/tensorflow/lite/micro/cortex_m_generic/%.cc TFLite/tensorflow/lite/micro/cortex_m_generic/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32L475xx -c -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/CMSIS/Include -I../Core/tensorflow/lite/micro/tools/make/downloads -I../Core/ -I../Core/Inc -I../Core -I../TFLite -I../TFLite/third_party/flatbuffers/include -I../TFLite/third_party/gemmlowp -I../TFLite/third_party/ruy -I../TFLite/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/DSP/Include -I../TFLite/tensorflow/lite/micro/tools/make/downloads/ -I../TFLite/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/NN/Include -O3 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-TFLite-2f-tensorflow-2f-lite-2f-micro-2f-cortex_m_generic

clean-TFLite-2f-tensorflow-2f-lite-2f-micro-2f-cortex_m_generic:
	-$(RM) ./TFLite/tensorflow/lite/micro/cortex_m_generic/debug_log.cyclo ./TFLite/tensorflow/lite/micro/cortex_m_generic/debug_log.d ./TFLite/tensorflow/lite/micro/cortex_m_generic/debug_log.o ./TFLite/tensorflow/lite/micro/cortex_m_generic/debug_log.su

.PHONY: clean-TFLite-2f-tensorflow-2f-lite-2f-micro-2f-cortex_m_generic

