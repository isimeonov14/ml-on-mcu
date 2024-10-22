################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../TFLite/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
../TFLite/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc 

CC_DEPS += \
./TFLite/tensorflow/lite/micro/memory_planner/greedy_memory_planner.d \
./TFLite/tensorflow/lite/micro/memory_planner/linear_memory_planner.d 

OBJS += \
./TFLite/tensorflow/lite/micro/memory_planner/greedy_memory_planner.o \
./TFLite/tensorflow/lite/micro/memory_planner/linear_memory_planner.o 


# Each subdirectory must supply rules for building sources it contributes
TFLite/tensorflow/lite/micro/memory_planner/%.o TFLite/tensorflow/lite/micro/memory_planner/%.su TFLite/tensorflow/lite/micro/memory_planner/%.cyclo: ../TFLite/tensorflow/lite/micro/memory_planner/%.cc TFLite/tensorflow/lite/micro/memory_planner/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32L475xx -c -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/CMSIS/Include -I../Core/tensorflow/lite/micro/tools/make/downloads -I../Core/ -I../Core/Inc -I../Core -I../TFLite -I../TFLite/third_party/flatbuffers/include -I../TFLite/third_party/gemmlowp -I../TFLite/third_party/ruy -I../TFLite/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/DSP/Include -I../TFLite/tensorflow/lite/micro/tools/make/downloads/ -I../TFLite/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/NN/Include -O3 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-TFLite-2f-tensorflow-2f-lite-2f-micro-2f-memory_planner

clean-TFLite-2f-tensorflow-2f-lite-2f-micro-2f-memory_planner:
	-$(RM) ./TFLite/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cyclo ./TFLite/tensorflow/lite/micro/memory_planner/greedy_memory_planner.d ./TFLite/tensorflow/lite/micro/memory_planner/greedy_memory_planner.o ./TFLite/tensorflow/lite/micro/memory_planner/greedy_memory_planner.su ./TFLite/tensorflow/lite/micro/memory_planner/linear_memory_planner.cyclo ./TFLite/tensorflow/lite/micro/memory_planner/linear_memory_planner.d ./TFLite/tensorflow/lite/micro/memory_planner/linear_memory_planner.o ./TFLite/tensorflow/lite/micro/memory_planner/linear_memory_planner.su

.PHONY: clean-TFLite-2f-tensorflow-2f-lite-2f-micro-2f-memory_planner

