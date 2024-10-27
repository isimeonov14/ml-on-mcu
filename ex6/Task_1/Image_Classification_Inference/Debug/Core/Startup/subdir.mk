################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../Core/Startup/startup_stm32l475vgtx.s 

S_DEPS += \
./Core/Startup/startup_stm32l475vgtx.d 

OBJS += \
./Core/Startup/startup_stm32l475vgtx.o 


# Each subdirectory must supply rules for building sources it contributes
Core/Startup/%.o: ../Core/Startup/%.s Core/Startup/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m4 -g3 -DDEBUG -DCMSIS_NN -c -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/tensorflow_lite" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/Drivers/CMSIS/Core/Include" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/Drivers/CMSIS/Device/ST/STM32L4xx" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/Drivers/CMSIS/DSP/Include" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/Drivers/CMSIS/DSP/PrivateInclude" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/Drivers/CMSIS/NN" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/Drivers/CMSIS/Include" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/tensorflow_lite/third_party/flatbuffers/include" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/tensorflow_lite/third_party/gemmlowp" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/tensorflow_lite/third_party/kissfft" -I"C:/Users/Ivan/Desktop/ETH/Semester3/ml-on-mcu/ex6/Task_1/Image_Classification_Inference/tensorflow_lite/third_party/ruy" -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

clean: clean-Core-2f-Startup

clean-Core-2f-Startup:
	-$(RM) ./Core/Startup/startup_stm32l475vgtx.d ./Core/Startup/startup_stm32l475vgtx.o

.PHONY: clean-Core-2f-Startup

