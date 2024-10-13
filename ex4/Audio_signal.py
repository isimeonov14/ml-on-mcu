import numpy as np
import matplotlib.pyplot as plt

with open("Audio_signal.txt", "r") as file:
    data = file.read()


data_list = data.split()
data_int_list = [int(item) for item in data_list]

data_array = np.array(data_int_list)
print(data_array.size)
print(data_array[:10])

plt.plot(data_array)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of Data Array')

plt.show()

