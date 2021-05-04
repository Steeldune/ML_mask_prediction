import numpy

print(1024**2)

test = 3*1024**2

list = []

for i in range(int(test/(1024**2))):
    print(i)
    if i+1 == 1:
        list.append([0, 1024**2])
    else:
        list.append([list[-1][1] + 1, (i+1)*1024**2])
    print(list)



print(list)