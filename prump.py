import math
sum = 0
i = 0
p=0.6
while sum > math.log2(0.9):
    log_sum = math.log2(math.pow(p, 25-i))+math.log2(math.pow(1-p, i))
    sum += log_sum
    i += 1

print(str(i))
