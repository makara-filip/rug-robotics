from math import sin, cos, pi
import numpy

theta = pi / 6 # 30 degrees
# theta = 0

def t(angle, length):
    return numpy.array([
        [cos(angle), -sin(angle), 0, length],
        [sin(angle), cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


a = t(theta, 0)
b = t(theta, 2)
c = t(theta, 2)

# final = numpy.dot(a, numpy.dot(b, c))
final = a @ b @ c
print(final)

end_effector = numpy.array([1, 0, 0, 1])
print("End effector in base frame:", final @ end_effector)

# print(a)
# print(b)
