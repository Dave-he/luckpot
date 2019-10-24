import pyautogui
import math
import keyboard

width, height = pyautogui.size()
r = 250 # 圆的半径
# 圆心
o_x = width/2
o_y = height/2
pi = 3.1415926

# for i in range(10):  # 转10圈
#     for angle in range(0, 360, 5): # 利用圆的参数方程
#         X = o_x + r * math.sin(angle*pi/180)
#         Y = o_y + r * math.cos(angle*pi/180)
#         pyautogui.moveTo(X, Y, duration=0.1)

keyboard.add_hotkey('ctrl', print, args=('aaa',))
keyboard.add_hotkey('alt', print, args=('bbb',))

recorded = keyboard.record(until='esc')
#当按下esc时结束按键监听，并输出所有按键事件
print(recorded)

