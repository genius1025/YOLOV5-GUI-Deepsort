**团队介绍**
组长：赵子阳 班级：电气2003班 学号：U202012286 主要分工：查找相关文献，收集数据集，YOLOv5环境搭建，编写和修改Deepsort部分代码，完成总结报告。
组员：张凯轩 班级：电气2003班 学号：U202012265 主要分工：YOLOv5整体代码的编写，对数据集进行训练，编写和修改Deepsort部分代码，整合GUI部分代码，对整个项目进行调试修改。
组员：俆展鹏 班级：电气2003班 学号：U202012271 主要分工：查找YOLOv5和Deepsort相关资料，答辩PPT的制作

**项目介绍**
本项目主要实现的目标是设计一个UI界面，在UI界面上能够选择进行图片检测、视频检测以及摄像头检测三个功能。在检测时能运用Deepsort算法对所检测出的物体各分配一个特定的ID，以此达到跟踪功能。

**文件构成与组成**
yolov5文件夹：包含models文件，用于保存模型；weights文件，用于保存权重文件；utils文件，用于检测核心代码；runs文件，保存运行好的视频等等。该文件夹是物体检测运行的主代码
deep_sort_pytorch文件夹：包含configs文件，保存deepsort一些参数；deep_sort文件，保存核心算法；utils文件等等。该文件夹是物体跟踪运行的主代码
detect.py文件：物体检测代码
detect_logical.py文件：进入检测界面
track.py文件：打开摄像头

**两种程序使用方式：**
直接运行detect_logical.py，进入检测界面
运行track.py，进入摄像头跟踪界面

**作业及报告位置**
平时作业在summercamp里面
预习报告。总结报告以及答辩PPT均在repo里面
![image](https://user-images.githubusercontent.com/111260252/190394701-6fca33aa-cee3-4a45-9390-9253a923e19e.png)



