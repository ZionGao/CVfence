
# Introduction
  敏感区域擅入baseline
  
# Requirment
  keras==2.2.4
  PIL==1.1.7
  opencv==3.4.2
  numpy==1.17.2 

# Models
  模型置于 model_data下
  detector 基于yoloV3 需转换成 .h5  
  h5下载 链接:https://pan.baidu.com/s/1I-UCvgzQ8kqwQAQPozvg5g  密码:v5fg  
  tracker下载 链接:https://pan.baidu.com/s/1gP_Iqbk5il2hBt0PY8UUhg  密码:xozp
 
# Run
  python main.py --input test.mp4

# Notice
  1 弹出视频首帧，顺时针点选矩形敏感区域顶点  
  2 检测到区域内出现行人，保存截图至output目录  
  3 输出目标跟踪视频至output.avi  
  4 判断人是否属于进入区域：  
        ①out-->in-->左侧线相交  
        ②out-->in-->右侧线相交  
        ③out-->in-->上侧线相交  
        ④out-->in持续5秒  
        ⑤out-->in-->消失5秒  