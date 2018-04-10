# ZTECompetition
Just a try about deploying a classification model on android platform

# Reference
1. deploying a model on android: https://zhuanlan.zhihu.com/p/32342366
2. get train and test imgs from Baidu:https://github.com/kong36088/BaiduImageSpider
3. install Android Studio: https://blog.csdn.net/u013038616/article/details/75315283

# Steps
1. Get train and test imgs
2. Start finetune
3. Using onnx to convert pytorch model into caffe2 model
4. clone AI Camera Demo from github: git clone https://github.com/bwasti/AICamera.git
5. change path of model and labelname

# Dependencies
1. pytorch, caffe2 and onnx
2. python2 and python3 environment
3. cuda8.0, ncll, cudnn7.1

# ClassNames
1. 狗 2. 手机 3. 猫 4. 杯子 5. 花 6. 书桌 7. 人 8. 电视机 9. 电脑 10. 钱包
11. 笔记本 12. 眼镜 13. 鼠标 14. 汽车 15. 自行车 16. 火车 17. 椅子 18. 餐桌 19. 沙发 20. 鸟
