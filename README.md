### 天池（新人赛）工业蒸汽预测
学习使用交流借鉴
[参赛链接](https://tianchi.aliyun.com/competition/entrance/231693/introduction?spm=5176.12281949.1003.1.355e4c2a9SSaO0)
#### Windows

1.下载Anaconda
<br/>
2.安装相应库:
pip install Keras
<br/>
注意：
###### python 依赖：
<br>
- Keras 
- Numpy 
- Sklearn
- Lightgbm
- Pandas
<br/>
<br>

###### 算法流程
* 构建机器学习模型    
     输入：训练数据的特征数据和标签数据
     输出：一个深度神经网络模型
	 参数：    
	 ReduceLROnPlateau
    monitor:被监测的数据
    factor：学习速率被降低的因素。新的学习速率 = 学习速率 * 因数
    patience：没有进步的训练轮数，在这之后训练速率会被降低。
    verbose:整数。0：安静，1：更新信息。
    mode： {auto, min, max} 其中之一。如果是 min 模式，学习速率会被降低如果被监测的数据已经停止下降； 
                                     在 max 模式，学习塑料会被降低如果被监测的数据已经停止上升； 
                                     在 auto 模式，方向会被从被监测的数据中自动推断出来
    min_delta: 对于测量新的最优化的阀值，只关注巨大的改变。
    cooldown:在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
    min_lr:学习速率的下边界。
* 创建多项式模型的函数
    输入：一个维度
    输出：一个对应维度的多项式机器学习模型
* 画出学习曲线->判断过拟合和欠拟合
* 交叉验证
* 贝叶斯岭回归
* 数据降维(主成分分析法)
* 贝叶斯岭回归
当数据有很多缺失或者矛盾的病态数据，可以考虑，对病态数据鲁棒性很高，也不用交叉验证选择超参数。
但是极大化似然函数的推断过程比较耗时，一般情况不推荐使用。 
* 将最终预测结果保存到文件当中
