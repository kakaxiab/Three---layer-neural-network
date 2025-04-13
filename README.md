# Three---layer-neural-network
提供两种代码，一种直接使用cpu，一种部署在kaggle的GPU
直接使用cpu的代码分为train，model，test，parameter_search,main几个文件
其中train用于训练模型，parameter_searc用于寻找合适的超参数，test用于测试模型，model构建了三层神经网络
部署在kaggle的GPU上的相应code在jupyternetebook中
每个单元格依次完成了数据读取、模型构建、参数遍历和选择、模型训练、模型存储、结果可视化等几个部分

报告中所展示结果来自GPU部署的three（1）文件训练结果
