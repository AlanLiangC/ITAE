暂时不用再往后优化trajectory tokenizer的模型架构了，

我觉得也可能跟数据集的体量有关，再后面的计划中我将再加入其他数据集提升模型的泛化性

现在有一个更要紧的问题

我想验证diffusion planner使用原始轨迹点还是我们tokenized后的action token收敛更快更好

所以请你选择一个目前为止你认为最好的一个tokenizer为baseline，做这个对比实验


任务要求是这样的：

输入当前帧，用diffusion policy （使用flow matching预测v）预测未来4s 40个轨迹点，请你在./docs先根据我的需求写一个code plan，我确认无误后开始落实代码，推理阶段允许5步