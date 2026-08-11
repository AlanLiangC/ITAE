请根据我下一步的计划在docs/navsim_task/plan.md写一个code plan，我检查无误后再实施：


下一步，我想增加数据量来训练我的action tokenizer

我要添加navsim数据集，其官方库我添加到了third_party/navsim，你可以参考，看看数据是如何读取的。记得先安装一下这个库，你可以根据库自行下载其他任何依赖。我下载了mini set在：/home/alan/AlanLiang/Dataset/navsim


我想使action tokenizer支持联合数据集训练，即在navsim和nuscenes上训练

setting还是预测未来4s的40个轨迹点，其中navsim是2Hz的，但是在计算PDMS指标时会将轨迹插值到40个点，你可以模仿这个操作来创建GT

我希望action tokenizer从scratch训练。

third_party/SUV是一个其他人员发布的使用navsim进行端到端规划的项目，供你参考。其中third_party/SUV/experiments可用于在navsim上测试指标，我后面也会使用，所以我将该文件夹复制到了本项目中。

tools中文件太多太杂乱了，你帮我整理一下，不同目的的文件放在一个文件夹。

