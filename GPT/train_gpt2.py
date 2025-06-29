import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

# 复原架构并成功推理
# 1. 设置参数
# 2. 加载huffing_face里的gpt2的架构,参数
# 3. 按照架构，复现gpt2,按照class_gpt,Block, MLP,causualAttentiong的顺序从外到里
# 4. 将huggingface权重加载到写的架构上，用注意@classmethod装饰器
# 5. 在GPT里面写farward方法
# 6. 设置设备，参数，encode tocken,准备推理
# 7. 推理脚本，topk概率,打印结果

# 自己训练
# 1. 数据集，打开莎士比亚数据集
# 2. 用tiktoken（gpt2的编码器）编码数据集
# 3. 巧妙构造训练集和lable(错开一位)
# 4. 计算loss,在farward传入target参数并计算loss(注意心里要有一个最开始大致的loss:-ln(1/50257))
# 5. 接下来loss.backward, 计算梯度，以及optimization,过拟合样本
# 6. 接下来我们不仅想要过拟合一个单一的样本，我们要进行optimization，我们需要迭代x,y批次，创建小的dataloder,保证我们总得到新的批次，优化合理的指标
# 7. 在GPT__init__方法中实施权重共享方案：但发现hugging_face权重transforer.wte.weight和lm_head_weight的一摸一样，而且指向同一个张量，这其实是权重共享方案（具体见attention is all you need)，于是我们添加此方案
# 同时权重共享策略也可以节约参数的数量（因为分别都是768*50257)
# 8. 接下来小心的初始化初始化参数，遵循gpt2的方式，定义在GPT类中定义_init_weights方法,并在init方法中调用apply方法（注意初始化的标准差0.02比较符合xviar初始化。即输入特征数平方根的倒数1/aqrt(768),1/sqrt(1600)
# 9. 在gpt-2论文中提到初始化时除以层数的平方根，控制前向传播中残差流内激活增长的方法，在CausalSelfAttemtion中添加self.c_proj.NANOGPT_SCALE_INIT = 1，设置一个flag。然后在GPT内_init_weight方法里修改初始化。（注意，残差次数时layer的两倍）
# 10.由于我们共享了WTE和LM头部权重，在旧的子模块中，我们实际上会两次访问那个张量，首先嵌入时会用0.02标准差初始化，回到线性层，我们还会再次初始化，会被初始化两次
# 11.设置随机数种子，训练
# 于是我们实现了GPT2的初始化方法，并且有了dataloader,可以训练了。下一步我们准备加快训练速度

# 加大训练速度
# 1. 首先思考你有怎样的硬件，它提供了什么，是否充分使用了它---- 在终端输入NVIDIA-SMI 显示了8个A100的信息
# 2. 提到远程通过vscode连接Lambda,是一个AI developer cloud,国内应该有其他方案
# 3. Pytorch 默认创立张量是float32精度， 每一个数字，激活值，权重等，都使用32位浮点表示，占用了大量内存，但实际证明，在深度学习这样的任务中，这远远超出了所需。深度学习和这些网络的训练可以容忍显著更低的精度。降低精度可以获得显著的算力提升。
# 不过注意,int8用于推理，不要用于训练，int8具有均匀的间距，我们实际需要一个浮点数更好的匹配神经网络训练期间出现的正太分布，其中激活值和权重都呈现正太分布，因此浮点数对于这种匹配非常重要。
# 但除此之外，如果这些数字have fewer bits of representation,那么就会更容易的移动（这里开始设计内存带宽和内存的地方）我们不仅有一个有限的GPU存储位数容量，而且访问这个内存还有一个速度问题，还有内存带宽。
# 许多用于训练的深度学习工作负载是受内存限制的。实际上进行这些极快的乘法张量核心(tensor core)大部分时间都在等待，大部分时间处于空闲状态，我们无法以足够快的速度向它们提供数据，我们无法从内存中快速加载数据。如果硬件利用率到达60%,就做的很好
# 内存带宽也很重要。如果我们降低所有浮点数精度，突然之间需要的内存就减少了，因此我们可以储存更多数据，更快访问，一切都加速了。
# 什么是tensor core?具体推荐去阅读NVIDIA A100 Tensor Core GPU Architecture 白皮书。是A100架构中的一个指令，使用4*4矩阵乘法（里面有很多配置，调节精度的选项，开关等）。
# 然后任何需要矩阵乘法的操作都会被分解成4*4的乘法指令（因为举证乘法很快）大部分计算工作发生在线性层（特别是这个GPT2的例子，最大的举证乘法实际是顶部的分类器，（768-50257.矩阵乘法变快了，它们隐藏在我们的线性层中，并通过TensorCourse得到了加速
# import time 来计时，在训练中计时。但注意，我们在GPU上运作时，它只在GPU上调度工作，相当于发送一个指令，然后继续运行。会快速跳过GPU运行的代码实际上我们为GPU建立了一个队列，需要的话，等待torch.cuda.synchronize。并计算每秒处理的tocken数量并打印
# 用torch.set_float32_matmul_precision('high').我们希望Pytorch在每个我们看到的nn.linear的地方能够在tensor_core里面运行，利用tf32精度（注意原来是FP32）
# 4. 分析结果，没有获得理论的8倍，因为所有的数字仍然是float32类型，这些float32类型通过内存系统在各处传输，仍然受限于内存。但我们仍然获得了3倍吞吐量的提升。注意FP16和FP32的区别。
# FP16不能表示FP32完整的范围，它有一个缩小范围（涉及到梯度缩放器，比较复杂）BP16简单得多
# 5. 接上，混合精度训练（具体见文档Automatic Mixed Precision)。with torch.autocast(device_type = device, dtype = torch.float16)上下文管理器，只用管理前向传播和loss计算，其他不用管。
# 不过只有部分会选择性地转换type为bfloat16比如矩阵乘法的部分，因为矩阵乘法对精度比较鲁棒
# 6. torch.compile(model): model = torch.compile(model)会让代码运行加快,几乎必用（加速主要来自减少python开销和GPU读取次数:
# （1）torch.compile会看到所有代码，他会移除pyhton解释器在前向传播的作用，将整个神经网络编译成一个不涉及python解释器的单一对象。
# （2）：读写改进:每一次运算在gpu和内存之间切换非常耗时间，它会
# 7. 1:57..:芯片，即gpu,是基本上所有计算发生的地方，gpu也有一些内存，但大部分内存在HBM，与gpu在空间上独立,但相连，可通信。在芯片上，有许多streaming multiprocessors(流式处理器),每一个都是SM,总共有120个。但除此之外，在芯片上，内存遍布整个芯片。
# L2缓存(cache)是位于芯片上的一定量的内存,在SM上，有l1(cache),还有很多寄存器。
# 但这种内存储存方式与HBM中的方式中的储存方式非常不同。磁盘的内存和GPU相连的HBM访问其非常expensive。然后在芯片本身，芯片本身所有东西都非常快，但是只有几十MB的内存。所以芯片上的内存非常昂贵,空间不足。
# 当我们有这些内核时,HBM更准确的描述，计算时，我们取这些默认储存在全局内存中的输入，进行一些计算，我们开始从全局内存向芯片传输数据，在芯片上计算，然后将结果传回并储存回全局内存。
# 因此，如果我们没有torch.compile,我们会通过芯片传输数据并保存到内存中，并且进行这些往返传输很多次。但如果通过torch编译的，我们像之前那样传输数据，但当我们在芯片上时，我们有chunk of data需要处理。现在那部分数据位于芯片上，操作速度很快。
# 因此如果我们有内核融合，我们可以以逐元素的方式在那里完成操作，并且操作成本很低。然后我们只进行一次往返传输并回到全局内存。
# 所以操作融合允许将数据保留在芯片上，并在写回之前进行大量运算，带来了巨大的节省
# 8. 但又一些操作torch.compile无法find ,一个amazing的例子是flash attention.它非常注重内存的层次结构，它非常注意哪些在高带宽内存中，哪些在共享内存中，并且非常谨慎的安排计算，减少对高带宽内存的读写。
# 它不实际生成端到端注意力矩阵（ATT）（这是一个非常大的矩阵，有上百万个数字），使得这个矩阵在任何时候都不会被具体化，不会被写入到(HBM)中。flashattention,flashattention2,基本算法依赖于之前提出的online softmax技巧（nvidia的paper首先提出假设），通过evaluate softmax而不用实际计算softmax归一化
# 9. 从数字上优化，优化参数数值，尽量为2的整数幂，比如将词汇表大小又    50257调整到50304，计算对齐2的整数次幂。加快速度原因:因为很多kurnel的block tiles的通常是很好的数字如62,32,如果计算时能够对齐这些数字，就会加快。因为如果数据不整齐，kernel会截断输入，第一阶段阶段的数据能够对齐，第一阶段完后，再处理第二阶段不好的数据。所以尽量不要启动第二阶段低效的计算
# 10. 接下来，我们使用torch.compile,它允许我们编译模型，并获得显著的加速。（AI edit)

# 目前我们提速了11倍，我们接下来算法改进和对实际优化本身的提升。gpt2的paper信息有限，代码model.py只是推理，信息有限，于是看gpt3
# gpt3和gpt2的架构相似，上下文长度从1024提升到2048,还有一些transformer的超参数发生变化。只是gpt3在更大数据集训练，并且有了更全面的评估
# 1. 参考论文，设置optimizer的超参数
# 2. 在loss.backward()后进行梯度裁剪，并打印可视化通常限制梯度的最大范数,对所有参数的梯度平方求和开根，这就是参数向量的范数, norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)，防止sample到不好数据，梯度流过大，让模型crush（梯度裁剪是一种取巧的方法，就像是给一个很深入的问题打补丁
# 3. 调整学习率，论文中也提到。不用固定学习率，而用余弦衰减学习率.此步骤在训练的for循环里调整，通过设置get_lr函数（Karpathy说pytorch本来有对应的方法，但他说他喜欢用自己设定确定的方法）。也有很多其他提出的学习率，这仍然是一个很活跃的领域
# 4. 接下来，论文提到了逐步增加批量大小的方法。所以批量有一个线性增长的阶段。从非常小的batch开始，逐渐增加到较大的批量。我们实际会跳过这一步，不会采用这种方法。因为会是计算变得复杂，每一步处理的token数量都在发生变化，我们希望保持数学计算的简单（据karpathy了解，这不是算法上的改进，更多是系统性能和速度上的提升，
# 大致来说，因为优化过程早期阶段，模型处于一个非常不典型的阶段(untypical),我们主要学习如何忽略在训练集中不经常出现的token,我们在学习非常简单的的biases。所以通过网络处理每一个样本，基本上都是在告诉模型使用这些token,不要用这些token。因此来自每一个样本的梯度实际上是高相关在初始化阶段的。
# 由于梯度都是非常相似并且高度相关，为什么还在训练初期使用百万大小的批次呢？如果用32k，基本上也能得到完全相同的梯度。而在后期，挑战才开始，才使用统计学相关方法
# 5. 权重衰减提供少量的正则化，因为我们将所有权重都拉下来，迫使使用更多的权重，不允许任何一个权重变得过大，迫使在网格上更多的通道上分配工作（因为权重本身就像一个引力池）。
# 我们重新在GPT内部设置configure_optimizers函数，返回优化器对象。然后训练时调用这个方法构建构造器。具体而言，那个方法里，我们将参数分成了那些应该进行权重衰减和不用进行权重衰减的。
# 不衰减偏执和一些其他的一维张量很常见。一维张量被划分在了不衰减的参数里，这些还包括layernorm的尺度和偏置，对这些进行衰减没有意义。我们主要想对参与矩阵乘法的权重进行权重衰减，以及对embedding矩阵进行衰减
# 6. 接下来复刻批次大小，原始论文中batch_size是0.5M(以token计数)，所以我们试图将B设置成0.5M/1024=488,不过GPU会炸掉，但是我们又想复刻，因为其他的优化超参数，学习率参数都与此适应。
# 于是我们使用所谓的梯度积累，它允许我们以串行(serial way)的方式模拟任意批次的大小，所以我们就可以使用0.5M的batch_size,我们只用处理多个序列，把梯度累加加起来，模拟50万批次的大小。我们选择合理的数值，比如524288,保留小B= 16 , T = 1024,所以有524288/(16*1024) = 32串,具体代码实现就在训练步骤那里加个for循环,梯度会积累。
# 但有一个问题，梯度积累基本上会等同于在损失中进行求和，所以实际上，应该没有除以数量, 缺少归一化因子.只要有最后平均的步骤，进行分布式梯度积累，都会出现类似的问题，因此梯度出现了偏差。用loss = loss / grad_accum_steps。还要积累loss打印，用loss.detach(),防止增加不必要的计算图以及干扰梯度计算


# 接下来重型武器:多个GPU训练，分布式数据并行
# 1. 有8个GPU，将启动8个进程，每个进程被分配一个GPU。对于每个GPU,都只是在处理我们所构建的部分（有一点不同)，一旦各自计算出了梯度，就会对这些梯度进行平均。这就是它们如何协作处理计算负载的方式
# 2. 为了使用8个GPU,我们不再仅仅用pytorch train-gpt2.py。我们将使用一个pytorch中的特殊的命令,torch.run.torch.run实际会保存并行8个实例，并且创建了环境变量，每个进程都可以查找。
# 比如,torch.run会设置RANK,LOCAL_RANK ,WORLD_SIZE的环境变量。这是一种检测DDP是否运行的糟糕方法。有一个rank,每个进程基本上会大致相同的时间运行完全相同的代码。
# 所有进程之间唯一的区别是它们都有一个不同的DDP rank,比如GPU 0会有DDPrank0,GPU1会有rank1。
# 它们都运行相同的脚本，只是DDPrank会是一个稍微不同的整数，这就是我们协调它们不运行相同数据的方式，我们需要它们运行在数据的不同部分。
# 3. local rank是在多节点设置中才会用到的东西。我们只有一个节点，带8块GPU。因此local rank是单个节点上GPU的rank,例如，从0-7。 但对我们来说，我们主要在单个机器上运行，所以我们关注rank(根据运行脚本实例的运行脚本特定实例的GPU而有所不同）和world_size(8)
# 4. 我们根据local rank,将设备设置成cuda:表示。如果有多个GPU,使用哪个GPU。所以，根据进程的local rank,它将使用适当的GPU
# 5. 创建布尔变量，即DDP rank等于零。因此，祝进程是任意的进程编号零，它负责大量的打印，日志记录，检查点等。而其他进程出要被视为辅助计算过程。主进程0会有一些额外的工作去做，所有其他进程只负责前向和后向的计算。
# 如果我们不使用DDP且这些变量都没有设置，我们就会回到单GPU训练。
# 6. 然后将grad_accum_steps调整，除以一个ddp_world_size(将其他B*T的部分换成B*T*ddp_world_size
# 7. 然后用if master_process: 调整打印日志，只有在主进程才打印，
# 8. 接下来，我们创建数据加载器，我们需要让它意识到这个多进程，我们不需要所有进程加载完全相同的数据，我们虚妄每个进程都能获取自己的数据块，这样他们都在处理数据集的不同部分。
# 一个简单的方式是将process_rank = ddp_rank， num_processes = ddp_world_size传递给DataloderLite,相应的在DataloderLite类中,初始化方法中添加对应的初始化
# 9. 接下来，我们将模型实际包装到分布式数据并行容器中，DDP方法.pytorch有DistributedDataParallel详尽文档，
# 10. DDP做的是，在前向传播中，它的行为实际不变，在反向传播，在最简单的setting，一旦每个独立的GPU完成了反向传递，每个独立的GPU就拥有了所有的参数梯度，DDP做的事，一旦反向传播结束，它会调用所谓的allReduce,基本上是对所有等级上的梯度进行平均，然后它会静这个平均事存放在每一个等级上，所以每一个等级都会得到这个平均值，在反向传播进行的同时，为梯度分派通信。因此，梯度的通信和它们的同步和反向传递之间存在重叠，更有效率。
#总结来说，正向传播不变，反向传播不变，只是在上面加了平均值
# 11. 转向优化部分，基本上，在做loss.backward()时，会计算梯度，并且同步。但由于梯度累积步骤循环，我们实际上不想在loss.backwaed之后进行同步，我们只是在累积。
# 我们是串行进行的，我们只只是希望它们累积，而不想每次都同步，这样会非常浪费资源。
#于是pytorch有了no_sync()这个上下文管理器，用于在DDP进程。在这个上下文中，梯度将积累，不会有通信。然后要求我们再次使用DDP处理另一个输入并进行反向传递。
# Karpathy真的不喜欢这个，必须复制粘贴代码并使用上下文管理器的。我么只希望这个变量在最后是为true,在所有其他微步迭代中为false.(具体见训练时的if ddp代码)。一旦结束，就会神奇的储存在所有rank上梯度的平均值
# 12. 接下来平均损失，当我们调用all_reduce时，就会平均所有loss到所有rank上,运行
# 13。 修改原来的optimizer,由于现在时ddp模型        
# 14. 发现step个数不一致，单个GPU 32个，多个GPU则不一样。原因是数据加载器我们以稍微不同的方式加载，因为我们寻找一整页的数据。。。。。（听不懂）让总批次变小，就会减少数据加载器的边际效应
# 15. 数据集：common crawl ; webtex2; books1 ;books2;但这些数据集GPT3还没有发表；但有一些其他不错的数据集:red pajama data/C4/github/arxiv/。提到新出的FineWebDataSet。
# 这是一个尝试，基本上是为了收集高质量的CommonCrawl/HuggingFace发布的FineWeb-Edu。
# 推荐我们去读文档FineWeb: decanting the web for the finest text data at scale很精彩的讲述了如何处理数据。我们不会在万亿个token上进行训练
#我们将使用FineWeb-edu sample-10BT进行训练，下载，处理，确保我们的dataloder能够处理
# 16. 这里介绍一个文件，fineweb.py 会从huggingface数据集中下载FineWedEDU,它会预处理(pre-process)和预标记(pre-tokenize)所有的数据，并将数据分片(save data shards)保存在磁盘上的一个文件夹中。
# 运行那个脚本后，可以看Dataset Vewer，以便了解发生了什么。这里有过滤器(llama 370b),基本上是LLMs在判断哪些内容是有教育性的。
#关于这个脚本（不会细讲），首先会加载数据集，通过huugging_face的代码来完成。需要通过pip安装datasets库，它下载了数据集，然后对数据集所有的文档进行tokenize,首先用文本结束标记开始标记（尽管它叫end of text)。
# 这是GPT-2tokenizer中的一个特殊标记.然后我们extend所有标记tokens,extend()。保存所有内容到分片(to shards),这些shards是NumPy文件，所以只是储存一个Numpy数组，和Torch很相似。有一个验证分片，其余都是训练分片
#这大概需要运行30分钟，然后在这个数据上进行训练。在这种情况下，我么会进行一些正规的预训练，（一个好的数据集，每秒处理大量标记，算力充足，8块GPU，代码准备好了）



#开始训练
#1. 查看数据集。ls edu_fineweb10B/ | wc -l有100个分片 (改变了数据集，注意不再是莎士比亚)
#2. 在DataloderLite加入split
#3. 接下来设置一些初始值：我们每步处理2**19次方的标记，我么需要处理10e9(100亿）个token，我们有那么多的的unice tokens。然后计算10e9/2**19得到大约19073步。
#GPT3在论文中说到他们在3.75亿个标记上进行预热学习率，375e6(3.75亿)/2**19=715steps(warmup_steps)
#4. 然后就可以进行预训练了，看日志，每一个step需要花费330ms,一共有19073步，所以大概要花19073*0.33这么多秒。（当前B设置成64不会进行梯度积累-grad_accum_steps,
#因为2**19/(64*T*GPU个数ddp_woeld_size),如果GPU没那么大的容量可以减少B值。这里的训练包括是学习率预热到最大值

#接下来我们要做到尽善尽美，在验证集上评估，并尝试弄清楚如何评估，如何进行日志记录，如何可视化损失，
#1. 设置训练集，验证集，并以split划分
#2. 在dataloader设置reset方法，会重置dataloader。这很有用，因为我们进入进入训练的住循环，每(100)次会进入推理模式，会重置loader，并且不涉及梯度，只是测量累积损失
#3. 谈论过拟合:上面的这种分割验证就可以查看是否在训练时过拟合了
#4. hello swag 验证:生成的句子是采用对抗性来源，对人类来说很简单，对大模型来说很困难。刚发表时的语言模型的正确率只有50%,但在GPT3时代已经百分之九十多的正确率了
#hello swag是一种平滑的评估工具，是一个所谓提供早期信号的评估，早期信号意味着即使是早期的语言模型，也会看到缓慢的提升，25%，26%，27%。
# （随机选择的正确率是0.25),gpt2的正确率是29%，(当前最sota的是九十多%)。所以我们试图打败gpt2
#5. 接下来又做了一些选项，将编译设置成可选，并且默认禁用。因为编译确实能让我们的代码运行更快，但是会破坏evaluation代码和采样代码
# (当时karpathy就说他遇到了他也看不懂的报错，没有解决，于是就没用编译，用cuda。并且说希望github上应该解决了这个问题）
#6. 在验证，采样，hello swag评估后，才有一个训练的过程，当我们进行训练得到损失后，我们将其写入一个文件
#7. 在play.ipynb里，与gpt2比较，比较训练损失和测试损失均超过了gpt2,但这并不公平，因为gpt2的训练集我们不知道。
#8. 比较hello swag是比较公平的。图中显示我们的模型超过了openai的gpt2,但远不如gpt3，值得注意的是，我们的模型是在100亿个tokens,而gpt2是在1000亿个
#tokens训练，因此，出于某些原因，我们能够用明显更少的标记进行训练，可能解释的原因为
#1. openai的gpt2是在更广泛分布的数据集上训练的（数学，多语言，代码，而不仅仅是英文），这些很杂的内容占据了模型容量，可能导致模型不佳的部分原因。之前的训练资料就是webtext，而现在
#的资料在去重，过滤，质量等方面有了更好的实践和审查，我们在训练集的每个token的质量可能更高
#2. helloswag可能很老了，甚至以某种方式进入了Fineweb的训练集
#9. 分析损失函数，看到有些奇怪，不平滑，可能是因为我们没有打乱，比较懒惰，可能继承了数据集中的顺序
#10. karpathy希望在之后会解决上面的一些问题，在仓库里的Errata会显示
#11，重新运行8小时，发现损失函数出现周期性的异常（每一个epo都出现），说明FineWebEDU数据集确是出现了奇怪现象。
# HelloSwag的表现直线上升，差点到达gpt3。gpt3用了3000亿的token,而我们仅仅用了400亿的token 
#顺便说一下，dataloader加载数据按照次序的，如果我们每次都打乱，增加随机行，对训练也会有好处，就不会看到哪些周期性的异常。
#但是训练集是一个文本，本来就应该有某些顺序，我们不应该打乱。可以尝试打破这种依赖
#12. 最大学习率实际上可以设置得更高，我么继承gpt3的参数过于保守了，我们实际上可以试试更高，训练速度会加快。其他超参数也可以尝试，不一定最好
#13. 注意如果想完全遵循gpt3,T为2048,B设置成32，其余的架构gpt2和gpt3非常相似
#14. 最后又增加了if master——process的逻辑，每5000步记录验证损失，检查点，实际上就是模型的字典状态。
#如果想resume the optimization，在保存模型的同时，还会保存优化器状态字典，（比如adam还有一些额外的buffer,)
#还必须小心处理seed
#15. 还提到换成其他评估的方式比如eleuther
#16. 当前只是预训练模式，不能想chatgpt那样聊天。如果想和模型聊天，需要将其微调成聊天格式。
#这并不复杂，如果关注监督微调（SFT），这实际上意味着将数据集更换成一个更加对话式的数据集，其中有user-assistant-user-assistant这样的结构，
#我就在这个数据集上进行fine-tune，填充用户token, 采样用户token.其实并没有深入，只是替换数据集并继续训练.


#总结
#1. 我们这次的项目是朝着Nano-GPT迈进的，但事实上还有另一个NanoGPT的实现，它隐藏在一个名为llm.c的文件中，（train-nanogpt.py?)
#拓展的有train_gpt2.cu,这是c_cuda implemention，涉及很多MPI,NICL,GPU,cuda ,c c++的知识，如果只是训练gpt2或gpt3的话，选择它会很好
#2. 我们研究了gpt2和gpt3,我们探讨了如何设置这些训练运行以及设计所有的考量因素，我们从零开始编写了一切，无论两小时的运行还是一夜的运行，我么很大程度的匹配了gpt2和gpt3
#原则上，我编写的代码如果有足够的耐心和资源，就可以训练一个更大的模型，所以我们可以考虑训练一个更大的checkpoint的模型
#3. 还有一些剩余的问题需要解决，比如这里损失值的变化，我怀疑与FineWebEDU的数据的采样有关。为什么在hello swag不能用torch.compile

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embed),
            }
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = (
            model.state_dict()
        )  # 把模型可以学习的参数提取出来，储存到sd,以字典的形式储存
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {
            pn: p for pn, p in self.named_parameters()
        }  # 是从 nn.Module 继承而来的方法，用于遍历当前模型中所有可训练的参数，同时返回每个参数的名字和张量
        param_dict = {
            pn: p for pn, p in param_dict.items() if p.requires_grad
        }  # 应该是过滤冻结参数
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(
            p.numel() for p in decay_params
        )  # p.numel() 是 PyTorch 中 Tensor 或 Parameter 的一个方法，表示：“这个张量中总共有多少个元素”（即它的“长度 × 宽度 × 深度 × …”）
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)这行有点问题，判断了个寂寞
        optimizer_args = {
            "params": optim_groups,
            "lr": learning_rate,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
        if fused_available and device_type == "cuda":
            optimizer_args["fused"] = True  # 只有在支持 fused 参数时才加进去

        optimizer = torch.optim.AdamW(**optimizer_args)
        return optimizer


class Block(nn.Module):
    def __init__(self, config):
        super().__init__(self)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__(self)
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


# ===========================================================================================================
# dadaloder
# =============================================================================================================
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y



# ==================================================================
# attempt to 推理
device = "cpu"

if torch.cuda.is_available():
    device = "cuda"

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"using device: {device}")

# device = "cpu"  # Override


num_return_sequences = 5
max_length = 30
# 推理阶段
# model = GPT.from_pretrained("gpt2")
# print("not crush")
# model.eval()
# model.to('cuda')
# 没有区分训练和评估的层

# ===========================================================================================================
# 只有一个batch,过拟合一个batch的样本
# =============================================================================================================
# #tocken
# import ticktoken

# enc = ticktoken.get_encoding('gpt2')
# # tockens = enc.encode("hello,I'm a language model,")
# # tockens = torch.tensor(tockens,dtype=torch.long)
# # tockens = tockens.unsqueeze(0).repeat(num_return_sequences,1)#(5,8)
# # x = tockens.to('cuda')
# with open('input.txt', 'r') as f:
#     text = f.read()

# text = text[:1000]
# tokens = enc.encode(text)
# B,T = 4,32
# buf = torch.tensor(tokens[:B*T +1])
# buf = buf.to(device)#注意张量类型是非原地改动，需要覆盖原来张量，这一点注意要和模型区分
# x = buf[:-1].view(B,T)
# y = buf[1:].view(B,T)

# 训练阶段的模型，随机初始化参数，得到logits,返回loss


# ===========================================================================================================
# 训练
# =============================================================================================================

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
B = 64  # micro batch size
T = 1024  # sequence length
assert (
    total_batch_size % (B * T*ddp_world_size) == 0
), "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")




# print("I am GPU", ddp_rank)
# print("Bye")


# import sys

# sys.exit(0)



torch.set_float32_matmul_precision("high")

# create model
model = GPT(GPTConfig(vocab_size=50257))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
model = torch.compile(model)

use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix 
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])

raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073# 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens


# logits, loss = model(x,y)
# print(logits.shape)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimization
# optimizer = torch.optim.AdamW(model.parameters, lr=3e-4,betas=(0.9,0.95),eps = 1e-8)
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device
)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile): # 每250步或者最后一步进行评估，不使用编译，因为编译会破坏evaluation代码和采样代码，并且编译会破坏HellaSwag的评估，
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()









# for step in range(max_steps):
#     t0 = time.time()
#     optimizer.zero_grad()  # 一定注意梯度清零，防止累计梯度
#     loss_accum = 0.0
#     for micro_step in range(grad_accum_steps):
#         x, y = train_loader.next_batch()
#         x, y = x.to(device), y.to(device)
#         with torch.autocast(device_type=device, dtype=torch.bfloat16):
#             logits, loss = model(x, y)

#         loss = loss / grad_accum_steps
#         loss_accum += loss.detach()
#         if ddp:
#             model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
#         loss.backward()
#     if ddp:
#         dict.all_reduce(loss_accum,op = dist.ReduceOp.AVG)
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     lr = get_lr(step)
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
#     optimizer.step()
#     torch.cuda.synchronize
#     t1 = time.time()

#     dt = (t1 - t0) * 1000  # time difference in milisecodes

#     tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
#     tokens_per_sec = tokens_processed / dt
    
#     if master_process:
#         print(
#             f"step:{i} | loss:{loss_accum.item()}| norm: {norm:.4f} | tok/sec:{tokens_per_sec}"
#         )  # loss是包含单个元素的张量，.item()将那个张量转化为单精度浮点数，这个浮点数会存在于cpu上，当调用这个方法，pytoech会将这个原来gpu上的张量送到cpu内存，并将其转换成我们可以打印的浮点数


# if ddp:
#     destroy_process_group()
# ===========================================================================================================
# 推理
# =============================================================================================================


import sys
sys.exit(0)


torch.manual_seed(114514)
torch.cuda.manual_seed(114514)

while x.size(1) < max.length:

    with torch.no_grad():
        logits = model(x)  # (B,T,vocab_size)

        logits = logits[:, -1, :]  # 取出最后一个，(B,vovab_size)
        # 得到概率
        probs = F.softmax(logits, dim=-1)
        # top-k
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1)  # (B,1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)

# ![image.png](attachment:image.png)

# print the generated text
for i in range(num_return_sequences):
    tockens = x[i, :max_length].tolist()
    decode = enc.decode(tockens)
    # print(">",decode)

