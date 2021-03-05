# Training command
python train_globalmemory2.py --config configs/zwh_vad_gloabal_memory2_unet_objectloss_avenue.yaml --gpu 1 --tag test0305

主要思路为：

将图像分为数个patch，如8*8个patch，每个patch对应一个memory，这个memory是Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection这篇文章的memory。即那篇论文只用了一个memory，我们这篇用了64个memory。

训练时，提取图像的目标，及其在图像上的全局位置。根据全局位置得到其对应的memory，然后read和update，得到新的memory和新的特征。

测试时也是如此。

这样的好处是，利用了目标的全局信息，并且通过多个memory使得能够更好地记录正常的模块。比如图像上的一个地方始终没有人出现的话，其对应的memory就没有被训练到，接近一个随机向量。这样在测试时那个位置如果出现了人，这个memory就会对重构预测产生负作用，从而更好检测异常。



