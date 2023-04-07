## ICF的实验设置

* normalize the ratings into the range [−1, 1]
* 用户分成两个不相交的部分，一部分作预训练，固定好item的向量后就不动了，另一部分作在线测试

  * 测试时item的隐向量不做更新
* 如何回答冷启动问题
* 跑20轮或者120轮 （T=20 or 120）

### 评价指标

$$
precsion@T = \frac{1}{\#user} \sum_u \sum_{t=1}^T \theta_{hit}
$$

$\theta_{hit}$如果用户评分大于等于4分认为其为1，否则为0.

$ recall@T $ 为 $\theta_{hit}$ 再除以用户满意的物品数 （T轮中 推荐到的满意的物品数 / 用户总共满意的物品数）

### NICF的实验设置

* 85%用来训练，5%用来作评估，10%用来测试
* These include latent dimensions d from {10, 20, 30, 40, 50}
* learning rate from {1, 0.1, 0.01, 0.001, 0.0001}
* 每个user跑40轮

### NGCF实验设置

70 ： 10 ：20
训练：验证（调参）：测试
