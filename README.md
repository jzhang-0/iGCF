TODO:

思路：
    选择和当前用户相关的用户和物品，聚合 equal ICF
    更好的协同

- [ ] 在save的cls中分析结果，调整参数（不理想的话，尝试调整正则项强度）
- [ ] 删除不需要分析的文件（save中的cls）

3.26:
    -[x] var(xy)公式推导
    -[x] 实现测试GCNICF

3.23:
    -[x] 按照后验分布来做

3.18~

- [X] 重新包装GCN函数，分模块，VI，优化的

3.9~

<!-- - [ ] 公式推好了，考虑用lasso或者人为选择特征进行优化. -->

2.14~

- [X] 修改gcnicf

2.11~

- [X] 修改NICF recall 代码
- [X] 封装数据(取消验证集)
- [X] 保存测试结果

1.28~

- [X] 修改modelbase，UI_cls
- [X] dataload 中数据集预处理增加一列 feedback01

- [pass] 合并 modelbase 中的online init func (兼容不同模型)

- [X] 预训练抽象
- [X] 整合具体的函数，实现model init和train两个方法

past~

- [X] 整理结果， 更多轮数（20000轮）在Kuairec上的结果（对比10000轮），ml-100k上的结果（不同算法比较）
- [X] 看下范数和流行度好评数好评率间的联系

继续做实验

<!-- - [ ] movielens 上的训练有很大的问题，考虑前期用pop得到正样本（难搞） / 考虑只推荐交互过的（随机算法不改） -->

- [ ] taste 的划分还未完成
- [ ] 考虑模拟实验
- [X] 预训练与online用户的划分。
- [X] 如何计算 cosine similarity between the genre vectors of the two periods
  （参考 Adaptive diversiﬁcation of recommendation results via latent factor portfolio）
  物品有类别向量0-1串表示，movielens里面是19个维度，算cosine就行

1.12~
    `<!-- - [ ] K>1时的效率问题 -->`

1.9~
    - [x] 实现LTS // 实现加噪声的核心部分。
    - [x] 测试gcn实验结果

12.19~
    - [X] 写light-gcn 代码
    - [x] 测试代码

12.11~
    - [x] 开始测试ICF
    - [x] 写评价指标
