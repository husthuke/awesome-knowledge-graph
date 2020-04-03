# awesome-knowledge-graph
整理知识图谱相关学习资料，提供系统化的知识图谱学习路径。

---

## 目录
- [理论及论文](#理论及论文)
- [图谱及数据集](#图谱及数据集)
- [工具及服务](#工具)
- [白皮书及报告](#白皮书及报告)
- [机构及人物](#机构及人物)
- [视频课程](#视频课程)
- [专栏合集](#专栏合集)
- [评测竞赛](#评测竞赛)
- [项目案例](#项目案例)
- [推广技术文章](#推广技术文章)


<!-- /MarkdownTOC -->
## 理论及论文
### Survey


- [Knowledge Graph Construction Techniques](./paper/知识图谱构建技术综述_刘峤.caj)
- [Review on Knowledge Graph Techniques](./paper/知识图谱技术综述.pdf)
- [Reviews on Knowledge Graph Research](./paper/知识图谱研究综述-李涓子.pdf)
- [The Research Advances of Knowledge Graph](./paper/知识图谱研究进展_漆桂林.caj)
- [A Survey on Knowledge Graphs: Representation, Acquisition and Applications (2020)](https://arxiv.org/pdf/2002.00388.pdf)
- [Knowledge Graphs (2020)](https://arxiv.org/pdf/2003.02320.pdf)


### KG-Augmented LMs(知识图谱增强语言模型) 
知识图谱增强语言模型是最近两年比较流行，主要发生在BERT出来之后，将知识先验信息融入到语言模型，可以说是知识图谱助力NLP十分关键的一环，将该专题放在比较靠前的位置。

- [Latent Relation Language Models](https://arxiv.org/pdf/1908.07690.pdf)
- [K-BERT: Enabling Language Representation with Knowledge Graph](https://arxiv.org/pdf/1909.07606.pdf)
- [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)
- [ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding
](https://arxiv.org/pdf/1907.12412.pdf)
- [SENSEMBERT: Context-Enhanced Sense Embeddings for Multilingual Word Sense Disambiguation](https://pasinit.github.io/papers/scarlini_etal_aaai2020.pdf)
- [Inducing Relational Knowledge from BERT](https://arxiv.org/pdf/1911.12753.pdf)
- [Integrating Graph Contextualized Knowledge into Pre-trained Language Models](https://arxiv.org/pdf/1912.00147.pdf)
- [Enhancing Pre-Trained Language Representations with Rich Knowledge
for Machine Reading Comprehension](https://www.aclweb.org/anthology/P19-1226.pdf)
- [K-ADAPTER- Infusing Knowledge into Pre-Trained Models with Adapters](https://arxiv.org/pdf/2002.01808)
- [Knowledge Enhanced Contextual Word Representations (EMNLP 2019)](https://arxiv.org/abs/1909.04164)
- [KEPLER: A Unified Model for Knowledge Embedding and
Pre-trained Language Representation (2020)](https://arxiv.org/pdf/1911.06136.pdf)


### Representation&Embedding（表示&嵌入）

- [Knowledge Representation Learning: A Review](./paper/知识表示学习研究进展_刘知远.caj)
- [Holographic embeddings of knowledge graphs](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828)
- [Context-dependent knowledge graph embedding. EMNLP 2015. Luo, Yuanfei and Wang, Quan and Wang, Bin and Guo, Li.](http://www.aclweb.org/anthology/D15-1191)
- [GAKE: graph aware knowledge embedding. COLING 2016. Feng, Jun and Huang, Minlie and Yang, Yang and Zhu, Xiaoyan.](http://www.aclweb.org/anthology/C16-1062)
- [Bootstrapping Entity Alignment with Knowledge Graph Embedding. IJCAI 2018. Zequn Sun, Wei Hu, Qingheng Zhang and Yuzhong Qu.](https://www.ijcai.org/proceedings/2018/0611.pdf)
- [KBGAN: Adversarial Learning for Knowledge Graph Embeddings. NAACL 2018. Cai, Liwei, and William Yang Wang.](https://arxiv.org/pdf/1711.04071.pdf)

### NER(命名实体识别)

### Entity aligning(实体对齐)
- [A Survey on Entity Alignment of Knowledge Base](./paper/知识库实体对齐技术综述.pdf)
- [Knowledge Graph Alignment Network with Gated Multi-hop Neighborhood Aggregation](https://arxiv.org/pdf/1911.08936.pdf)
- [Coordinated Reasoning for Cross-Lingual Knowledge Graph Alignment](https://arxiv.org/pdf/2001.08728.pdf)


### KG Completion(图谱补全)
- [Differentiable Reasoning on Large Knowledge Bases and Natural Language](https://arxiv.org/pdf/1912.10824.pdf)
- [Diachronic Embedding for Temporal Knowledge Graph Completion](https://arxiv.org/pdf/1907.03143.pdf)
- [Commonsense Knowledge Base Completion with Structural and Semantic Context](https://arxiv.org/pdf/1910.02915.pdf)
- [KG-BERT: BERT for Knowledge Graph Completion](https://arxiv.org/pdf/1909.03193.pdf)

### Reasoning(推理)

- [ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning](./paper/ATOMIC-An_Atlas_of_Machine_Commonsense_for_If-Then_Reasoning.pdf)
- [Reasoning on Knowledge Graphs with Debate Dynamics](https://arxiv.org/pdf/2001.00461.pdf)


### 知识库问答-KBQA

- [Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/pdf/1909.05855.pdf)
- [Improving Knowledge-aware Dialogue Generation via
Knowledge Base Question Answering](https://arxiv.org/pdf/1912.07491.pdf)
- [Graph-Based Reasoning over Heterogeneous External Knowledge for Commonsense Question Answering](https://arxiv.org/pdf/1909.05311.pdf)

### 动态或时序知识图谱

- [Learning Sequence Encoders for Temporal Knowledge Graph Completion](./paper/Learning_Sequence_Encoders_for_Temporal_Knowledge_Graph_Completion.pdf)

### Tracing(知识追踪)
本类别并不是传统知识图谱中的任务，而是与教育领域结合的广义上的知识图谱任务。

- [Knowledge tracing- Modeling the acquisition of procedural knowledge](./paper/Knowledge_tracing-Modeling_the_acquisition_of_procedural_knowledge.pdf)
- [Individualized Bayesian Knowledge Tracing Models](./paper/Individualized_Bayesian_Knowledge_Tracing_Models.pdf)
- [Deep Knowledge Tracing](./paper/Deep_Knowledge_Tracing.pdf)
- [Tracking Knowledge Proficiency of Students with Educational Priors](./paper/Tracking_Knowledge_Proficiency_of_Students_with_Educational_Priors.pdf)



## 图谱及数据集
### 开放知识图谱
#### 中文开放知识图谱(OpenKG.CN)

中文开放知识图谱（简称OpenKG.CN）旨在促进中文知识图谱数据的开放与互联，促进知识图谱和语义技术的普及和广泛应用，包括了众多的数据集以及工具。

- [官网地址](http://openkg.cn/)


### 领域知识图谱
#### 学术知识图谱AceKG

最新发布的Acemap知识图谱（AceKG）描述了超过1亿个学术实体、22亿条三元组信息，涵盖了全面的学术信息。具体而言，AceKG包含了61,704,089篇paper、52,498,428位学者、50,233个研究领域、19,843个学术研究机构、22,744个学术期刊、1,278个学术会议以及3个学术联盟（如C9联盟）。

同时，AceKG也为每个实体提供了丰富的属性信息，在网络拓扑结构的基础上加上语义信息，旨在为众多学术大数据挖掘项目提供全面支持。

- [访问地址(http://acemap.sjtu.edu.cn/)](http://acemap.sjtu.edu.cn/)

### 数据集

#### SQuAD


- [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

#### YAGO
YAGO是由德国马普研究所研制的链接数据库。YAGO主要集成了Wikipedia、WordNet和GeoNames三个来源的数据。YAGO将WordNet的词汇定义与Wikipedia的分类体系进行了融合集成，使得YAGO具有更加丰富的实体分类体系。YAGO还考虑了时间和空间知识，为很多知识条目增加了时间和空间维度的属性描述。目前，YAGO包含1.2亿条三元组知识。YAGO是IBM Watson的后端知识库之一。由于完成的YAGO数据集过于庞大，在使用过程中经常会选取其中一部分进行，比如可以抽取中带有时间注释（time annotations）的部分形成YAGO11k数据集。

- [完整数据集下载地址](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/)

#### WikiData
WikiData的目标是构建一个免费开放、多语言、任何人或机器都可以编辑修改的大规模链接知识库。WikiData由维基百科于2012年启动，早期得到微软联合创始人Paul Allen、Gordon Betty Moore基金会以及Google的联合资助。WikiData继承了Wikipedia的众包协作的机制，但与Wikipedia不同，WikiData支持的是以三元组为基础的知识条目（Items）的自由编辑。一个三元组代表一个关于该条目的陈述（Statements）。

- [WikiData中文部分-截至2017.01](http://openkg.cn/dataset/http-pan-baidu-com-s-1c2ovnks)


#### NLPCC 2017 KBQA

该任务来自NLPCC 2017评测任务，开放域问答评价任务主要包括三项子任务，基于知识库的问答（kbqa），基于文档的问答（dbqa），和基于表的问答（tbqa）。kbqa的任务是基于知识库的中文问题回答。dbqa的任务是通过选择一个或多个句子从一个给定的文档，作为答案回答中文问题。tbqa的任务是一个全新的QA任务，旨在通过从收集的表格中抽取一个或多个表回答英语问题。

[下载链接](https://biendata.com/ccf_tcci2018/datasets/tcci_tag/11)

#### GDELT
GDELT（Global Database of Events, Language, and Tone）是最大的综合人类社会关系数据库，以100多种语言监控来自每个国家几乎每个角落的广播、印刷和网络新闻，并确定推动我们全球社会的人、地点、组织、主题、来源、情感、计数、报价、图像和事件每天的每一秒，它的全球知识图将世界的人，组织，地点，主题，计数，图像和情感连接到整个地球上的单一整体网络。为整个世界的计算创建一个免费的开放平台。

[下载链接](https://www.gdeltproject.org)


#### ICEWS
ICEWS（Integrated Crisis Early Warning System）捕获和处理来自数字化新闻媒体，社交媒体和其他来源的数百万条数据，以预测，跟踪和响应世界各地的事件，主要用于早期预警。该数据集在知识图谱领域主要用于动态事件预测等动态图谱方面。

[下载链接](https://dataverse.harvard.edu/dataverse/icews)

#### OAG
OAG（Open Academic Graph包含来自MAG的166,192,182篇论文和来自AMiner的154,771,162篇论文，并生成了两个图之间的64,639,608个链接（匹配）关系。它可以作为研究引文网络，论文内容等的统一大型学术图表，也可以用于研究多个学术图表的整合。

[下载链接](https://www.aminer.cn/open-academic-graph)

## 工具
根据知识图谱的通用基本构建流程为依据，每个阶段都整理部分工具。
### 知识建模

### 知识抽取
#### Deepdive




### 知识推理

- [官网地址](http://deepdive.stanford.edu/)
- [Github地址](https://github.com/HazyResearch/deepdive)


### 知识表示
#### OpenKE
清华大学NLP实验室基于TensorFlow开发的知识嵌入平台，实现了大部分知识表示学习方法。

- [官网地址](http://openke.thunlp.org/)
- [Github地址](https://github.com/thunlp/OpenKE)

#### 知识融合


## 白皮书及报告
- [CCKS2018-知识图谱发展报告](./report/CCKS2018-知识图谱发展报告.pdf)
- [知识图谱标准化白皮书(2019版)](./report/知识图谱标准化白皮书(2019版).pdf)



## 机构及人物
本部分介绍在知识图谱领域前沿研究或者有一定影响力的机构以及个人。

#### 机构

#### 人物
- 李娟子:[清华大学网页](http://keg.cs.tsinghua.edu.cn/persons/ljz/)
- 刘知远:[清华大学网页](http://nlp.csai.tsinghua.edu.cn/~lzy/)、[知乎主页](https://www.zhihu.com/people/zibuyu9/activities)
- 漆桂林:[东南大学网页](https://cse.seu.edu.cn/2019/0103/c23024a257135/page.htm)
- 肖仰华:[复旦大学网页](http://gdm.fudan.edu.cn/GDMWiki/Wiki.jsp?page=Yanghuaxiao)
- 刘康:[中科院网页](http://people.ucas.ac.cn/~liukang)
- 刘挺:[哈工大网页](http://homepage.hit.edu.cn/liuting)
- 王昊奋:


## 视频课程

### 小象学院知识图谱课程
- [知识图谱](https://www.chinahadoop.cn/course/1048)


### 贪心学院知识图谱课程
- [知识图谱的技术与应用](https://www.greedyai.com/course/19/summary/introduce)
- [教你搭建一个工业级知识图谱系统](https://www.greedyai.com/course/30/summary/knowledgeMapProject)


### 炼数成金知识图谱课程

### CSDN视频课
- [知识图谱系统架构剖析](https://edu.csdn.net/course/detail/10286)
- [AI开发者大会——知识图谱专题](https://edu.csdn.net/course/detail/10284)




## 专栏合集

#### 知乎集合

#### 简书集合

## 评测竞赛
- [“达观杯”文本智能信息抽取挑战赛](https://www.biendata.com/competition/datagrand/)
- [CCKS 2019 公众公司公告信息抽取](https://www.biendata.com/competition/ccks_2019_5/)
- [CCKS 2019 医疗命名实体识别](https://www.biendata.com/competition/ccks_2019_1/)
- [CCKS 2019 医疗命名实体识别](https://www.biendata.com/competition/ccks_2019_4/)
- [CCKS 2019 人物关系抽取](https://biendata.com/competition/ccks_2019_ipre/)
- [CCKS 2019 中文短文本的实体链指](https://biendata.com/competition/ccks_2019_el/)
- [CCIR 2019 基于电子病历的数据查询类问答](https://www.biendata.com/competition/ccir2019/)
- [瑞金医院MMC人工智能辅助构建知识图谱大赛](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.457933afBacvqN&raceId=231687)
- [CCKS 2018 面向中文电子病历的命名实体识别](https://www.biendata.com/competition/CCKS2018_1/)
- [CCKS 2018 面向音乐领域的命令理解任务](https://www.biendata.com/competition/CCKS2018_2/)
- [CCKS 2018 微众银行智能客服问句匹配大赛](https://www.biendata.com/competition/CCKS2018_3/)
- [CCKS 2018 开放领域的中文问答任务](https://www.biendata.com/competition/CCKS2018_4/)
- [CCKS 2017 问题命名实体识别和链接任务](https://www.biendata.com/competition/CCKS2017_1/)
- [CCKS 2017 面向电子病历的命名实体识别](https://www.biendata.com/competition/CCKS2017_2/)

## 会议交流及讲座
### AICon
- [AICon 2017知识图谱技术实践](https://aicon2017.geekbang.org/training/1)
- [AICon 2018知识图谱深度培训](https://aicon2018.geekbang.org/training/1315)

### BDTC
- [BDTC 2018 医疗知识图谱的构建和应用](./conference/医疗知识图谱的构建和应用.pdf)
- [BDTC 2018 从知识图谱到人工智能：产品演进路径上的思考](./conference/从知识图谱到人工智能-产品演进路径上的思考.pdf)
- [BDTC 2018 基于cnSchema的大规模金融知识图谱实战](./conference/基于cnSchema的大规模金融知识图谱实战.pdf)
- [BDTC 2017 Event Extraction from Texts](./conference/Event_Extraction_from_Texts.pdf)
- [BDTC 2017 知性会话：基于知识图谱的人机对话系统方法与实践](./conference/知性会话-基于知识图谱的人机对话系统方法与实践.pdf)
- [BDTC 2017 基于图的海量知识图谱数据管理](./conference/基于图的海量知识图谱数据管理.pdf)
- [CSDN AI 2018 医疗知识图谱的敏捷构建和实践](./conference/医疗知识图谱的敏捷构建和实践.pdf)
- [CSDN AI 2018 知识图谱的表示和推理](./conference/知识图谱的表示和推理.pdf)
- [CSDN AI 2018 大规模通用知识图谱构建及应用](./conference/大规模通用知识图谱构建及应用.pdf)
- [CSDN AI 2018 大规模通用知识图谱构建及应用](./conference/大规模通用知识图谱构建及应用.pdf)



### 其他
- [知识图谱中的深度学习技术应用概述](https://v.qq.com/x/page/i0700c29hw1.html)
- [2018云栖大会上海-人工智能专场](https://yunqi.youku.com/2018/shanghai/review?spm=a2c4e.11165380.1076033.1)
- [AI研习社-知识图谱的嵌入：更好更快的负采样](http://www.mooc.ai/open/course/640)

## 项目案例
### 教育领域知识图谱

### 金融领域知识图谱

#### 利用网络上公开的数据构建一个小型的证券知识图谱/知识库
- https://github.com/lemonhu/stock-knowledge-graph.git

#### 上市公司高管图谱
- https://github.com/Shuang0420/knowledge_graph_demo

### 医疗领域知识图谱

### 农业领域知识图谱

#### 使用爬虫获取Wikidata数据构建
- https://github.com/CrisJk/Agriculture-KnowledgeGraph-Data.git


### 知识工程领域知识图谱

### 其他知识图谱

#### 红楼梦人物关系图谱
- https://github.com/chizhu/KGQA_HLM

#### 通用领域知识图谱
- https://github.com/Pelhans/Z_knowledge_graph

#### 免费1.5亿实体通用领域知识图谱
- https://github.com/ownthink/KnowledgeGraph

#### 简易电影领域知识图谱及KBQA系统
- https://github.com/SimmerChan/KG-demo-for-movie


## 推广技术文章
### 2016
- [构建 LinkedIn 知识图谱
](https://www.infoq.cn/article/constructing-linkedin-knowledge-map)


### 2017
- [阿里知识图谱首次曝光：每天千万级拦截量，亿级别全量智能审核](https://mp.weixin.qq.com/s/MZE_SXsNg6Yt4dz2fmB1sA)
- [百度王海峰：知识图谱是 AI 的基石](https://www.infoq.cn/article/2017/11/Knowledge-map-cornerstone-AI)
- [哈工大刘挺：从知识图谱到事理图谱](https://mp.weixin.qq.com/s/1nl56AdZIkT03gnmimt8nQ)
- [智能导购？你只看到了阿里知识图谱冰山一角](https://www.csdn.net/article/a/2017-12-08/15937080)



### 2018
- [张伟博士：阿里巴巴百亿级别的三元组知识图谱掌舵者](https://www.shangyexinzhi.com/article/details/id-28524/)
- [知识图谱在互联网金融行业的应用](https://mp.weixin.qq.com/s/YeSzOw6dRNiX32PmdWgLow)
- [上交大发布知识图谱AceKG，超1亿实体，近100G数据量](https://mp.weixin.qq.com/s/qsRTBR5g5LZ6UR7Wtqagyw)
- [知识图谱数据构建的“硬骨头”，阿里工程师如何拿下？](https://yq.aliyun.com/articles/544941)
- [这是一份通俗易懂的知识图谱技术与应用指南](https://www.jiqizhixin.com/articles/2018-06-20-4)
- [一文揭秘！自底向上构建知识图谱全过程](https://102.alibaba.com/detail?id=134)
- [健康知识图谱，阿里工程师如何实现？](https://102.alibaba.com/detail?id=176)
- [为电商而生的知识图谱，如何感应用户需求？](https://yq.aliyun.com/articles/632483)
- [肖仰华谈知识图谱：知识将比数据更重要，得知识者得天下](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/83451280)
- [知识图谱在旅游领域有哪些应用？携程度假团队这样回答](https://mp.weixin.qq.com/s?__biz=MjM5MDI3MjA5MQ==&mid=2697267537&idx=1&sn=3011302613b90749d7ffe0cc3a805d1f)
- [快手结合知识图谱进行多模态内容理解](https://www.infoq.cn/article/2018/09/Multimedia-Understanding-AI)
- [腾讯互娱刘伟：知识图谱让AI更有学识](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651009590&idx=4&sn=e684d70e85b17d4bbb3e48f990014a0d&chksm=bdbeca658ac943737373d9a697bd0296c29b8c20b21638b3b58063c36d846bf4e66635efc79c&scene=27#wechat_redirect)
- [美团大脑：知识图谱的建模方法及其应用](https://tech.meituan.com/2018/11/01/meituan-ai-nlp.html)
- [美团餐饮娱乐知识图谱——美团大脑揭秘](https://tech.meituan.com/2018/11/22/meituan-brain-nlp-01.html)
- [人力资源知识图谱搭建及应用](https://www.jiqizhixin.com/articles/2018-11-23-3)
- [基于概念知识图谱的短文本理解——王仲远](https://blog.csdn.net/TgqDT3gGaMdkHasLZv/article/details/79736982)

### 2019
- [大众点评搜索基于知识图谱的深度学习排序实践](https://www.infoq.cn/article/JZ_qdBDiMc1pHpBMDR2Q)
- [知识图谱已成AI下一风口，但你知道它进展到哪了吗？](https://36kr.com/p/5170293)
- [下一代 AI 系统基石：知识图谱将何去何从？](https://www.infoq.cn/article/DCf3GUp_alTIMuyxYWl3)
- [阿里巴巴电商认知图谱揭秘](https://www.secrss.com/articles/9743)
- [为电商而生的知识图谱，如何感应用户需求？](https://yq.aliyun.com/articles/714353?spm=a2c4e.11163080.searchblog.41.2c1c2ec1qTNAAh)
- [阿里小蜜：知识结构化推动智能客服升级](https://www.infoq.cn/article/ocHiWF5rKuaBDxM5S28x)
- [CCKS 2019:百度CTO王海峰详解知识图谱与语义理解](https://www.jiqizhixin.com/articles/2019-09-12-4)
- [反守为攻！从华为知识图谱窥探AI布局](https://view.inews.qq.com/a/20190528A0SR9E00)






