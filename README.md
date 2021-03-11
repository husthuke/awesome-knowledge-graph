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
### 整体概念架构
随着知识图谱的发展，与之相关的概念也越来越多，在阅读论文时先准确的把握该论文所要解决问题处于的层级或者位置对于更好的理解论文也比较有帮助，在此对知识图谱的概念进行了总结整理，整体概念架构图如下图所示，后面的论文分类也按照该整体架构概念图从顶向下，从整体到细节的方式组织。

![image](https://github.com/husthuke/awesome-knowledge-graph/blob/master/image/framework.png)

### 综述综合
#### 大综述


- [Knowledge Graph Construction Techniques](./paper/知识图谱构建技术综述_刘峤.caj)
- [Review on Knowledge Graph Techniques (2016)](./paper/知识图谱技术综述.pdf)[一篇经典的中文综述，适合入门。]
- [Reviews on Knowledge Graph Research (2017)](./paper/知识图谱研究综述-李涓子.pdf)[清华大学李娟子老师的综述，十分经典，对知识图谱走入大众视野功不可没。]
- [The Research Advances of Knowledge Graph (2017)](./paper/知识图谱研究进展_漆桂林.caj)[东南大学漆桂林老师的综述，也是2017年发表的，同样对知识图谱走入大众视野起到很大作用。]
- [A Survey on Knowledge Graphs: Representation, Acquisition and Applications (2020)](https://arxiv.org/pdf/2002.00388.pdf)
- [Knowledge Graphs (2020)](https://arxiv.org/pdf/2003.02320.pdf)[2020年初的一篇作者众多、内容很全的综述，适合系统性的建立知识图谱的知识体系。]


#### Knowledge-Augmented LMs(知识增强语言模型) 
知识图谱增强语言模型是最近两年比较流行，主要发生在BERT出来之后，将知识先验信息融入到语言模型，可以说是知识图谱助力NLP十分关键的一环，将该专题放在比较靠前的位置。


- [ERNIE: Enhanced Representation through Knowledge Integration(2019)](https://arxiv.org/abs/1904.09223)[百度版本ERNIE，在预训练阶段Mask Token时引入了Entity级别和Phase级别，似的模型在学习时能够将某些特定知识作为一个整体进行学习。]
- [ERNIE: Enhanced Language Representation with Informative Entities(2019)](https://arxiv.org/abs/1905.07129)[清华版本ERNIE，将从句子中识别出的Entity的Embedding与原句子Embedding同时K-Encoder新设计的模块，在该模块中也采用多头注意力机制之后融合编码在分别输出到下一层。]
- [Latent Relation Language Models(2019)](https://arxiv.org/pdf/1908.07690.pdf)[将文本中实体在知识图谱中的结构作为条件建模到概率语言模型中。]
- [K-BERT: Enabling Language Representation with Knowledge Graph](https://arxiv.org/pdf/1909.07606.pdf)
- [KG-BERT: BERT for Knowledge Graph Completion(2019)](https://arxiv.org/abs/1909.03193)[与ERNIE系列处理的问题正好相反，是将Bert的模型应用到知识图谱的补全任务中，根据h,r->t,h,r,t->{0,1}的任务特点设计出两个fine-tuning任务。]
- [Enriching BERT with Knowledge Graph Embeddings for Document Classification(2019)](https://arxiv.org/abs/1909.08402)[结合Bert和知识图谱embedding应用到具体的文档分类任务，将Bert输出、人工设计的Meta特征、作者的kg embedding进行concat之后输入mlp进行分类。]
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
- [Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model (ICLR 2020)](https://arxiv.org/abs/1912.09637)[在预训练任务中结合wikepedia知识将句子中的实体替换为同类型的其他实体，使预训练模型能够在很少的训练数据甚至是无训练数据的情况下在某些特定QA任务取得不错的效果。]
- [Language Models as Knowledge Bases?](https://arxiv.org/pdf/1909.01066.pdf)[设计出一种基于完形填空任务的探测结构LAMA验证了类BERT预研模型具备一定的知识库能力]
- [A Frame-based Sentence Representation for Machine Reading Comprehension (ACL 2020)](https://www.aclweb.org/anthology/2020.acl-main.83.pdf)[将句子中包含的FrameNet信息自动标注出来之后，平铺展开填充形成quadruples，再将quadruples采用不同的Aggregation Model表示为句子表示，然后采用BERT等神经网络进行编码进行后续的阅读理解任务。]


#### 常识图谱(Commonsense)
目前人工智能在很多方面表现的比较智障的原因，很多学者仍为是由于AI缺乏基本常识知识的原因，因此，从感知智能到认知智能常识知识起着很重要的作用，而常识图谱作为常识知识的一个重要表示手段也越来越受到重视。


- [KILT: a Benchmark for Knowledge Intensive Language Tasks(2020)](https://arxiv.org/pdf/2009.02252v1.pdf)[facebook 针对知识集中型的语言任务设计的baenchmark，其中包括设计常识的任务。]
- [TransOMCS: From Linguistic Graphs to Commonsense Knowledge(ICJAI 2020)](https://arxiv.org/pdf/2005.00206.pdf)[基于语言图谱构建常识知识，采用ConceptNet中子集作为常识种子，从语言图谱中挖掘满足种子(h,r,t)的模式，然后再基于这些模式从语言图谱中找常识，同时为了避免错误模式带来的影响，引入一个打分机制筛选模式，并且对找出来的常识进行可信度排序。]
- [ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning (AAAI 2019)](https://arxiv.org/abs/1811.00146)[将常识信息抽象成一系列if then表示的框架，并基于词框架设计众包任务获得数据集，通过encoder(ELMo)+decoder结构的model进行训练，最后采用BLEU score来评价机器在指定指令下的推断输出。]
- [COMET: Commonsense Transformers for Automatic Knowledge Graph Construction (ACL 2019)](https://arxiv.org/abs/1906.05317)[结合预训练模型提出了一种常识知识图谱构建框架，并在ATOMIC和ConceptNet常识知识图谱上进行实验，同时也验证了模型参数采用预训练之后的参数比随机初始化效果明显要好。]





### 知识应用
#### 对话系统
知识图谱落地应用最为广泛的一个方向，研究人数也众多，个人觉得在工业界可发挥的空间比较大。

- [Commonsense Knowledge Aware Conversation Generation with Graph Attention (IJCAI 2018)](https://www.ijcai.org/Proceedings/2018/0643.pdf)[先将原始文本中的实体转化为常识知识图谱中的子图片段，再将子图片段采用图注意力方式embedding之后的向量同时输入到encoder和decoder中进行融合。]
- [Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End
Task-Oriented Dialog Systems(2018)](https://arxiv.org/pdf/1804.08217.pdf)[提出一种Men2Seq的改进结构来配合copy 机制更好的处理任务型对话系统。]
- [Augmenting End-to-End Dialogue Systems with Commonsense Knowledge(2018)](https://arxiv.org/pdf/1709.05453.pdf)[将问句x中的实体所对应的常识概念以及之间的关系作为额外信息a，将x与y，a与y分别使用LSTM打分之后的总和作为回答y的分值。]

#### 知识库问答-KBQA
- [Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/pdf/1909.05855.pdf)
- [Improving Knowledge-aware Dialogue Generation via
Knowledge Base Question Answering](https://arxiv.org/pdf/1912.07491.pdf)
- [Graph-Based Reasoning over Heterogeneous External Knowledge for Commonsense Question Answering](https://arxiv.org/pdf/1909.05311.pdf)

### 推荐系统
- [Multi-modal Knowledge Graphs for Recommender Systems(CIKM 2020)](https://zheng-kai.com/paper/cikm_2020_sun.pdf)[将多模态信息引入知识图谱推荐，整体结构上包括多模态知识图谱graph embedding和recommendation两部分，其中graph embedding包括entity encoder和attention，用于综合表示节点以及其周围节点信息。]

### 知识计算
#### Representation（知识表示）
知识应用的基础，目前分布式表示或者embedding大有一统江湖的意思，各种花式embedding眼花缭乱。

- [Knowledge Representation Learning: A Review](./paper/知识表示学习研究进展_刘知远.caj)
- [Interstellar: Searching Recurrent Architecture for Knowledge Graph Embedding(NeurIPS 2020)](https://arxiv.org/pdf/1911.07132.pdf)[基于由三元组组成的关系路径（relational path）提出 Interstellar 模型，通过搜索一种递归神经网络，来处理关系路径中的短链、长链信息,达到根据不同任务，有针对性地对关系路径进行建模的目的。]
- [Holographic embeddings of knowledge graphs](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828)
- [Context-dependent knowledge graph embedding. EMNLP 2015. Luo, Yuanfei and Wang, Quan and Wang, Bin and Guo, Li.](http://www.aclweb.org/anthology/D15-1191)
- [GAKE: graph aware knowledge embedding. COLING 2016. Feng, Jun and Huang, Minlie and Yang, Yang and Zhu, Xiaoyan.](http://www.aclweb.org/anthology/C16-1062)
- [KBGAN: Adversarial Learning for Knowledge Graph Embeddings. Cai, Liwei, and William Yang Wang.(NAACL 2018)](https://arxiv.org/pdf/1711.04071.pdf)
- [Bootstrapping Entity Alignment with Knowledge Graph Embedding. Zequn Sun, Wei Hu, Qingheng Zhang and Yuzhong Qu.(IJCAI 2018)](https://www.ijcai.org/proceedings/2018/0611.pdf)
- [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space(ICLR 2019)](https://arxiv.org/pdf/1902.10197.pdf)[在Tran系列的embedding上进了一步，将三元组(h,r,t)中的关系r建模成复平面的旋转，使得r更具表示力，能够很好的表示Symmetry、Antisymmetry以及Inversion关系，使用了自对抗负采样损失方法。]
- [Quaternion Knowledge Graph Embeddings(2019)](https://arxiv.org/pdf/1904.10281.pdf)[相比RotatE更进了一步，通过4元组的Hamilton Product来表示关系变换，比复数具备了更多的自由度。]
- [Knowledge Graph Embeddings and Explainable AI(2020)](https://arxiv.org/pdf/2004.14843.pdf)[一篇综述性质文章，介绍了目前sota的KGE方法，并分析embedding与可解释性的关系和联系。]




#### Reasoning(知识推理)
听起来高大上的方向，实际落地感觉很不容易，学术界发paper可能还行，但是在工业界容易跪，要推理也尽量离线展开，不要在线推理。


- [Reasoning on Knowledge Graphs with Debate Dynamics](https://arxiv.org/pdf/2001.00461.pdf)
- [Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge(2016)](https://arxiv.org/pdf/1606.04422.pdf)[提出一种通过tenor网络结构设计实现真值逻辑的框架，尝试通过深度学习解决逻辑推理问题，比较有创新性。]
- [Differentiable Learning of Logical Rules for Knowledge Base Reasoning.(2017)](https://arxiv.org/pdf/1702.08367.pdf)[]
- [Query2box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings(ICLR 2020)](https://openreview.net/pdf?id=BJgr4kSFDS)[将问题映射为向量空间中的box，答案实体为向量空间中的向量，回答问题的过程可以建模成Projection、 Intersection和Union，最终获得问题的box，通过定义好的query和entity距离计算方法计算距离，小于阈值则为答案集合。]
- [Conversational Neuro-Symbolic Commonsense Reasoning(2020)]()
- [Neural-Symbolic Reasoning on Knowledge Graphs(2020)](https://arxiv.org/pdf/2010.05446.pdf)[系统的整理了知识图谱推理中的Symbolic reasoning以及Neural reasoning的方法，并介绍了其不同的优劣势，对KGC和KBQA问题在统一推力网络中进行了总结。]

#### KG Completion(图谱补全)
- [Learning Sequence Encoders for Temporal Knowledge Graph Completion](./paper/Learning_Sequence_Encoders_for_Temporal_Knowledge_Graph_Completion.pdf)
- [Differentiable Reasoning on Large Knowledge Bases and Natural Language](https://arxiv.org/pdf/1912.10824.pdf)
- [Diachronic Embedding for Temporal Knowledge Graph Completion](https://arxiv.org/pdf/1907.03143.pdf)
- [Commonsense Knowledge Base Completion with Structural and Semantic Context](https://arxiv.org/pdf/1910.02915.pdf)
- [KG-BERT: BERT for Knowledge Graph Completion](https://arxiv.org/pdf/1909.03193.pdf)


#### Coreference Resolution(指代消解)
- [Intra-document Coreference Resolution: The state of the art (2007)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.362.5521&rep=rep1&type=pdf)[指代消解，较全面的介绍了指代消解的发展、分类方法和评测标准，基本都是偏传统的方法。]
- [指代消解综述 (2010)](http://www.ecice06.com/CN/10.3969/j.issn.1000-3428.2010.08.012)[一篇入门级的综述，介绍了如何标注以及基于句法、基于规则、基于统计、基于分类等相对较为传统指代消解的方法。]
- [Cross-Document Co-Reference Resolution using Sample-Based Clustering
with Knowledge Enrichment (EMNLP 2015)](http://www.emnlp2015.org/proceedings/EMNLP-TACL/pdf/EMNLP-TACL5.pdf)[Pipeline方式，首先采用通用工具计算出文档内的mention组，再采用谱聚类和图相关算法优化相关度计算并完成跨文档的mention指代消解。]
- [Higher-order Coreference Resolution with Coarse-to-fine Inference (ACL2018)](https://www.aclweb.org/anthology/N18-2108.pdf)
- [BERT for Coreference Resolution: Baselines and Analysis (2019)](https://arxiv.org/pdf/1908.09091.pdf)[将上一篇paper中lstm等encode部分更换为bert的优化方法，套路与其他任务换成bert类似。]
- [Deep Reinforcement Learning for Mention-Ranking Coreference Models (ACL2016)](https://www.aclweb.org/anthology/D16-1245.pdf)





### 知识获取
#### NER(命名实体识别)
也是自然语言处理的基础任务，十分重要。

#### Entity aligning(实体对齐)

- [A Survey on Entity Alignment of Knowledge Base](./paper/知识库实体对齐技术综述.pdf)
- [Knowledge Graph Alignment Network with Gated Multi-hop Neighborhood Aggregation](https://arxiv.org/pdf/1911.08936.pdf)
- [Coordinated Reasoning for Cross-Lingual Knowledge Graph Alignment](https://arxiv.org/pdf/2001.08728.pdf)
- [End-to-End Neural Entity Linking (2018)](https://arxiv.org/abs/1808.07699) [将mention的字、词向量拼接之后过bilstm得到mention embedding，将候选mention的头尾词向量和基于attention的soft head拼接得到候选mention embedding，两个embedding一起通过FFNN打分，从而实现端到端的实体链接。]
- [Zero-shot Entity Linking with Efficient Long Range Sequence Modeling (2020)](https://arxiv.org/pdf/2010.06065.pdf)

#### Coreference Resolution(指代消解)
- [Using Type Information to Improve Entity Coreference Resolution](https://arxiv.org/pdf/2010.05738.pdf)



### 知识建模

#### Taxonomy(本体构建)
- [AutoKnow: Self-Driving Knowledge Collection for Products of Thousands of Types(KDD 2020)](https://dl.acm.org/doi/pdf/10.1145/3394486.3403323)[结合商品领域特点将传统SPO三元组结构的建模方式进行了修改，并提出一种自动进行实体收集的架构]
- [A Short Survey on Taxonomy Learning from Text Corpora: Issues, Resources and Recent Advances (ACL2017)](https://www.aclweb.org/anthology/D17-1123)[以两段式流程框架介绍了基于语料构建本体的各种方法，两段式流程框架是指is-a关系的抽取和基于is-a关系的本体构建。]


### 其他扩展
#### Tracing(知识追踪)
本类别并不是传统知识图谱中的任务，而是与教育领域结合的广义上的知识图谱任务。

- [Context-Aware Attentive Knowledge Tracing(KDD 2020)](https://dl.acm.org/doi/pdf/10.1145/3394486.3403282)[]
- [Knowledge Query Network for Knowledge Tracing()](https://arxiv.org/pdf/1908.02146.pdf)[将学生外在表现建模成knowledge vetor和skill vetor的点积]
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

- [百度知识中台白皮书(2020版)](./report/百度知识中台白皮书(2020版).pdf)
- [面向人工智能新基建的知识图谱行业白皮书(艾瑞咨询2020版)](./report/面向人工智能新基建的知识图谱行业白皮书(艾瑞咨询2020版).pdf)
- [知识图谱标准化白皮书(2019版)](./report/知识图谱标准化白皮书(2019版).pdf)
- [CCKS2018-知识图谱发展报告](./report/CCKS2018-知识图谱发展报告.pdf)




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
- [CCKS 2019 面向金融领域的事件主体抽取](https://www.biendata.com/competition/ccks_2019_4/)
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
- [AICon 2019知识图谱专题](https://aicon.infoq.cn/2019/beijing/track/658)
- [AICon 2018知识图谱深度培训](https://aicon2018.geekbang.org/training/1315)
- [AICon 2017知识图谱技术实践](https://aicon2017.geekbang.org/training/1)

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

- [小型金融知识图谱构流程示范](https://github.com/jm199504/Financial-Knowledge-Graphs)

- [利用网络上公开的数据构建一个小型的证券知识图谱知识库](https://github.com/lemonhu/stock-knowledge-graph)

- [上市公司高管图谱](https://github.com/Shuang0420/knowledge_graph_demo)

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

### 2020

- [基于知识图谱的语义理解技术及应用](https://www.infoq.cn/article/jp2QIvGFAp1vP6EHxcs5)
- [阿里巴巴资深算法专家张伟：百科类知识图谱构建和应用已到深水区，行业和多模态知识图谱引关注](https://mp.weixin.qq.com/s?__biz=MzU1NDA4NjU2MA==&mid=2247501445&idx=2&sn=93350c260914201f791531832e33929a&chksm=fbea7f4acc9df65cbdf983597e31bc03240820a2365d5ce9db5edcc39a05cfababf234c3b5b7&scene=27#wechat_redirect)
- [华为开发者大会 HDC.Cloud 硬核技术解读：知识图谱构建流程及方法](https://www.infoq.cn/article/POTuVeAk0Lh0VmW9T9a0)
- [腾讯云知识图谱技术与应用实践之路](https://www.infoq.cn/article/kYjJqkao020DcHDMJINI)
- [知识图谱在小米的落地与挑战](https://www.infoq.cn/article/EWZDX9au7s6h7339DxiJ)
- [构建行业知识图谱切勿操之过急，这里有三大前提条件](https://www.infoq.cn/article/PF9fjAKPf8MVGNK6aSpw)


### 2019
- [大众点评搜索基于知识图谱的深度学习排序实践](https://www.infoq.cn/article/JZ_qdBDiMc1pHpBMDR2Q)
- [知识图谱已成AI下一风口，但你知道它进展到哪了吗？](https://36kr.com/p/5170293)
- [下一代 AI 系统基石：知识图谱将何去何从？](https://www.infoq.cn/article/DCf3GUp_alTIMuyxYWl3)
- [阿里巴巴电商认知图谱揭秘](https://www.secrss.com/articles/9743)
- [为电商而生的知识图谱，如何感应用户需求？](https://yq.aliyun.com/articles/714353?spm=a2c4e.11163080.searchblog.41.2c1c2ec1qTNAAh)
- [阿里小蜜：知识结构化推动智能客服升级](https://www.infoq.cn/article/ocHiWF5rKuaBDxM5S28x)
- [CCKS 2019:百度CTO王海峰详解知识图谱与语义理解](https://www.jiqizhixin.com/articles/2019-09-12-4)
- [反守为攻！从华为知识图谱窥探AI布局](https://view.inews.qq.com/a/20190528A0SR9E00)


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

### 2017
- [阿里知识图谱首次曝光：每天千万级拦截量，亿级别全量智能审核](https://mp.weixin.qq.com/s/MZE_SXsNg6Yt4dz2fmB1sA)
- [百度王海峰：知识图谱是 AI 的基石](https://www.infoq.cn/article/2017/11/Knowledge-map-cornerstone-AI)
- [哈工大刘挺：从知识图谱到事理图谱](https://mp.weixin.qq.com/s/1nl56AdZIkT03gnmimt8nQ)
- [智能导购？你只看到了阿里知识图谱冰山一角](https://www.csdn.net/article/a/2017-12-08/15937080)

### 2016
- [构建 LinkedIn 知识图谱
](https://www.infoq.cn/article/constructing-linkedin-knowledge-map)






