# Wave-CMNet
1.how we adjust the parameters using Wave-CMNet?
  We have applied Wave-CMNet in real-world cellular networks in China Mobile. Based on the handover parameter adjusting effect, we can support the operation 
  staff to make decision.
2.Dataset Details.
  We create the COP-KPI dataset to evaluate the model performance. The data was collected from 475 gNBs in a major region of China Mobile between May 4,
  2022 and May 26, 2022. 380 gNBs are selected for training and 95 gNBs for testing. Each gNB covers actual data of X, Y , and C, and the sampling interval is 15 minutes. Handoverrelated COP adjustments are performed at least once per gNB.
  C contains 4 kinds of cops related to handover, X contains 23 kinds of observed data, and Y is the estimated KPI. Details are as follows.
  Indicator name|	Indicator Chinese name| Indicator type
  Number of intra frequency handover out attempts between eNodeB|	eNodeB间同频切换出尝试次数| X
  Number of inter frequency outgoing handover attempts between eNodeB| eNodeB间异频切换出尝试次数| X
  Number of inter frequency handover out attempts within eNodeB| eNodeB内异频切换出尝试次数|	X
  Number of intra frequency handover out attempts within eNodeB| eNodeB内同频切换出尝试次数|	X
  Number of intra frequency handover out successful times between eNodeB| eNodeB间同频切换出成功次数|	X
  Number of inter frequency handover out successful times between eNodeB| eNodeB间异频切换出成功次数|	X
  Number of inter frequency handover out successful times within eNodeB| eNodeB内异频切换出成功次数| X
  Number of intra frequency handover out successful times within eNodeB| eNodeB内同频切换出成功次数|	X
  Average number of users in the cell| 小区内的平均用户数|	X
  Average number of available PRB for PUSCH| PUSCH可用PRB平均个数|	X
  Number of available PRBs for uplink| 上行可用的PRB个数|	X
  Average number of available PRBs for downlink| 下行Physical Resource Block被使用的平均个数|	X
  Number of available PRBs for downlink| 下行可用的PRB个数|	X
  Maximum number of active users| 最大激活用户数|	X
  Total throughput of downlink data sent by the PDCP layer of the cell| 小区PDCP层所发送的下行数据的总吞吐量(比特)|	X
  Total throughput of uplink data received by the PDCP layer of the cell| 小区PDCP层所接收到的上行数据的总吞吐量(比特)|	X
  Azimuth| 方位角|	X
  Ant_height| 天线挂高|	X
  cell_machine_cor| 机械下倾角|	X
  cell_ele_bend| 电下倾角|	X
  Total dip| 总下倾角|	X
  NMS acquisition frequency band| 网管采集频段|	X
  frequency point number| 频点号|	X
  average number of effective RRC connections| 有效RRC连接平均数|	X
  maximum number of effective RRC connections| 有效RRC连接最大数|	X
  upstream traffic M| 上行流量M|	X
  downstream traffic M| 下行流量M|	X
  Maximum number of RRC connections| 最大RRC连接用户数|	Y
  INTERFREQHOA1THDRSRP| INTERFREQHOA1THDRSRP|	C
  INTERFREQHOA2THDRSRP| INTERFREQHOA2THDRSRP|	C
  REFERENCESIGNALPWR| REFERENCESIGNALPWR| C
  INTERFREQHOA4THDRSRP| INTERFREQHOA4THDRSRP| C
3.Experiment Setting Details.
  The representation learning module is implied with 4 hidden layers and 2 heads. The outcome prediction module has 3 hidden layers and 2 heads. And LSTM
  has 3 layers. Balanced representation module works with a MLP network. We train DCDN for 10 epochs using the Adam optimizer with a batch size of 64 and a learning rate of 5e-5.
  The dropout rate is set to 0.1. We implement the proposed DCDN model with the PyTorch framework and Python3.
4.Future Work.
  In the future, we would like to do more experiments to prove the effectiveness of each module, for example, ablation study, support downstream
  tasks respectively, etc.In addition, we will explore more optimization schemes to evolve support decision into automatic decision.
