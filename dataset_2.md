# Wave-CMNet
1.Dataset Details.
  Consistent with the first collection, the $2^{nd}$ dataset of COP-KPIs was collected from 475 gNBs over the period from May 4, 2022, to May 26, 2022. Data were captured at 15-minute intervals from each gNB. This dataset comprised 16 observational covariates and 4 handover-related treatment COP metrics, with the number of RRC-connected users serving as the outcome variable. A sliding window approach was utilized for sampling, set with a window size of 192 and a stride of 48, resulting in a total of 5575 samples across the 475 gNBs.
 T contains 4 kinds of cops related to handover, X contains 16 kinds of observed data, and Y is the estimated KPI. Details are as follows.
 
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
  Maximum number of RRC connections| 最大RRC连接用户数|	Y
  INTERFREQHOA1THDRSRP| INTERFREQHOA1THDRSRP|	T
  INTERFREQHOA2THDRSRP| INTERFREQHOA2THDRSRP|	T
  REFERENCESIGNALPWR| REFERENCESIGNALPWR| T
  INTERFREQHOA4THDRSRP| INTERFREQHOA4THDRSRP| T


