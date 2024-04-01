# Wave-CMNet
1.Dataset Details.
  The $1^{st}$ COP-KPI dataset was collected from the China Mobile network, encompassing 518 gNBs, over the period from July 6, 2023, to July 20, 2023. Each gNB systematically gathered real-time data for covariates, outcomes, and treatment at 15-minute intervals. Notably, each gNB recorded at least once COP adjustment within the data acquisition timeframe. Base station power was chosen as the treatment and 24 types of observed data as the covariates. The outcome variables included Downlink User Throughput, Uplink Traffic (MB), Downlink Traffic (MB), Average Downlink CQI, and Interference Level. To enhance the dataset, a sliding window technique was employed for data sampling, with the window size set to 672 and a stride of 48. In total, 7617 data points were collected from 518 gNBs.
  T contains 1 kinds of cops related to handover, X contains 24 kinds of observed data, and Y is the estimated KPI. Details are as follows.
  
  Indicator name| Indicator Chinese name| Indicator type
Reference Signal Power| 参考信号功率| T
Intra-frequency Handover Out Success Count| 同频切换出成功次数| X
Inter-frequency Handover Out Success Count| 异频切换出成功次数| X
Inter-eNB Handover Out Success Count| eNB间切换出成功次数| X
Inter-eNB Handover Out Request Count| eNB间切换出请求次数| X
Intra-eNB Handover Out Success Count| eNB内切换出成功次数| X
Intra-eNB Handover Out Request Count| eNB内切换出请求次数| X
Inter-eNB X2 Handover Out Request Count| eNB间X2切换出请求次数| X
Inter-eNB X2 Handover Out Success Count| eNB间X2切换出成功次数| X
Total Handover Out Count| 切换出总次数| X
Handover Request Count (QCI=1)| 切换请求次数(QCI=1)| X
Handover Success Count (QCI=1)| 切换成功次数(QCI=1)| X
Average RRC Connection Count| RRC连接平均数| X
Maximum RRC Connection Count| RRC连接最大数| X
PDCCH Channel CCE Utilization Rate| PDCCH信道CCE占用率| X
E-RAB Establishment Success Count| E-RAB建立成功数| X
E-RAB Establishment Request Count| E-RAB建立请求数| X
E-RAB Establishment Success Count (QCI=1)| E-RAB建立成功数(QCI=1)| X
E-RAB Establishment Request Count (QCI=1)| E-RAB建立请求数(QCI=1)| X
RRC Connection Establishment Success Count| RRC连接建立成功次数| X
RRC Connection Re-establishment Request Count| RRC连接重建请求次数| X
Initial Context Establishment Success Count| 初始上下文建立成功次数| X
eNB-initiated E-RAB Release Count| eNB请求释放的E-RAB数| X
Residual E-RAB Count| 遗留E-RAB个数| X
eNB-initiated Context Release Count| eNB请求释放上下文数| X
Downlink User Throughput| 下行用户速率| Y
Uplink Traffic Volume (MB)| 上行流量(MB)| Y
Downlink Traffic Volume (MB)| 下行流量(MB)| Y
Average Downlink CQI| 下行平均CQI| Y
Interference Level| 干扰电平| Y
