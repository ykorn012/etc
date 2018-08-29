Big Data Capabilities Applied to Semiconductor Manufacturing Advanced Process Control

디지털 우주는 매 2 년마다 두 배씩 증가하고 있으며 2020 년까지 40,000 엑서 바이트 (40 조 기가 바이트)에 이를 것으로 예상됩니다 [1]. 데이터 생산량, 속도, 품질, 병합 및 분석에 대한 요구 사항이 반도체 제조에서 급격히 증가함에 따라 우리는 팹 전반에 걸쳐 데이터 관리 및 사용에 대한 새로운 접근법에 대한 필요성에 직면 해 있습니다. 이 문제는 여러 산업 분야에서 발생하고 있으며 이에 대응하여 "큰 데이터"노력이 대두되었습니다. 우리 업계에서는 대규모 데이터 솔루션이 Fault Detection and Classification (FDC)과 Run-to-Run (R2R) 제어 솔루션을 포함하는 고급 프로세스 제어 (APC) 솔루션을 미세 조정하여 제어 및 진단 수준을 높이는 데 핵심 역할을 합니다. 그러나 주된 영향은 Predictive Maintenance (PdM), 예측 스케줄링, Virtual Metrology (VM) 및 수율 예측과 같은 보다 효과적인 예측 기술을 더 잘 활용하는 것입니다 [2], [3].

SECTION I. Introduction

반도체에 대한 국제 기술 로드맵 (ITRS)은 큰 데이터 문제의 크기를 볼륨, 속도, 다양성 (즉, 데이터 병합), Veracity (데이터 품질) 및 Value (즉, , 분석) [3]. 대형 데이터 솔루션으로 전환하는 것은 다양한 레벨에서 5 개의 V를 처리하는 것입니다. 오늘날에는 대용량 데이터 또는 향상된 데이터 품질을 지원하기 위해 기존 시스템을 개선하여 때로는 이를 수행합니다. 그러나 장기적으로는 모든 제조가 다음과 같은 구성 요소를 포함하는 더 큰 데이터 친화적 솔루션으로 이동하게 될 것으로 예상됩니다. [5] :

- 실시간 수집 및 분석 : 증가 된 데이터 수집 속도 (Velocity) 외에도 시간 결정적 환경에서 실시간 의사 결정을 지원하는 분석 기능이 구현되고 있습니다.
- Apache Hadoop : Hadoop은 상용 하드웨어 클러스터에서 데이터 세트의 저장 및 대규모 처리를위한 오픈 소스 소프트웨어 프레임 워크를 제공합니다 [4]. Hadoop은 병렬 처리 및 확장성 기능을 활용하여 추적 데이터와 같은 대규모 시계열 데이터 세트에 적합한 솔루션을 제공합니다.
- Hadoop 분산 파일 시스템 (Hadoop Distributed Filing System, HDFS) : 대용량 데이터 볼륨 처리 및 가용성 극대화를 위해 특별히 설계되고 최적화 된 분산 파일 시스템입니다.
- MapReduce- 유형 프레임 워크 : 대규모 데이터 처리를 위해 프로그래밍 모델이 필요합니다. 예를 들어 MapReduce (처음에는 독점 Google 기술을 언급했지만 이후 일반화되었습니다)입니다.
- 데이터웨어 하우징 : 데이터 볼륨의 제한을 완화하는 확장 가능한 저장 기능이 필요합니다.
- 분석 : 대용량 데이터의 "가치"구성 요소를 처리함으로써 (특히 예측적인) 분석을 통해 대용량 데이터를 신속하게 분석 할 수 있습니다.
이 논문에서는 APC 시스템을 위한 대형 데이터 솔루션으로의 전환을 설명하고 이점을 설명합니다. 특히 II 장에서는 APC 용 빅 데이터 솔루션으로 이동할 때 고려해야 할 사항을 논의하고 빅 데이터 솔루션 구성의 성능을 기존 관계형 시스템과 비교하는 평가가 제공됩니다. 섹션 III에서는 기존 APC 기능이 큰 데이터 개선을 통해 활용할 수 있는 몇 가지 이점을 간략하게 설명하고 예측 솔루션의 기능을 개선하고 심지어 활성화하는 데 큰 역할을 하는 역할에 대해서는 IV 절에서 설명합니다. 또한 이 논문에서는 V 섹션의 팹에 대한 대용량 데이터 솔루션의 장기 마이그레이션 경로에 대해 설명하고 큰 데이터 진화의 일부가 될 것으로 예상되는 향후 방향 및 기회를 살펴 봅니다.

SECTION II. Implementing an APC Big Data Solution

A. Design Considerations: Hadoop Ecosystem Customized to Semiconductor Manufacturing Requirements
반도체 제조 데이터는 다양한 수준의 구조를 가지며 다중 서명으로 특징 지어 질 수 있습니다. 전통적인 관계형 기술에 비해 저장 효율성과 쿼리 성능을 향상시키기 위해 쓰기 빈도와 요구되는 읽기 횟수에 따라 다양한 구조가 사용됩니다. 예를 들어, 도 1에 도시 된 솔루션 접근법에서, 1 회 기입, 판독 - 다수의 요구를 갖는 데이터 카테고리에 대해 원주 형 저장 접근법이 적용된다. write-many, read-many로 특징 지워지는 데이터는 압축, 메모리 내 작업 및 확률적으로 대량의 sparse 데이터를 저장하는 내결함성있는 방법을 제공하는 오픈 소스 비 관계형 분산 데이터베이스 인 HBase에 저장됩니다 필터는 컬럼 단위로 적용된다 [5].

Fig. 1. Illustration of different big data storage mechanisms used to address data that has the requirement of write-once read-many (top), and write-many read-many (bottom).

대부분의 응용 프로그램은 두 가지 범주의 상점에서 효율적이고 투명하며 병렬로 데이터를 검색해야 합니다. 분산 쿼리 엔진 (임팔라)과 원시 Hadoop 데이터 처리 기술은 그림 2와 같이 이 용도로 사용됩니다. 

Fig. 2. Distributed query engine to support real-time and batch queries; HDFS—Hadoop Distributed File System, HQL—HBase Query Language [5].
