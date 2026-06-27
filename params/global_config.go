package params

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
)

var (
	// The following parameters can be set in main.go.
	// default values:
	NodesInShard = 4  // \# of Nodes in a shard.
	ShardNum     = 16 // \# of shards.
)

// consensus layer & output file path
var (
	ConsensusMethod = 0 // ConsensusMethod an Integer, which indicates the choice ID of methods / consensuses. Value range: [0, 4), representing [CLPA_Broker, CLPA, Broker, Relay]"

	PbftViewChangeTimeOut = 10000 // The view change threshold of pbft. If the process of PBFT is too slow, the view change mechanism will be triggered.

	Block_Interval = 5000 // The time interval for generating a new block

	MaxBlockSize_global = 2000  // The maximum number of transactions a block contains
	BlocksizeInBytes    = 20000 // The maximum size (in bytes) of block body
	UseBlocksizeInBytes = 0     // Use blocksizeInBytes as the blocksize measurement if '1'.

	InjectSpeed   = 2000   // The speed of transaction injection
	TotalDataSize = 160000 // The total number of txs to be injected
	TxBatchSize   = 16000  // The supervisor read a batch of txs then send them. The size of a batch is 'TxBatchSize'

	BrokerNum            = 10 // The # of Broker accounts used in Broker / CLPA_Broker.
	RelayWithMerkleProof = 0  // When using a consensus about "Relay", nodes will send Tx Relay with proof if "RelayWithMerkleProof" = 1

	// SPRING mode:
	// 0 = 原始 Hash Relay
	// 1 = SPRING-Heuristic
	// 2 = SPRING-PPO
	// 3 = SPRING-Random baseline（随机基线）
	// 4 = MinState baseline（最少状态优先）
	SpringMode = 2

	// SpringOnlineTrain:
	// 1 = PPO 在线训练：采样动作 + 生成 online_update + 更新模型
	// 0 = PPO 验证模式：不采样 + 不更新模型
	SpringOnlineTrain = 0

	// SpringEvalSample:
	// 1 = 验证模式下也使用采样，但不更新模型
	// 0 = 验证模式下使用最大概率动作
	SpringEvalSample = 0

	// SpringRandomSeed:
	// SpringMode = 3 时使用的随机种子，保证 random baseline（随机基线）可复现。
	SpringRandomSeed int64 = 7

	// Candidate filtering（候选分片过滤）默认对应 16 分片 TopK8 Guard1.3 w45-40 调参组。
	// Capacity guard（容量保护）打开，用于观察负载保护对吞吐、延迟和负载方差的影响。
	SpringCandidateTopK       = 8
	SpringCapacityGuard       = 1
	SpringCapacityGuardFactor = 1.3
	SpringCandidateLoadWeight = 1.0

	// SpringRewardLambda:
	// SPRING reward 中跨片率奖励和负载均衡奖励的权重。
	// 论文默认 λ = 0.5。
	SpringRewardLambda = 0.5

	// SpringRewardBeta:
	// SPRING reward 中负载均衡项 r_wlb = exp(-β * abs_diff) 的衰减系数。
	// 论文默认 β = 0.1。
	SpringRewardBeta = 0.1

	// SpringSenderPosMode:
	// 0 = bidirectional TxBatch relation graph.
	// 1 = paper-like recipient -> senders sender_pos.
	// 2 = reserved for temporal-neighbor enhanced sender_pos.
	SpringSenderPosMode = 1

	// SpringIOTMode:
	// 0 = 普通 SPRING MDP（11k+1 维）
	// 1 = IoT MDP（16 分片时 203 维）：读取 sidecar，追加 current_load 和 10 个轻量场景特征。
	SpringIOTMode = 1

	SpringIOTFeatureDim  = 10
	SpringIOTSidecarFile = "./data_iot/iot_flow_sidecar_multi_anchor_full.csv"
	SpringModelFile      = `E:\project_iot\实验结果7\models\ppo_top8_guard13_w4540_16s_seed7.pt`

	// IoT dense-balanced reward weights. These defaults match
	// spring_lite/offline_env.py so online BlockEmulator feedback and
	// offline PPO training explain the same objective.
	SpringIOTCSTRWeight       = 0.45
	SpringIOTBalanceWeight    = 0.40
	SpringIOTCommCostWeight   = 0.10
	SpringIOTHotspotWeight    = 0.05
	SpringIOTHotspotThreshold = 0.45

	ExpDataRootDir     = `E:\project_iot\实验结果7\block_eval\ppo_top8_guard13_w4540_16s_2M_seed7\expTest` // The root dir where the experimental data should locate.
	DataWrite_path     = ExpDataRootDir + "/result/"                                                        // Measurement data result output path
	LogWrite_path      = ExpDataRootDir + "/log"                                                            // Log output path
	DatabaseWrite_path = ExpDataRootDir + "/database/"                                                      // database write path

	SupervisorAddr = "127.0.0.1:18800"                                  // Supervisor ip address
	DatasetFile    = `./data_iot/selectedTxs_iot_multi_anchor_full.csv` // The raw BlockTransaction data path

	ReconfigTimeGap = 50 // The time gap between epochs. This variable is only used in CLPA / CLPA_Broker now.
)

// network layer
var (
	Delay       int // The delay of network (ms) when sending. 0 if delay < 0
	JitterRange int // The jitter range of delay (ms). Jitter follows a uniform distribution. 0 if JitterRange < 0.
	Bandwidth   int // The bandwidth limit (Bytes). +inf if bandwidth < 0
)

// read from file
type globalConfig struct {
	ConsensusMethod int `json:"ConsensusMethod"`

	SpringMode int `json:"SpringMode"`

	SpringOnlineTrain int `json:"SpringOnlineTrain"`

	SpringRewardLambda float64 `json:"SpringRewardLambda"`
	SpringRewardBeta   float64 `json:"SpringRewardBeta"`

	SpringSenderPosMode *int `json:"SpringSenderPosMode"`

	SpringEvalSample int `json:"SpringEvalSample"`

	SpringRandomSeed int64 `json:"SpringRandomSeed"`

	SpringCandidateTopK       int     `json:"SpringCandidateTopK"`
	SpringCapacityGuard       int     `json:"SpringCapacityGuard"`
	SpringCapacityGuardFactor float64 `json:"SpringCapacityGuardFactor"`
	SpringCandidateLoadWeight float64 `json:"SpringCandidateLoadWeight"`

	SpringIOTMode             int      `json:"SpringIOTMode"`
	SpringIOTFeatureDim       int      `json:"SpringIOTFeatureDim"`
	SpringIOTSidecarFile      string   `json:"SpringIOTSidecarFile"`
	SpringModelFile           string   `json:"SpringModelFile"`
	SpringIOTCSTRWeight       *float64 `json:"SpringIOTCSTRWeight"`
	SpringIOTBalanceWeight    *float64 `json:"SpringIOTBalanceWeight"`
	SpringIOTCommCostWeight   *float64 `json:"SpringIOTCommCostWeight"`
	SpringIOTHotspotWeight    *float64 `json:"SpringIOTHotspotWeight"`
	SpringIOTHotspotThreshold *float64 `json:"SpringIOTHotspotThreshold"`

	PbftViewChangeTimeOut int `json:"PbftViewChangeTimeOut"`

	ExpDataRootDir string `json:"ExpDataRootDir"`

	BlockInterval int `json:"Block_Interval"`

	BlocksizeInBytes    int `json:"BlocksizeInBytes"`
	MaxBlockSizeGlobal  int `json:"BlockSize"`
	UseBlocksizeInBytes int `json:"UseBlocksizeInBytes"`

	InjectSpeed   int `json:"InjectSpeed"`
	TotalDataSize int `json:"TotalDataSize"`

	TxBatchSize          int    `json:"TxBatchSize"`
	BrokerNum            int    `json:"BrokerNum"`
	RelayWithMerkleProof int    `json:"RelayWithMerkleProof"`
	DatasetFile          string `json:"DatasetFile"`
	ReconfigTimeGap      int    `json:"ReconfigTimeGap"`

	Delay       int `json:"Delay"`
	JitterRange int `json:"JitterRange"`
	Bandwidth   int `json:"Bandwidth"`
}

func ReadConfigFile() {
	// read configurations from paramsConfig.json
	data, err := os.ReadFile("paramsConfig.json")
	if err != nil {
		log.Fatalf("Error reading file: %v", err)
	}
	var config globalConfig
	err = json.Unmarshal(data, &config)
	if err != nil {
		log.Fatalf("Error unmarshalling JSON: %v", err)
	}

	// output configurations
	fmt.Printf("Config: %+v\n", config)

	// set configurations to params
	// consensus params
	ConsensusMethod = config.ConsensusMethod

	SpringMode = config.SpringMode

	SpringOnlineTrain = config.SpringOnlineTrain
	if config.SpringRandomSeed != 0 {
		SpringRandomSeed = config.SpringRandomSeed
	}
	SpringCandidateTopK = config.SpringCandidateTopK
	if SpringCandidateTopK < 0 {
		SpringCandidateTopK = 0
	}
	SpringCapacityGuard = config.SpringCapacityGuard
	if config.SpringCapacityGuardFactor > 0 {
		SpringCapacityGuardFactor = config.SpringCapacityGuardFactor
	}
	if SpringCapacityGuardFactor < 1.0 {
		SpringCapacityGuardFactor = 1.0
	}
	if config.SpringCandidateLoadWeight > 0 {
		SpringCandidateLoadWeight = config.SpringCandidateLoadWeight
	}

	SpringRewardLambda = config.SpringRewardLambda
	if SpringRewardLambda < 0 || SpringRewardLambda > 1 {
		SpringRewardLambda = 0.5
	}

	SpringRewardBeta = config.SpringRewardBeta
	if SpringRewardBeta <= 0 {
		SpringRewardBeta = 0.1
	}

	if config.SpringSenderPosMode != nil {
		SpringSenderPosMode = *config.SpringSenderPosMode
	}
	if SpringSenderPosMode < 0 || SpringSenderPosMode > 2 {
		SpringSenderPosMode = 1
	}

	SpringIOTMode = config.SpringIOTMode
	if SpringIOTMode != 1 {
		SpringIOTMode = 0
	}

	if config.SpringIOTFeatureDim > 0 {
		SpringIOTFeatureDim = config.SpringIOTFeatureDim
	}
	if strings.TrimSpace(config.SpringIOTSidecarFile) != "" {
		SpringIOTSidecarFile = config.SpringIOTSidecarFile
	}
	if strings.TrimSpace(config.SpringModelFile) != "" {
		SpringModelFile = config.SpringModelFile
	}
	if config.SpringIOTCSTRWeight != nil {
		SpringIOTCSTRWeight = *config.SpringIOTCSTRWeight
	}
	if config.SpringIOTBalanceWeight != nil {
		SpringIOTBalanceWeight = *config.SpringIOTBalanceWeight
	}
	if config.SpringIOTCommCostWeight != nil {
		SpringIOTCommCostWeight = *config.SpringIOTCommCostWeight
	}
	if config.SpringIOTHotspotWeight != nil {
		SpringIOTHotspotWeight = *config.SpringIOTHotspotWeight
	}
	if config.SpringIOTHotspotThreshold != nil {
		SpringIOTHotspotThreshold = *config.SpringIOTHotspotThreshold
	}
	if SpringIOTCSTRWeight < 0 {
		SpringIOTCSTRWeight = 0
	}
	if SpringIOTBalanceWeight < 0 {
		SpringIOTBalanceWeight = 0
	}
	if SpringIOTCommCostWeight < 0 {
		SpringIOTCommCostWeight = 0
	}
	if SpringIOTHotspotWeight < 0 {
		SpringIOTHotspotWeight = 0
	}
	if SpringIOTHotspotThreshold < 0 {
		SpringIOTHotspotThreshold = 0
	}
	if SpringIOTHotspotThreshold >= 1 {
		SpringIOTHotspotThreshold = 0.999999
	}

	PbftViewChangeTimeOut = config.PbftViewChangeTimeOut

	// data file params
	ExpDataRootDir = config.ExpDataRootDir
	DataWrite_path = ExpDataRootDir + "/result/"
	LogWrite_path = ExpDataRootDir + "/log"
	DatabaseWrite_path = ExpDataRootDir + "/database/"

	Block_Interval = config.BlockInterval

	MaxBlockSize_global = config.MaxBlockSizeGlobal
	BlocksizeInBytes = config.BlocksizeInBytes
	UseBlocksizeInBytes = config.UseBlocksizeInBytes

	InjectSpeed = config.InjectSpeed
	TotalDataSize = config.TotalDataSize
	TxBatchSize = config.TxBatchSize

	SpringEvalSample = config.SpringEvalSample

	BrokerNum = config.BrokerNum
	RelayWithMerkleProof = config.RelayWithMerkleProof
	DatasetFile = config.DatasetFile

	ReconfigTimeGap = config.ReconfigTimeGap

	// network params
	Delay = config.Delay
	JitterRange = config.JitterRange
	Bandwidth = config.Bandwidth
}
