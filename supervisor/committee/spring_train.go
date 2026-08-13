package committee

import (
	"blockEmulator/params"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"
)

type SpringFeedbackRewardRecord struct {
	TimeUnixNano int64 `json:"time_unix_nano"`

	Epoch int `json:"epoch"`

	TotalTx     int     `json:"total_tx"`
	TotalInner  int     `json:"total_inner"`
	TotalRelay1 int     `json:"total_relay1"`
	TotalRelay2 int     `json:"total_relay2"`
	EffectiveTx float64 `json:"effective_tx"`
	CrossTx     float64 `json:"cross_tx"`

	Loads          []int     `json:"loads"`
	EffectiveLoads []float64 `json:"effective_loads"`
	Inners         []int     `json:"inners"`
	Relay1         []int     `json:"relay1"`
	Relay2         []int     `json:"relay2"`

	DecisionBatchIDs []uint64 `json:"decision_batch_ids"`

	CrossRate              float64 `json:"cross_rate"`
	RawLoadVariance        float64 `json:"raw_load_variance"`
	NormalizedLoadVariance float64 `json:"normalized_load_variance"`

	// SPRING-style reward 分解项
	RCSTR       float64 `json:"r_cstr"`
	RWLB        float64 `json:"r_wlb"`
	AbsLoadDiff float64 `json:"abs_load_diff"`

	CommunicationCost      float64 `json:"communication_cost"`
	CommunicationCostSum   float64 `json:"communication_cost_sum"`
	CommunicationCostCount int     `json:"communication_cost_count"`
	MaxLoadShare           float64 `json:"max_load_share"`
	HotspotPenalty         float64 `json:"hotspot_penalty"`

	Reward     float64 `json:"reward"`
	RewardMode string  `json:"reward_mode"`

	Lambda float64 `json:"lambda"`
	Beta   float64 `json:"beta"`

	IOTCSTRWeight       float64 `json:"iot_cstr_weight"`
	IOTBalanceWeight    float64 `json:"iot_balance_weight"`
	IOTCommCostWeight   float64 `json:"iot_comm_cost_weight"`
	IOTHotspotWeight    float64 `json:"iot_hotspot_weight"`
	IOTHotspotThreshold float64 `json:"iot_hotspot_threshold"`

	RunningAvgCrossRate float64 `json:"running_avg_cross_rate"`
	RunningAvgReward    float64 `json:"running_avg_reward"`
	RunningAvgNormVar   float64 `json:"running_avg_norm_var"`
	RewardedEpochCount  int     `json:"rewarded_epoch_count"`
}

type SpringTrainAction struct {
	BatchID    uint64    `json:"batch_id"`
	Address    string    `json:"address"`
	Related    string    `json:"related"`
	State      []float64 `json:"state"`
	Action     int       `json:"action"`
	LogProb    float64   `json:"log_prob"`
	Value      float64   `json:"value"`
	Entropy    float64   `json:"entropy"`
	Confidence float64   `json:"confidence"`
	NextState  []float64 `json:"next_state"`

	Reward      float64 `json:"reward"`
	Done        bool    `json:"done"`
	LocalReward float64 `json:"local_reward"`

	RelatedKnown          bool      `json:"related_known"`
	RelatedShard          int       `json:"related_shard"`
	RelatedWeight         float64   `json:"related_weight"`
	RelatedCount          int       `json:"related_count"`
	SenderPos             []float64 `json:"sender_pos"`
	ActionMask            []int     `json:"action_mask,omitempty"`
	ChosenShard           int       `json:"chosen_shard"`
	SameAsRelated         bool      `json:"same_as_related"`
	RelatedInCurrentBatch bool      `json:"related_in_current_batch"`
	ShardLoadBefore       int       `json:"shard_load_before"`
	ShardLoadMeanBefore   float64   `json:"shard_load_mean_before"`
	LoadPenalty           float64   `json:"load_penalty"`
}

type SpringFeedbackAggregate struct {
	Count      int     `json:"count"`
	FirstEpoch int     `json:"first_epoch"`
	LastEpoch  int     `json:"last_epoch"`
	WeightSum  float64 `json:"weight_sum"`

	RewardSum                 float64 `json:"reward_sum"`
	CrossRateSum              float64 `json:"cross_rate_sum"`
	EffectiveTxSum            float64 `json:"effective_tx_sum"`
	CrossTxSum                float64 `json:"cross_tx_sum"`
	NormalizedLoadVarianceSum float64 `json:"normalized_load_variance_sum"`
	RCSTRSum                  float64 `json:"r_cstr_sum"`
	RWLBSum                   float64 `json:"r_wlb_sum"`
	AbsLoadDiffSum            float64 `json:"abs_load_diff_sum"`
	CommunicationCostSum      float64 `json:"communication_cost_sum"`
	MaxLoadShareSum           float64 `json:"max_load_share_sum"`
	HotspotPenaltySum         float64 `json:"hotspot_penalty_sum"`
	LambdaSum                 float64 `json:"lambda_sum"`
	BetaSum                   float64 `json:"beta_sum"`
	TotalTxSum                float64 `json:"total_tx_sum"`
	TotalInnerSum             float64 `json:"total_inner_sum"`
	TotalRelay1Sum            float64 `json:"total_relay1_sum"`
	TotalRelay2Sum            float64 `json:"total_relay2_sum"`
}

type SpringTrainBatch struct {
	// 这里的 BatchID 现在表示真实 TxBatch 编号，不是单个 PPO 推理请求编号。
	BatchID uint64 `json:"batch_id"`

	TxStartNonce uint64 `json:"tx_start_nonce"`
	TxEndNonce   uint64 `json:"tx_end_nonce"`
	TxCount      int    `json:"tx_count"`

	EnqueueUnixNano int64 `json:"enqueue_unix_nano"`
	IsFinalBatch    bool  `json:"is_final_batch"`

	FeedbackMatches    int                     `json:"feedback_matches"`
	LastFeedbackEpoch  int                     `json:"last_feedback_epoch"`
	FirstFeedbackEpoch int                     `json:"first_feedback_epoch"`
	FeedbackAggregate  SpringFeedbackAggregate `json:"feedback_aggregate"`

	Actions []SpringTrainAction `json:"actions"`
}

type SpringOnlineUpdateInput struct {
	TimeUnixNano int64 `json:"time_unix_nano"`

	// 这里的 BatchID 表示真实 TxBatch 编号。
	BatchID uint64 `json:"batch_id"`

	MatchedBatchIDs []uint64 `json:"matched_batch_ids"`

	TxStartNonce uint64 `json:"tx_start_nonce"`
	TxEndNonce   uint64 `json:"tx_end_nonce"`
	TxCount      int    `json:"tx_count"`

	FeedbackEpoch int `json:"feedback_epoch"`
	Shards        int `json:"shards"`
	IOTFeatureDim int `json:"iot_feature_dim,omitempty"`

	Actions    []SpringTrainAction `json:"actions"`
	Reward     float64             `json:"reward"`
	NextStates [][]float64         `json:"next_states"`
	Done       bool                `json:"done"`

	CrossRate              float64 `json:"cross_rate"`
	EffectiveTx            float64 `json:"effective_tx"`
	CrossTx                float64 `json:"cross_tx"`
	NormalizedLoadVariance float64 `json:"normalized_load_variance"`
	RCSTR                  float64 `json:"r_cstr"`
	RWLB                   float64 `json:"r_wlb"`
	AbsLoadDiff            float64 `json:"abs_load_diff"`
	CommunicationCost      float64 `json:"communication_cost"`
	MaxLoadShare           float64 `json:"max_load_share"`
	HotspotPenalty         float64 `json:"hotspot_penalty"`
	RewardMode             string  `json:"reward_mode"`
	Lambda                 float64 `json:"lambda"`
	Beta                   float64 `json:"beta"`
	TotalTx                int     `json:"total_tx"`
	TotalInner             int     `json:"total_inner"`
	TotalRelay1            int     `json:"total_relay1"`
	TotalRelay2            int     `json:"total_relay2"`

	RunningAvgCrossRate float64 `json:"running_avg_cross_rate"`
	RunningAvgReward    float64 `json:"running_avg_reward"`
	RunningAvgNormVar   float64 `json:"running_avg_norm_var"`
	RewardedEpochCount  int     `json:"rewarded_epoch_count"`
	FlushUpdate         bool    `json:"flush_update"`

	FeedbackAggregateCount  int     `json:"feedback_aggregate_count"`
	FeedbackFirstEpoch      int     `json:"feedback_first_epoch"`
	FeedbackLastEpoch       int     `json:"feedback_last_epoch"`
	FeedbackWeightSum       float64 `json:"feedback_weight_sum"`
	FeedbackAggregateWindow int     `json:"feedback_aggregate_window"`
	FeedbackMaxMatches      int     `json:"feedback_max_matches"`
}

const springPendingFeedbackTTL = 10
const springFeedbackAggregateWindow = 6
const springMaxTrainFeedbackMatches = 8
const springFeedbackDecay = 0.85
const springActionRewardScale = 0.1

func springClampFloat64(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func springRoundToInt(v float64) int {
	if v <= 0 {
		return 0
	}
	return int(math.Round(v))
}

func springFeedbackWeight(matchIndex int) float64 {
	if matchIndex <= 0 {
		return 1.0
	}
	return math.Pow(springFeedbackDecay, float64(matchIndex))
}

func (agg *SpringFeedbackAggregate) add(record SpringFeedbackRewardRecord, weight float64) {
	if weight <= 0 {
		weight = 1.0
	}
	if agg.Count == 0 {
		agg.FirstEpoch = record.Epoch
	}
	agg.Count++
	agg.LastEpoch = record.Epoch
	agg.WeightSum += weight

	agg.RewardSum += record.Reward * weight
	agg.CrossRateSum += record.CrossRate * weight
	agg.EffectiveTxSum += record.EffectiveTx * weight
	agg.CrossTxSum += record.CrossTx * weight
	agg.NormalizedLoadVarianceSum += record.NormalizedLoadVariance * weight
	agg.RCSTRSum += record.RCSTR * weight
	agg.RWLBSum += record.RWLB * weight
	agg.AbsLoadDiffSum += record.AbsLoadDiff * weight
	agg.CommunicationCostSum += record.CommunicationCost * weight
	agg.MaxLoadShareSum += record.MaxLoadShare * weight
	agg.HotspotPenaltySum += record.HotspotPenalty * weight
	agg.LambdaSum += record.Lambda * weight
	agg.BetaSum += record.Beta * weight
	agg.TotalTxSum += float64(record.TotalTx) * weight
	agg.TotalInnerSum += float64(record.TotalInner) * weight
	agg.TotalRelay1Sum += float64(record.TotalRelay1) * weight
	agg.TotalRelay2Sum += float64(record.TotalRelay2) * weight
}

func (agg SpringFeedbackAggregate) mean(sum float64) float64 {
	if agg.WeightSum <= 0 {
		return 0
	}
	return sum / agg.WeightSum
}

func (agg SpringFeedbackAggregate) reward() float64 {
	return agg.mean(agg.RewardSum)
}

func (agg SpringFeedbackAggregate) crossRate() float64 {
	return agg.mean(agg.CrossRateSum)
}

func (agg SpringFeedbackAggregate) effectiveTx() float64 {
	return agg.mean(agg.EffectiveTxSum)
}

func (agg SpringFeedbackAggregate) crossTx() float64 {
	return agg.mean(agg.CrossTxSum)
}

func (agg SpringFeedbackAggregate) normalizedLoadVariance() float64 {
	return agg.mean(agg.NormalizedLoadVarianceSum)
}

func (agg SpringFeedbackAggregate) rCSTR() float64 {
	return agg.mean(agg.RCSTRSum)
}

func (agg SpringFeedbackAggregate) rWLB() float64 {
	return agg.mean(agg.RWLBSum)
}

func (agg SpringFeedbackAggregate) absLoadDiff() float64 {
	return agg.mean(agg.AbsLoadDiffSum)
}

func (agg SpringFeedbackAggregate) communicationCost() float64 {
	return agg.mean(agg.CommunicationCostSum)
}

func (agg SpringFeedbackAggregate) maxLoadShare() float64 {
	return agg.mean(agg.MaxLoadShareSum)
}

func (agg SpringFeedbackAggregate) hotspotPenalty() float64 {
	return agg.mean(agg.HotspotPenaltySum)
}

func (agg SpringFeedbackAggregate) lambda() float64 {
	return agg.mean(agg.LambdaSum)
}

func (agg SpringFeedbackAggregate) beta() float64 {
	return agg.mean(agg.BetaSum)
}

func springBatchReadyForTraining(batch SpringTrainBatch, currentEpoch int, flushUpdate bool) bool {
	if batch.FeedbackAggregate.Count == 0 || batch.FeedbackAggregate.WeightSum <= 0 {
		return false
	}
	if flushUpdate {
		return true
	}
	if batch.FeedbackAggregate.Count >= springMaxTrainFeedbackMatches {
		return true
	}
	return currentEpoch-batch.FeedbackAggregate.FirstEpoch >= springFeedbackAggregateWindow
}

// springBuildFeedbackRewardRecord 根据一个 epoch 中所有 shard 的真实出块反馈计算 reward。
// 第一版 reward：
// reward = λ * (1 - crossRate) - (1 - λ) * normalizedLoadVariance
//
// 注意：
// 1. crossRate 只用 Relay1 计算，不把 Relay2 重复算入跨片率。
// 2. 如果这个 epoch 只有 Relay2，没有 inner 和 Relay1，说明只是跨片第二阶段收尾，跳过，不生成 reward。
// 3. 不做显式负载保护，只通过 reward 惩罚负载不均衡。
func (rthm *RelayCommitteeModule) springBuildFeedbackRewardRecord(
	epoch int,
	shardStats map[uint64]SpringBlockStat,
) (SpringFeedbackRewardRecord, bool) {
	decisionBatchSet := make(map[uint64]bool)
	lambda := params.SpringRewardLambda
	if lambda < 0 || lambda > 1 {
		lambda = 0.5
	}

	beta := params.SpringRewardBeta
	if beta <= 0 {
		beta = 0.1
	}

	eps := 1e-6

	loads := make([]int, params.ShardNum)
	effectiveLoads := make([]float64, params.ShardNum)
	inners := make([]int, params.ShardNum)
	relay1s := make([]int, params.ShardNum)
	relay2s := make([]int, params.ShardNum)

	totalTx := 0
	totalInner := 0
	totalRelay1 := 0
	totalRelay2 := 0
	effectiveTx := 0.0
	crossTx := 0.0
	communicationCostSum := 0.0
	communicationCostCount := 0

	for sid := 0; sid < params.ShardNum; sid++ {
		stat := shardStats[uint64(sid)]

		loads[sid] = stat.NumTx
		effectiveLoads[sid] = stat.EffectiveTx
		inners[sid] = stat.InnerTx
		relay1s[sid] = stat.Relay1Tx
		relay2s[sid] = stat.Relay2Tx

		totalTx += stat.NumTx
		totalInner += stat.InnerTx
		totalRelay1 += stat.Relay1Tx
		totalRelay2 += stat.Relay2Tx
		effectiveTx += stat.EffectiveTx
		crossTx += stat.CrossTx
		communicationCostSum += stat.CommunicationCostSum
		communicationCostCount += stat.CommunicationCostCount

		for _, bid := range stat.DecisionBatchIDs {
			decisionBatchSet[bid] = true
		}
	}

	// 全空 epoch 跳过。
	if totalTx == 0 && totalInner == 0 && totalRelay1 == 0 && totalRelay2 == 0 {
		return SpringFeedbackRewardRecord{}, false
	}

	// 如果这个 epoch 只有 Relay2，没有 inner 和 Relay1，
	// 说明它只是跨片第二阶段收尾，不应该作为 PPO 放置动作的反馈。
	decisionRelatedTx := totalInner + totalRelay1
	if decisionRelatedTx == 0 {
		return SpringFeedbackRewardRecord{}, false
	}

	if effectiveTx <= eps {
		return SpringFeedbackRewardRecord{}, false
	}

	// crossRate = 跨片交易比例。
	// Relay1 表示唯一跨片交易数，Relay2 是第二阶段，不重复计入。
	crossRate := float64(totalRelay1) / float64(decisionRelatedTx)

	crossRate = crossTx / effectiveTx
	if crossRate < 0 {
		crossRate = 0
	}
	if crossRate > 1 {
		crossRate = 1
	}

	// Paper formula: r_cstr = sum(num_tx_i) / (sum(cross_tx_i) + eps).
	// 但在 SPRING-Lite 小规模实验中，如果 cross=0，原式会非常大，导致 reward 爆炸。
	// 所以这里使用裁剪版：保留“跨片越少越好”的方向，但限制最大值。
	// In SPRING-Lite we keep CSTR bounded. The raw inverse-CSTR term can
	// become very large at cross=0 and encourages a single-shard collapse.
	rCSTR := 1.0 - crossRate
	if rCSTR < 0 {
		rCSTR = 0
	}
	if rCSTR > 1 {
		rCSTR = 1
	}

	// Paper workload-balance term: r_wlb = exp(-beta * abs_diff).
	avgLoad := effectiveTx / float64(params.ShardNum)

	rawAbsDiff := 0.0
	rawVar := 0.0

	for _, load := range effectiveLoads {
		diff := load - avgLoad
		rawAbsDiff += math.Abs(diff)
		rawVar += diff * diff
	}

	rawVar = rawVar / float64(params.ShardNum)

	// 保留 normalizedLoadVariance 用于日志观察
	normVar := rawVar / (avgLoad*avgLoad + eps)
	normVar = normVar / (1.0 + normVar)

	// 关键修改：用归一化负载差计算 r_wlb
	// 原来：rWLB = exp(-beta * rawAbsDiff)
	// 现在：rWLB = exp(-beta * normalizedAbsDiff)
	//
	// normalizedAbsDiff = sum_i |load_i - avgLoad| / avgLoad
	// 含义：总负载偏离量相当于几个“平均分片负载”
	normalizedAbsDiff := 0.0
	if avgLoad > eps {
		normalizedAbsDiff = rawAbsDiff / (avgLoad + eps)
	}

	balanceScore := 1.0 - normVar
	if balanceScore < 0 {
		balanceScore = 0
	}
	if balanceScore > 1 {
		balanceScore = 1
	}

	rWLB := math.Exp(-beta*normalizedAbsDiff) * balanceScore
	if rWLB < 0 {
		rWLB = 0
	}
	if rWLB > 1 {
		rWLB = 1
	}

	// SPRING-style reward:
	// r_t = λ * r_cstr + (1 - λ) * r_wlb
	maxLoadShare := 0.0
	if effectiveTx > eps {
		for _, load := range effectiveLoads {
			share := load / (effectiveTx + eps)
			if share > maxLoadShare {
				maxLoadShare = share
			}
		}
	}
	maxLoadShare = springClampFloat64(maxLoadShare, 0.0, 1.0)

	hotspotThreshold := springClampFloat64(params.SpringIOTHotspotThreshold, 0.0, 1.0-eps)
	hotspotPenalty := springClampFloat64(
		(maxLoadShare-hotspotThreshold)/(1.0-hotspotThreshold+eps),
		0.0,
		1.0,
	)

	communicationCost := 0.0
	if communicationCostCount > 0 {
		communicationCost = communicationCostSum / float64(communicationCostCount)
	}
	communicationCost = springClampFloat64(communicationCost, 0.0, 1.0)

	// Standard SPRING / Python paper mode reward. IoT mode below replaces it
	// with the dense-balanced reward used by the IoT mainline experiments.
	reward := lambda*rCSTR + (1.0-lambda)*rWLB
	rewardMode := "spring_legacy"
	if params.SpringIOTMode == 1 {
		// Match spring_lite/offline_env.py iot_dense_balanced:
		// 0.55*CSTR + 0.30*WLB - 0.10*communication_cost - 0.05*hotspot.
		rewardMode = "iot_dense_balanced"
		reward = params.SpringIOTCSTRWeight*rCSTR +
			params.SpringIOTBalanceWeight*rWLB -
			params.SpringIOTCommCostWeight*communicationCost -
			params.SpringIOTHotspotWeight*hotspotPenalty
	}

	if math.IsNaN(reward) || math.IsInf(reward, 0) {
		reward = 0.0
	}
	if math.IsNaN(crossRate) || math.IsInf(crossRate, 0) {
		crossRate = 0.0
	}
	if math.IsNaN(normVar) || math.IsInf(normVar, 0) {
		normVar = 0.0
	}
	if math.IsNaN(rCSTR) || math.IsInf(rCSTR, 0) {
		rCSTR = 0.0
	}
	if math.IsNaN(rWLB) || math.IsInf(rWLB, 0) {
		rWLB = 0.0
	}
	if math.IsNaN(communicationCost) || math.IsInf(communicationCost, 0) {
		communicationCost = 0.0
	}
	if math.IsNaN(maxLoadShare) || math.IsInf(maxLoadShare, 0) {
		maxLoadShare = 0.0
	}
	if math.IsNaN(hotspotPenalty) || math.IsInf(hotspotPenalty, 0) {
		hotspotPenalty = 0.0
	}

	decisionBatchIDs := make([]uint64, 0, len(decisionBatchSet))
	for bid := range decisionBatchSet {
		decisionBatchIDs = append(decisionBatchIDs, bid)
	}

	sort.Slice(decisionBatchIDs, func(i, j int) bool {
		return decisionBatchIDs[i] < decisionBatchIDs[j]
	})

	record := SpringFeedbackRewardRecord{
		TimeUnixNano: time.Now().UnixNano(),

		Epoch: epoch,

		TotalTx:     totalTx,
		TotalInner:  totalInner,
		TotalRelay1: totalRelay1,
		TotalRelay2: totalRelay2,
		EffectiveTx: effectiveTx,
		CrossTx:     crossTx,

		Loads:          loads,
		EffectiveLoads: effectiveLoads,
		Inners:         inners,
		Relay1:         relay1s,
		Relay2:         relay2s,

		DecisionBatchIDs: decisionBatchIDs,

		CrossRate:              crossRate,
		RawLoadVariance:        rawVar,
		NormalizedLoadVariance: normVar,

		RCSTR:       rCSTR,
		RWLB:        rWLB,
		AbsLoadDiff: normalizedAbsDiff,

		CommunicationCost:      communicationCost,
		CommunicationCostSum:   communicationCostSum,
		CommunicationCostCount: communicationCostCount,
		MaxLoadShare:           maxLoadShare,
		HotspotPenalty:         hotspotPenalty,

		Reward:     reward,
		RewardMode: rewardMode,

		Lambda: lambda,
		Beta:   beta,

		IOTCSTRWeight:       params.SpringIOTCSTRWeight,
		IOTBalanceWeight:    params.SpringIOTBalanceWeight,
		IOTCommCostWeight:   params.SpringIOTCommCostWeight,
		IOTHotspotWeight:    params.SpringIOTHotspotWeight,
		IOTHotspotThreshold: hotspotThreshold,
	}

	return record, true
}

func (rthm *RelayCommitteeModule) springAppendFeedbackRecord(record SpringFeedbackRewardRecord) {
	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		return
	}

	b, err := json.Marshal(record)
	if err != nil {
		return
	}

	path := filepath.Join("spring_io", "feedback_records.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	_, _ = f.Write(append(b, '\n'))
}

// 注意：这个函数默认在 rthm.springLock 已经加锁时调用。
// 不要在这里重复加锁，避免死锁。
func (rthm *RelayCommitteeModule) springEnqueueTrainActionsLocked(
	batchID uint64,
	actions []SpringTrainAction,
	txStartNonce uint64,
	txEndNonce uint64,
	txCount int,
	isFinalBatch bool,
) {
	if batchID == 0 || len(actions) == 0 {
		return
	}

	newBatch := SpringTrainBatch{
		BatchID:         batchID,
		TxStartNonce:    txStartNonce,
		TxEndNonce:      txEndNonce,
		TxCount:         txCount,
		EnqueueUnixNano: time.Now().UnixNano(),
		IsFinalBatch:    isFinalBatch,
		Actions:         actions,
	}

	if len(rthm.springPendingTrainBatches) > 0 {
		lastIdx := len(rthm.springPendingTrainBatches) - 1
		last := rthm.springPendingTrainBatches[lastIdx]

		if last.BatchID == batchID {
			if len(actions) >= len(last.Actions) {
				rthm.springPendingTrainBatches[lastIdx] = newBatch
			}

			rthm.sl.Slog.Printf(
				"[SPRING ONLINE ACTIONS] replace tx_batch_id=%d tx_nonce=[%d,%d] tx_count=%d actions=%d pending_batches=%d\n",
				batchID,
				txStartNonce,
				txEndNonce,
				txCount,
				len(actions),
				len(rthm.springPendingTrainBatches),
			)
			return
		}
	}

	rthm.springPendingTrainBatches = append(rthm.springPendingTrainBatches, newBatch)

	rthm.sl.Slog.Printf(
		"[SPRING ONLINE ACTIONS] enqueue tx_batch_id=%d tx_nonce=[%d,%d] tx_count=%d actions=%d pending_batches=%d\n",
		batchID,
		txStartNonce,
		txEndNonce,
		txCount,
		len(actions),
		len(rthm.springPendingTrainBatches),
	)
}

// 注意：这个函数默认在 rthm.springLock 已经加锁时调用。
// 它把最早的 PPO 动作批次和当前 reward 绑定，生成一个 online_update 输入。
func (rthm *RelayCommitteeModule) springPrunePendingTrainBatchesLocked(currentEpoch int) int {
	if springPendingFeedbackTTL <= 0 || len(rthm.springPendingTrainBatches) == 0 {
		return 0
	}

	kept := make([]SpringTrainBatch, 0, len(rthm.springPendingTrainBatches))
	pruned := 0

	for _, batch := range rthm.springPendingTrainBatches {
		if rthm.springTrainedBatchIDs[batch.BatchID] {
			pruned++
			continue
		}
		if batch.FeedbackAggregate.Count > 0 &&
			currentEpoch-batch.FeedbackAggregate.FirstEpoch > springPendingFeedbackTTL+springFeedbackAggregateWindow {
			pruned++
			continue
		}
		if batch.FeedbackAggregate.Count == 0 && currentEpoch-int(batch.BatchID) > springPendingFeedbackTTL {
			pruned++
			continue
		}
		kept = append(kept, batch)
	}

	rthm.springPendingTrainBatches = kept
	return pruned
}

// The update input is emitted after each tx batch has collected a short
// window of delayed feedback, so one action batch is still trained once.
func (rthm *RelayCommitteeModule) springBuildOnlineUpdateInputLocked(
	rewardRecord SpringFeedbackRewardRecord,
) (SpringOnlineUpdateInput, bool) {
	flushUpdate := rthm.nowDataNum >= rthm.dataTotalNum

	if len(rthm.springPendingTrainBatches) == 0 {
		alreadyTrainedTargets := 0
		for _, bid := range rewardRecord.DecisionBatchIDs {
			if rthm.springTrainedBatchIDs[bid] {
				alreadyTrainedTargets++
			}
		}
		if len(rewardRecord.DecisionBatchIDs) > 0 && alreadyTrainedTargets == len(rewardRecord.DecisionBatchIDs) {
			rthm.sl.Slog.Printf(
				"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=all_decision_batches_already_trained decision_batches=%v trained=%d pending_batches=0 reward=%.6f\n",
				rewardRecord.Epoch,
				rewardRecord.DecisionBatchIDs,
				alreadyTrainedTargets,
				rewardRecord.Reward,
			)
			return SpringOnlineUpdateInput{}, false
		}
		rthm.sl.Slog.Printf(
			"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=no_pending_train_batch reward=%.6f decision_batches=%v\n",
			rewardRecord.Epoch,
			rewardRecord.Reward,
			rewardRecord.DecisionBatchIDs,
		)
		return SpringOnlineUpdateInput{}, false
	}

	targetBatchSet := make(map[uint64]bool)
	alreadyTrainedTargets := 0
	for _, bid := range rewardRecord.DecisionBatchIDs {
		if rthm.springTrainedBatchIDs[bid] {
			alreadyTrainedTargets++
			continue
		}
		targetBatchSet[bid] = true
	}

	matchedThisEpoch := make([]uint64, 0)
	readyBatches := make([]SpringTrainBatch, 0)
	remainingBatches := make([]SpringTrainBatch, 0, len(rthm.springPendingTrainBatches))

	for _, batch := range rthm.springPendingTrainBatches {
		if rthm.springTrainedBatchIDs[batch.BatchID] {
			continue
		}

		if targetBatchSet[batch.BatchID] && batch.FeedbackMatches < springMaxTrainFeedbackMatches {
			weight := springFeedbackWeight(batch.FeedbackMatches)
			batch.FeedbackAggregate.add(rewardRecord, weight)
			batch.LastFeedbackEpoch = rewardRecord.Epoch
			if batch.FirstFeedbackEpoch == 0 {
				batch.FirstFeedbackEpoch = rewardRecord.Epoch
			}
			batch.FeedbackMatches++
			matchedThisEpoch = append(matchedThisEpoch, batch.BatchID)
		}

		if batch.IsFinalBatch {
			flushUpdate = true
		}

		if springBatchReadyForTraining(batch, rewardRecord.Epoch, flushUpdate) {
			readyBatches = append(readyBatches, batch)
			continue
		}

		remainingBatches = append(remainingBatches, batch)
	}

	rthm.springPendingTrainBatches = remainingBatches

	prunedPending := rthm.springPrunePendingTrainBatchesLocked(rewardRecord.Epoch)

	if len(readyBatches) == 0 {
		firstPending := uint64(0)
		if len(rthm.springPendingTrainBatches) > 0 {
			firstPending = rthm.springPendingTrainBatches[0].BatchID
		}
		if len(matchedThisEpoch) > 0 {
			rthm.sl.Slog.Printf(
				"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=aggregating_feedback matched_this_epoch=%v pending_batches=%d first_pending=%d pruned=%d reward=%.6f window=%d max_matches=%d\n",
				rewardRecord.Epoch,
				matchedThisEpoch,
				len(rthm.springPendingTrainBatches),
				firstPending,
				prunedPending,
				rewardRecord.Reward,
				springFeedbackAggregateWindow,
				springMaxTrainFeedbackMatches,
			)
			return SpringOnlineUpdateInput{}, false
		}
		if len(rewardRecord.DecisionBatchIDs) == 0 {
			rthm.sl.Slog.Printf(
				"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=no_decision_batch_ids reward=%.6f pending_batches=%d first_pending=%d pruned=%d\n",
				rewardRecord.Epoch,
				rewardRecord.Reward,
				len(rthm.springPendingTrainBatches),
				firstPending,
				prunedPending,
			)
			return SpringOnlineUpdateInput{}, false
		}
		if len(targetBatchSet) == 0 {
			rthm.sl.Slog.Printf(
				"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=all_decision_batches_already_trained decision_batches=%v trained=%d pending_batches=%d first_pending=%d pruned=%d reward=%.6f\n",
				rewardRecord.Epoch,
				rewardRecord.DecisionBatchIDs,
				alreadyTrainedTargets,
				len(rthm.springPendingTrainBatches),
				firstPending,
				prunedPending,
				rewardRecord.Reward,
			)
			return SpringOnlineUpdateInput{}, false
		}
		rthm.sl.Slog.Printf(
			"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=no_ready_train_batch decision_batches=%v pending_batches=%d first_pending=%d pruned=%d reward=%.6f\n",
			rewardRecord.Epoch,
			rewardRecord.DecisionBatchIDs,
			len(rthm.springPendingTrainBatches),
			firstPending,
			prunedPending,
			rewardRecord.Reward,
		)
		return SpringOnlineUpdateInput{}, false
	}

	sort.Slice(readyBatches, func(i, j int) bool {
		return readyBatches[i].BatchID < readyBatches[j].BatchID
	})

	matchedIDs := make([]uint64, 0, len(readyBatches))
	actions := make([]SpringTrainAction, 0)

	txStartNonce := readyBatches[0].TxStartNonce
	txEndNonce := readyBatches[0].TxEndNonce
	txCount := 0

	metricWeightSum := 0.0
	rewardSum := 0.0
	crossRateSum := 0.0
	effectiveTxSum := 0.0
	crossTxSum := 0.0
	normVarSum := 0.0
	rCSTRSum := 0.0
	rWLBSum := 0.0
	absLoadDiffSum := 0.0
	communicationCostMetricSum := 0.0
	maxLoadShareMetricSum := 0.0
	hotspotPenaltyMetricSum := 0.0
	lambdaSum := 0.0
	betaSum := 0.0
	totalTxSum := 0.0
	totalInnerSum := 0.0
	totalRelay1Sum := 0.0
	totalRelay2Sum := 0.0
	feedbackAggregateCount := 0
	feedbackWeightSum := 0.0
	feedbackFirstEpoch := 0
	feedbackLastEpoch := 0

	// SPRING-Lite reward propagation:
	// keep the delayed block-level CSTR/WLB signal, but give each action an
	// immediate shaped reward based on sender_pos and pre-action shard load.
	const localRewardWeight = 0.65

	for _, batch := range readyBatches {
		matchedIDs = append(matchedIDs, batch.BatchID)
		if batch.TxStartNonce < txStartNonce {
			txStartNonce = batch.TxStartNonce
		}
		if batch.TxEndNonce > txEndNonce {
			txEndNonce = batch.TxEndNonce
		}

		txCount += batch.TxCount
		if batch.IsFinalBatch {
			flushUpdate = true
		}

		agg := batch.FeedbackAggregate
		metricWeight := float64(len(batch.Actions))
		if metricWeight <= 0 {
			metricWeight = 1.0
		}
		metricWeightSum += metricWeight

		batchReward := agg.reward()
		rewardSum += batchReward * metricWeight
		crossRateSum += agg.crossRate() * metricWeight
		effectiveTxSum += agg.effectiveTx() * metricWeight
		crossTxSum += agg.crossTx() * metricWeight
		normVarSum += agg.normalizedLoadVariance() * metricWeight
		rCSTRSum += agg.rCSTR() * metricWeight
		rWLBSum += agg.rWLB() * metricWeight
		absLoadDiffSum += agg.absLoadDiff() * metricWeight
		communicationCostMetricSum += agg.communicationCost() * metricWeight
		maxLoadShareMetricSum += agg.maxLoadShare() * metricWeight
		hotspotPenaltyMetricSum += agg.hotspotPenalty() * metricWeight
		lambdaSum += agg.lambda() * metricWeight
		betaSum += agg.beta() * metricWeight
		totalTxSum += agg.mean(agg.TotalTxSum) * metricWeight
		totalInnerSum += agg.mean(agg.TotalInnerSum) * metricWeight
		totalRelay1Sum += agg.mean(agg.TotalRelay1Sum) * metricWeight
		totalRelay2Sum += agg.mean(agg.TotalRelay2Sum) * metricWeight

		feedbackAggregateCount += agg.Count
		feedbackWeightSum += agg.WeightSum
		if feedbackFirstEpoch == 0 || (agg.FirstEpoch > 0 && agg.FirstEpoch < feedbackFirstEpoch) {
			feedbackFirstEpoch = agg.FirstEpoch
		}
		if agg.LastEpoch > feedbackLastEpoch {
			feedbackLastEpoch = agg.LastEpoch
		}

		blockSignal := springClampFloat64(batchReward, -1.0, 1.0)
		for _, action := range batch.Actions {
			localSignal := springClampFloat64(action.LocalReward, -1.0, 1.0)
			shapedReward := localRewardWeight*localSignal + (1.0-localRewardWeight)*blockSignal
			shapedReward = springClampFloat64(shapedReward, -1.0, 1.0)

			action.Reward = springActionRewardScale * shapedReward
			action.Done = false
			actions = append(actions, action)
		}
	}

	if len(actions) == 0 {
		return SpringOnlineUpdateInput{}, false
	}

	for _, bid := range matchedIDs {
		rthm.springTrainedBatchIDs[bid] = true
	}

	actions[len(actions)-1].Done = true

	nextStates := make([][]float64, 0, len(actions))
	for idx, action := range actions {
		if idx+1 < len(actions) && len(actions[idx+1].State) > 0 {
			nextStates = append(nextStates, actions[idx+1].State)
		} else if len(action.NextState) > 0 {
			nextStates = append(nextStates, action.NextState)
		} else {
			nextStates = append(nextStates, action.State)
		}
	}

	avgMetric := func(sum float64) float64 {
		if metricWeightSum <= 0 {
			return 0
		}
		return sum / metricWeightSum
	}

	aggregateReward := avgMetric(rewardSum)
	aggregateCrossRate := avgMetric(crossRateSum)
	aggregateEffectiveTx := avgMetric(effectiveTxSum)
	aggregateCrossTx := avgMetric(crossTxSum)
	aggregateNormVar := avgMetric(normVarSum)
	aggregateRCSTR := avgMetric(rCSTRSum)
	aggregateRWLB := avgMetric(rWLBSum)
	aggregateAbsLoadDiff := avgMetric(absLoadDiffSum)
	aggregateCommunicationCost := avgMetric(communicationCostMetricSum)
	aggregateMaxLoadShare := avgMetric(maxLoadShareMetricSum)
	aggregateHotspotPenalty := avgMetric(hotspotPenaltyMetricSum)

	rthm.sl.Slog.Printf(
		"[SPRING ALIGN AGGREGATE] ready_batches=%v matched_this_epoch=%v tx_nonce=[%d,%d] tx_count=%d actions=%d -> feedback_epoch=%d feedback_window=[%d,%d] feedback_count=%d reward=%.6f crossRate=%.6f rWLB=%.6f normVar=%.6f commCost=%.6f maxShare=%.6f hotspot=%.6f pending_kept=%d pruned=%d flush=%v\n",
		matchedIDs,
		matchedThisEpoch,
		txStartNonce,
		txEndNonce,
		txCount,
		len(actions),
		rewardRecord.Epoch,
		feedbackFirstEpoch,
		feedbackLastEpoch,
		feedbackAggregateCount,
		aggregateReward,
		aggregateCrossRate,
		aggregateRWLB,
		aggregateNormVar,
		aggregateCommunicationCost,
		aggregateMaxLoadShare,
		aggregateHotspotPenalty,
		len(rthm.springPendingTrainBatches),
		prunedPending,
		flushUpdate,
	)

	input := SpringOnlineUpdateInput{
		TimeUnixNano: time.Now().UnixNano(),

		// batch_id 保留为 matched 的第一个 batch，方便兼容旧日志命名。
		BatchID: matchedIDs[0],

		MatchedBatchIDs: matchedIDs,

		TxStartNonce: txStartNonce,
		TxEndNonce:   txEndNonce,
		TxCount:      txCount,

		FeedbackEpoch: rewardRecord.Epoch,
		Shards:        params.ShardNum,
		IOTFeatureDim: springConfiguredIOTFeatureDim(),

		Actions: actions,

		Reward:     aggregateReward,
		NextStates: nextStates,

		Done: true,

		CrossRate:               aggregateCrossRate,
		EffectiveTx:             aggregateEffectiveTx,
		CrossTx:                 aggregateCrossTx,
		NormalizedLoadVariance:  aggregateNormVar,
		RCSTR:                   aggregateRCSTR,
		RWLB:                    aggregateRWLB,
		AbsLoadDiff:             aggregateAbsLoadDiff,
		CommunicationCost:       aggregateCommunicationCost,
		MaxLoadShare:            aggregateMaxLoadShare,
		HotspotPenalty:          aggregateHotspotPenalty,
		RewardMode:              rewardRecord.RewardMode,
		Lambda:                  avgMetric(lambdaSum),
		Beta:                    avgMetric(betaSum),
		TotalTx:                 springRoundToInt(avgMetric(totalTxSum)),
		TotalInner:              springRoundToInt(avgMetric(totalInnerSum)),
		TotalRelay1:             springRoundToInt(avgMetric(totalRelay1Sum)),
		TotalRelay2:             springRoundToInt(avgMetric(totalRelay2Sum)),
		RunningAvgCrossRate:     rewardRecord.RunningAvgCrossRate,
		RunningAvgReward:        rewardRecord.RunningAvgReward,
		RunningAvgNormVar:       rewardRecord.RunningAvgNormVar,
		RewardedEpochCount:      rewardRecord.RewardedEpochCount,
		FlushUpdate:             flushUpdate,
		FeedbackAggregateCount:  feedbackAggregateCount,
		FeedbackFirstEpoch:      feedbackFirstEpoch,
		FeedbackLastEpoch:       feedbackLastEpoch,
		FeedbackWeightSum:       feedbackWeightSum,
		FeedbackAggregateWindow: springFeedbackAggregateWindow,
		FeedbackMaxMatches:      springMaxTrainFeedbackMatches,
	}

	return input, true
}

func (rthm *RelayCommitteeModule) springWriteOnlineUpdateInput(input SpringOnlineUpdateInput) {
	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		return
	}

	b, err := json.Marshal(input)
	if err != nil {
		rthm.sl.Slog.Printf("[SPRING ONLINE UPDATE FILE] marshal failed: %v\n", err)
		return
	}

	path := filepath.Join(
		"spring_io",
		"online_update_batch_"+uint64ToString(input.BatchID)+"_epoch_"+intToString(input.FeedbackEpoch)+".json",
	)

	if err := os.WriteFile(path, b, 0644); err != nil {
		rthm.sl.Slog.Printf("[SPRING ONLINE UPDATE FILE] write failed: %v\n", err)
		return
	}

	rthm.sl.Slog.Printf(
		"[SPRING ONLINE UPDATE FILE] batch_id=%d epoch=%d actions=%d reward=%.6f mode=%s effective=%.1f cross=%.1f crossRate=%.6f runCross=%.6f normVar=%.6f commCost=%.6f maxShare=%.6f hotspot=%.6f feedback_count=%d feedback_window=[%d,%d] flush=%v file=%s\n",
		input.BatchID,
		input.FeedbackEpoch,
		len(input.Actions),
		input.Reward,
		input.RewardMode,
		input.EffectiveTx,
		input.CrossTx,
		input.CrossRate,
		input.RunningAvgCrossRate,
		input.NormalizedLoadVariance,
		input.CommunicationCost,
		input.MaxLoadShare,
		input.HotspotPenalty,
		input.FeedbackAggregateCount,
		input.FeedbackFirstEpoch,
		input.FeedbackLastEpoch,
		input.FlushUpdate,
		path,
	)
}

func uint64ToString(v uint64) string {
	return strconv.FormatUint(v, 10)
}

func intToString(v int) string {
	return strconv.FormatInt(int64(v), 10)
}
