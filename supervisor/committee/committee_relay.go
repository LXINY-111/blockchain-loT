package committee

import (
	"blockEmulator/core"
	"blockEmulator/message"
	"blockEmulator/networks"
	"blockEmulator/params"
	"blockEmulator/supervisor/signal"
	"blockEmulator/supervisor/supervisor_log"
	"blockEmulator/utils"
	"encoding/csv"
	"encoding/json"
	"io"
	"log"
	"math"
	"math/big"
	"math/rand"
	"os"
	"strings"
	"time"

	"sort"
	"sync"
)

type SpringBlockStat struct {
	NumTx       int     `json:"num_tx"`
	InnerTx     int     `json:"inner_tx"`
	Relay1Tx    int     `json:"relay1_tx"`
	Relay2Tx    int     `json:"relay2_tx"`
	CrossTx     float64 `json:"cross_tx"`
	EffectiveTx float64 `json:"effective_tx"`

	CommunicationCostSum   float64 `json:"communication_cost_sum"`
	CommunicationCostCount int     `json:"communication_cost_count"`

	// 当前 shard block 中和 PPO 放置决策直接相关的 TxBatch。
	// 只统计 InnerShardTxs + Relay1Txs，不统计 Relay2Txs，避免跨片第二阶段重复计算。
	DecisionBatchIDs []uint64 `json:"decision_batch_ids"`
}

func springCollectDecisionBatchIDs(txGroups ...[]*core.Transaction) []uint64 {
	batchSet := make(map[uint64]bool)

	txBatchSize := uint64(params.TxBatchSize)
	if txBatchSize == 0 {
		txBatchSize = 1
	}

	for _, txs := range txGroups {
		for _, tx := range txs {
			if tx == nil {
				continue
			}

			// nonce 从 0 开始，tx_batch_id 从 1 开始
			batchID := tx.Nonce/txBatchSize + 1
			batchSet[batchID] = true
		}
	}

	batchIDs := make([]uint64, 0, len(batchSet))
	for bid := range batchSet {
		batchIDs = append(batchIDs, bid)
	}

	sort.Slice(batchIDs, func(i, j int) bool {
		return batchIDs[i] < batchIDs[j]
	})

	return batchIDs
}

type RelayCommitteeModule struct {
	csvPath      string
	dataTotalNum int
	nowDataNum   int
	batchDataNum int

	IpNodeTable map[uint64]map[uint64]string
	sl          *supervisor_log.SupervisorLog
	Ss          *signal.StopSignal

	// SPRING: Supervisor 临时代替 A-Shard 保存全局地址放置表
	springLock      sync.Mutex
	springAddrShard map[string]uint64
	springShardLoad []int

	// SPRING: 保存最近若干个块的每分片交易数和跨片交易数
	// 后续 PPO 状态向量要用
	springStats map[uint64][]SpringBlockStat

	// SPRING 在线训练第一步：按 epoch 收集真实区块反馈，用于计算 reward
	springEpochFeedback map[int]map[uint64]SpringBlockStat
	springRewardedEpoch map[int]bool

	// SPRING 在线训练：等待和真实区块 reward 匹配的 PPO 动作批次
	springPendingTrainBatches []SpringTrainBatch
	springTrainedBatchIDs     map[uint64]bool

	// SPRING metrics: running averages for each feedback epoch.
	springRewardCount  int
	springCrossRateSum float64
	springRewardSum    float64
	springNormVarSum   float64

	// SPRING: 新地址放置动作编号，用于生成 action_1.json、action_2.json
	springActionSeq uint64

	// SPRING random baseline（随机基线）：按固定 seed（随机种子）复现随机放置。
	springRandom *rand.Rand

	// SPRING: 真实 TxBatch 编号，用于核对 PPO action 和 block reward 是否对齐
	springTxBatchSeq uint64

	// IoT MDP: tx_index -> sidecar 特征。打开 SpringIOTMode 后，
	// Supervisor 会把原始交易地址映射为“状态对象账户 -> 设备锚点账户”。
	springIOTByTxIndex map[uint64]SpringIOTTxFeature
}

func NewRelayCommitteeModule(Ip_nodeTable map[uint64]map[uint64]string, Ss *signal.StopSignal, slog *supervisor_log.SupervisorLog, csvFilePath string, dataNum, batchNum int) *RelayCommitteeModule {
	springStats := make(map[uint64][]SpringBlockStat)
	for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
		springStats[sid] = make([]SpringBlockStat, 0, 5)
	}
	rthm := &RelayCommitteeModule{
		csvPath:      csvFilePath,
		dataTotalNum: dataNum,
		batchDataNum: batchNum,
		nowDataNum:   0,
		IpNodeTable:  Ip_nodeTable,
		Ss:           Ss,
		sl:           slog,

		springAddrShard:       make(map[string]uint64),
		springShardLoad:       make([]int, params.ShardNum),
		springStats:           springStats,
		springEpochFeedback:   make(map[int]map[uint64]SpringBlockStat),
		springRewardedEpoch:   make(map[int]bool),
		springTrainedBatchIDs: make(map[uint64]bool),
		springRandom:          rand.New(rand.NewSource(params.SpringRandomSeed)),
	}

	if springIOTEnabled() {
		features, err := springLoadIOTSidecar(params.SpringIOTSidecarFile)
		if err != nil {
			log.Panicf("load IoT sidecar failed: %v", err)
		}
		rthm.springIOTByTxIndex = features
		if rthm.sl != nil {
			rthm.sl.Slog.Printf(
				"[IOT MDP] loaded sidecar=%s rows=%d feature_dim=%d model=%s\n",
				params.SpringIOTSidecarFile,
				len(features),
				params.SpringIOTFeatureDim,
				params.SpringModelFile,
			)
		}
	}

	return rthm
}

// transfrom, data to transaction
// check whether it is a legal txs meesage. if so, read txs and put it into the txlist
func data2tx(data []string, nonce uint64) (*core.Transaction, bool) {
	if data[6] == "0" && data[7] == "0" && len(data[3]) > 16 && len(data[4]) > 16 && data[3] != data[4] {
		val, ok := new(big.Int).SetString(data[8], 10)
		if !ok {
			log.Panic("new int failed\n")
		}
		tx := core.NewTransaction(data[3][2:], data[4][2:], val, nonce, time.Now())
		return tx, true
	}
	return &core.Transaction{}, false
}

func (rthm *RelayCommitteeModule) HandleOtherMessage([]byte) {}

// SPRING: 判断地址是否已经被放置；如果没有，就根据 SpringMode 给它分片
func (rthm *RelayCommitteeModule) springEnsurePlaced(
	addr utils.Address,
	related utils.Address,
	batchPlacement map[string]uint64,
) uint64 {
	if sid, ok := rthm.springAddrShard[string(addr)]; ok {
		return sid
	}

	var sid uint64

	switch params.SpringMode {
	case 1:
		// SPRING-Heuristic：只使用 Go 里的启发式规则，不调用 Python
		sid = rthm.springChooseShard(addr, related)

	case 2:
		// SPRING-PPO：调用 Python PPO；如果 Python 失败，springChooseShardPPO 内部会自动回退到启发式
		sid = rthm.springChooseShardPPO(addr, related)

	case 3:
		// SPRING-Random baseline（随机基线）：只随机选择分片，不使用交互关系。
		sid = rthm.springChooseShardRandom()

	case 4:
		// MinState baseline（最少状态优先）：新状态放到当前状态数量最少的分片，不使用通信关系或 PPO。
		sid = rthm.springChooseShardMinState()

	default:
		// SpringMode = 0 或其他非法值：退化为原始 Hash 放置
		sid = uint64(utils.Addr2Shard(addr))
	}

	rthm.springAddrShard[string(addr)] = sid
	rthm.springShardLoad[sid]++
	batchPlacement[string(addr)] = sid

	if rthm.sl != nil && rthm.sl.Slog != nil {
		rthm.sl.Slog.Printf(
			"[SPRING PLACE] mode=%d addr=%s shard=%d related=%s totalPlaced=%d\n",
			params.SpringMode,
			addr,
			sid,
			related,
			len(rthm.springAddrShard),
		)
	}

	return sid
}

func (rthm *RelayCommitteeModule) springEnsurePlacedWithBatchRelated(
	addr utils.Address,
	related utils.Address,
	batchPlacement map[string]uint64,
	batchRelated map[string]map[string]bool,
) uint64 {
	if sid, ok := rthm.springAddrShard[string(addr)]; ok {
		return sid
	}

	senderPos, _, relatedKnown, _, _, _, _ := rthm.springBuildSenderPos(
		string(addr),
		related,
		batchPlacement,
		batchRelated,
	)

	var sid uint64
	if relatedKnown {
		sid = rthm.springChooseShardFromSenderPos(addr, senderPos)
	} else {
		sid = rthm.springChooseShard(addr, related)
	}
	if guardedSid, changed := rthm.springApplyCandidateGuard(addr, senderPos, sid); changed {
		sid = guardedSid
	}

	rthm.springAddrShard[string(addr)] = sid
	rthm.springShardLoad[sid]++
	batchPlacement[string(addr)] = sid
	return sid
}

func (rthm *RelayCommitteeModule) springChooseShardFromSenderPos(
	addr utils.Address,
	senderPos []float64,
) uint64 {
	hashSid := uint64(utils.Addr2Shard(addr))
	bestSid := uint64(0)
	bestScore := math.Inf(-1)

	for sid := 0; sid < params.ShardNum; sid++ {
		relatedScore := 0.0
		if sid < len(senderPos) {
			relatedScore = senderPos[sid]
		}
		score := relatedScore*1000.0 - float64(rthm.springShardLoad[sid])
		if uint64(sid) == hashSid {
			score += 0.001
		}
		if score > bestScore {
			bestScore = score
			bestSid = uint64(sid)
		}
	}

	return bestSid
}

func (rthm *RelayCommitteeModule) springChooseShardRandom() uint64 {
	if params.ShardNum <= 0 {
		return 0
	}
	if rthm.springRandom == nil {
		rthm.springRandom = rand.New(rand.NewSource(params.SpringRandomSeed))
	}
	return uint64(rthm.springRandom.Intn(params.ShardNum))
}

func (rthm *RelayCommitteeModule) springChooseShardMinState() uint64 {
	if params.ShardNum <= 0 {
		return 0
	}

	bestSid := 0
	bestLoad := int(^uint(0) >> 1)
	for sid := 0; sid < params.ShardNum; sid++ {
		load := 0
		if rthm != nil && sid < len(rthm.springShardLoad) {
			load = rthm.springShardLoad[sid]
		}
		if load < bestLoad {
			bestLoad = load
			bestSid = sid
		}
	}
	return uint64(bestSid)
}

// SPRING 第一版简单策略：
// 1. 如果 related 地址已经有分片，优先放到 related 的分片，降低跨片交易
// 2. 同时考虑当前放置负载，避免所有新地址都堆到一个分片
// 3. 如果没有 related，则退化为负载最小 + 哈希打破平局
func (rthm *RelayCommitteeModule) springChooseShard(
	addr utils.Address,
	related utils.Address,
) uint64 {
	relatedSid := -1
	if related != "" {
		if sid, ok := rthm.springAddrShard[string(related)]; ok {
			relatedSid = int(sid)
		}
	}

	hashSid := uint64(utils.Addr2Shard(addr))

	bestSid := uint64(0)
	bestScore := -1 << 60

	for sid := 0; sid < params.ShardNum; sid++ {
		score := 0

		// 交互关系分：如果新地址的交易对手在这个分片，强烈倾向于放一起
		if sid == relatedSid {
			score += 1000
		}

		// 负载惩罚：分片已经放置的地址越多，分数越低
		score -= rthm.springShardLoad[sid]

		// 哈希结果只作为平局打破项
		if uint64(sid) == hashSid {
			score += 1
		}

		if score > bestScore {
			bestScore = score
			bestSid = uint64(sid)
		}
	}

	return bestSid
}

// SPRING: 对一批交易提前完成新地址放置
func (rthm *RelayCommitteeModule) springPreparePlacement(
	txlist []*core.Transaction,
) map[string]uint64 {
	rthm.springLock.Lock()
	defer rthm.springLock.Unlock()

	batchPlacement := make(map[string]uint64)

	switch params.SpringMode {
	case 0:
		// 原始 Hash Relay，不做 SPRING 放置。
		return batchPlacement

	case 1:
		if springIOTEnabled() {
			// IoT heuristic baseline：设备账户是外部锚点，先用 hash（哈希）
			// 固定到分片；heuristic（启发式）基于多锚点 sender_pos 放置通信状态对象账户。
			batchRelated := rthm.springBuildIOTBatchRelatedMap(txlist)
			rthm.springSeedIOTAnchorShards(txlist, batchPlacement)
			for _, tx := range txlist {
				rthm.springEnsurePlacedWithBatchRelated(tx.Sender, tx.Recipient, batchPlacement, batchRelated)
			}
			rthm.springFillTouchedPlacement(txlist, batchPlacement)
			return batchPlacement
		}
		for _, tx := range txlist {
			rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)
			rthm.springEnsurePlaced(tx.Recipient, tx.Sender, batchPlacement)
		}
		rthm.springFillTouchedPlacement(txlist, batchPlacement)
		return batchPlacement

	case 2:
		rthm.springPreparePlacementPPOBatch(txlist, batchPlacement)
		rthm.springFillTouchedPlacement(txlist, batchPlacement)
		return batchPlacement

	case 3:
		if springIOTEnabled() {
			// IoT random baseline：设备/锚点仍用 hash（哈希）固定，只有状态对象随机放置。
			rthm.springSeedIOTAnchorShards(txlist, batchPlacement)
			for _, tx := range txlist {
				rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)
			}
			rthm.springFillTouchedPlacement(txlist, batchPlacement)
			return batchPlacement
		}
		for _, tx := range txlist {
			rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)
			rthm.springEnsurePlaced(tx.Recipient, tx.Sender, batchPlacement)
		}
		rthm.springFillTouchedPlacement(txlist, batchPlacement)
		return batchPlacement

	case 4:
		if springIOTEnabled() {
			// IoT MinState baseline（最少状态优先）：锚点仍用 hash 固定，只把状态对象放到状态数量最少的分片。
			rthm.springSeedIOTAnchorShards(txlist, batchPlacement)
			for _, tx := range txlist {
				rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)
			}
			rthm.springFillTouchedPlacement(txlist, batchPlacement)
			return batchPlacement
		}
		for _, tx := range txlist {
			rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)
			rthm.springEnsurePlaced(tx.Recipient, tx.Sender, batchPlacement)
		}
		rthm.springFillTouchedPlacement(txlist, batchPlacement)
		return batchPlacement

	default:
		// 非法模式兜底：当作 Hash 放置，但仍同步 PlacementMap，避免空映射导致异常。
		for _, tx := range txlist {
			rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)
			rthm.springEnsurePlaced(tx.Recipient, tx.Sender, batchPlacement)
		}
		return batchPlacement
	}
}

func (rthm *RelayCommitteeModule) springFillTouchedPlacement(
	txlist []*core.Transaction,
	batchPlacement map[string]uint64,
) {
	for _, tx := range txlist {
		if sid, ok := rthm.springAddrShard[string(tx.Sender)]; ok {
			batchPlacement[string(tx.Sender)] = sid
		}

		if sid, ok := rthm.springAddrShard[string(tx.Recipient)]; ok {
			batchPlacement[string(tx.Recipient)] = sid
		}
	}
}

func springBuildBatchRelatedMap(txlist []*core.Transaction) map[string]map[string]bool {
	related := make(map[string]map[string]bool)

	add := func(a, b utils.Address) {
		if a == "" || b == "" || a == b {
			return
		}
		key := string(a)
		if _, ok := related[key]; !ok {
			related[key] = make(map[string]bool)
		}
		related[key][string(b)] = true
	}

	for _, tx := range txlist {
		if tx == nil {
			continue
		}

		switch params.SpringSenderPosMode {
		case 1:
			// Paper-like sender_pos: for a recipient, count senders in the
			// current A-Shard block that are already placed.
			add(tx.Recipient, tx.Sender)
		default:
			// Existing SPRING-Lite behavior: use the bidirectional relation
			// graph built from the current TxBatch.
			add(tx.Sender, tx.Recipient)
			add(tx.Recipient, tx.Sender)
		}
	}

	return related
}

func (rthm *RelayCommitteeModule) springPreparePlacementPPOBatch(
	txlist []*core.Transaction,
	batchPlacement map[string]uint64,
) {
	// 这个 txBatchID 是真正的 TxBatch 编号，不再使用单个 PPO 推理请求的 batch_id 代替。
	// springPreparePlacement() 外层已经持有 springLock，所以这里直接自增即可。
	rthm.springTxBatchSeq++
	txBatchID := rthm.springTxBatchSeq

	txStartNonce := uint64(0)
	txEndNonce := uint64(0)
	if len(txlist) > 0 {
		txStartNonce = txlist[0].Nonce
		txEndNonce = txlist[len(txlist)-1].Nonce
	}

	rthm.sl.Slog.Printf(
		"[SPRING ALIGN PREPARE] tx_batch_id=%d tx_nonce=[%d,%d] tx_count=%d inject_speed=%d tx_batch_size=%d\n",
		txBatchID,
		txStartNonce,
		txEndNonce,
		len(txlist),
		params.InjectSpeed,
		params.TxBatchSize,
	)

	// Build a block-level relation map first. This is closer to SPRING's
	// sender_pos semantics than using only one counterparty from one tx.
	batchRelated := springBuildBatchRelatedMap(txlist)

	// SPRING paper semantics: sender_pos is updated after each placement
	// action within the current A-Shard block.
	trainActions := make([]SpringTrainAction, 0)

	if springIOTEnabled() {
		// IoT MDP：PPO 只放置通信状态对象账户；设备账户作为外部锚点先稳定哈希分片。
		batchRelated = rthm.springBuildIOTBatchRelatedMap(txlist)
		rthm.springSeedIOTAnchorShards(txlist, batchPlacement)
		for _, tx := range txlist {
			if action, ok := rthm.springPlaceAddressPPOSequential(
				tx.Sender,
				tx.Recipient,
				batchPlacement,
				batchRelated,
				rthm.springIOTFeaturesForTx(tx),
			); ok {
				trainActions = append(trainActions, action)
			}
		}
	} else if params.SpringSenderPosMode == 1 {
		// Put senders first, then recipients. This makes recipient sender_pos
		// closer to the SPRING paper definition: distribution of related
		// senders' shards in the current block.
		for _, tx := range txlist {
			if action, ok := rthm.springPlaceAddressPPOSequential(tx.Sender, "", batchPlacement, batchRelated); ok {
				trainActions = append(trainActions, action)
			}
		}
		for _, tx := range txlist {
			if action, ok := rthm.springPlaceAddressPPOSequential(tx.Recipient, tx.Sender, batchPlacement, batchRelated); ok {
				trainActions = append(trainActions, action)
			}
		}
	} else {
		for _, tx := range txlist {
			if action, ok := rthm.springPlaceAddressPPOSequential(tx.Sender, tx.Recipient, batchPlacement, batchRelated); ok {
				trainActions = append(trainActions, action)
			}

			if action, ok := rthm.springPlaceAddressPPOSequential(tx.Recipient, tx.Sender, batchPlacement, batchRelated); ok {
				trainActions = append(trainActions, action)
			}
		}
	}

	if params.SpringOnlineTrain == 1 && len(trainActions) > 0 {
		rthm.springEnqueueTrainActionsLocked(
			txBatchID,
			trainActions,
			txStartNonce,
			txEndNonce,
			len(txlist),
			rthm.nowDataNum >= rthm.dataTotalNum,
		)
	} else {
		rthm.sl.Slog.Printf(
			"[SPRING ALIGN PREPARE NO_ACTION] tx_batch_id=%d tx_nonce=[%d,%d] tx_count=%d actions=%d online_train=%d\n",
			txBatchID,
			txStartNonce,
			txEndNonce,
			len(txlist),
			len(trainActions),
			params.SpringOnlineTrain,
		)
	}
}

func (rthm *RelayCommitteeModule) springPlaceAddressPPOSequential(
	addr utils.Address,
	related utils.Address,
	batchPlacement map[string]uint64,
	batchRelated map[string]map[string]bool,
	iotFeatures ...[]float64,
) (SpringTrainAction, bool) {
	key := string(addr)
	if key == "" {
		return SpringTrainAction{}, false
	}

	// 已经放置过的地址，不再重复决策。
	if _, ok := rthm.springAddrShard[key]; ok {
		return SpringTrainAction{}, false
	}

	relatedKey := string(related)

	// related_in_current_batch：
	// 表示 related 地址是否是在当前 TxBatch 内已经被前面的 sequential 决策放置过。
	relatedInCurrentBatch := false
	if relatedKey != "" {
		if _, ok := batchPlacement[relatedKey]; ok {
			relatedInCurrentBatch = true
		}
	}

	// 构造 PPO 状态。
	// 当前代码里的 springBuildState 已经会把最近 5 个块的 totalTx/crossTx 归一化到 0~1。
	state := rthm.springBuildState(related, batchPlacement)

	// 从 state 的 sender_pos 区域反推 related 是否已知、在哪个 shard。
	// 这样可以保证诊断字段和 PPO 实际看到的 state 一致。
	relatedKnown, relatedShard := springExtractRelatedShardFromState(state)

	senderPos, relatedSummary, aggRelatedKnown, aggRelatedShard, relatedWeight, aggRelatedInCurrentBatch, relatedCount := rthm.springBuildSenderPos(
		key,
		related,
		batchPlacement,
		batchRelated,
	)
	// Override the old one-counterparty state with block-level sender_pos.
	relatedKey = relatedSummary
	relatedKnown = aggRelatedKnown
	relatedShard = aggRelatedShard
	relatedInCurrentBatch = aggRelatedInCurrentBatch
	var extraFeatures []float64
	if len(iotFeatures) > 0 {
		extraFeatures = iotFeatures[0]
	}
	state = rthm.springBuildStateFromSenderPos(senderPos, springAddressFlagFromRelatedCount(relatedCount), extraFeatures)
	actionMask := rthm.springBuildCandidateActionMask(addr, senderPos)

	item := SpringBatchInferItem{
		Address:    key,
		Related:    relatedKey,
		State:      state,
		ActionMask: actionMask,
	}

	results, inferCostUs, ok := rthm.springCallPythonBatch([]SpringBatchInferItem{item})

	result := SpringBatchInferResult{}
	if ok && len(results) == 1 {
		result = results[0]
	}

	var sid uint64
	source := ""
	confidence := 0.0
	entropy := 0.0

	if ok && result.Shard >= 0 && result.Shard < params.ShardNum {
		sid = uint64(result.Shard)
		source = result.Source
		confidence = result.Confidence
		entropy = result.Entropy
		if source == "" {
			source = "python_ppo"
		}
	} else {
		if aggRelatedKnown {
			sid = rthm.springChooseShardFromSenderPos(addr, senderPos)
			source = "go_multi_anchor_heuristic_sequential_fallback"
		} else {
			sid = rthm.springChooseShard(addr, related)
			source = "go_heuristic_sequential_fallback"
		}
	}
	if guardedSid, changed := rthm.springApplyCandidateGuard(addr, senderPos, sid); changed {
		sid = guardedSid
		if source == "" {
			source = "python_ppo"
		}
		source = source + "_capacity_guard"
		confidence = 0.0
		entropy = 0.0
		result.LogProb = 0.0
	}

	// sequential 语义的关键：
	// 当前地址决策后立即落表，后续地址构造 state 时能看到它。
	chosenShard := int(sid)
	shardLoadBefore := 0
	if chosenShard >= 0 && chosenShard < len(rthm.springShardLoad) {
		shardLoadBefore = rthm.springShardLoad[chosenShard]
	}
	loadMeanBefore := springMeanInt(rthm.springShardLoad)
	localReward, loadPenalty := springLocalActionReward(
		chosenShard,
		relatedKnown,
		relatedShard,
		relatedWeight,
		shardLoadBefore,
		loadMeanBefore,
	)

	rthm.springAddrShard[key] = sid
	rthm.springShardLoad[sid]++
	batchPlacement[key] = sid

	sameAsRelated := relatedKnown && chosenShard == relatedShard
	nextSenderPos, _, _, _, _, _, nextRelatedCount := rthm.springBuildSenderPos(
		key,
		related,
		batchPlacement,
		batchRelated,
	)
	nextState := rthm.springBuildStateFromSenderPos(nextSenderPos, springAddressFlagFromRelatedCount(nextRelatedCount), extraFeatures)

	rthm.springAppendDecisionRecord(
		result.BatchID,
		key,
		item.Related,
		sid,
		source,
		confidence,
		entropy,
		result.LogProb,
		result.Value,
		state,
		inferCostUs,
		1,
		actionMask,
	)
	/*
		rthm.sl.Slog.Printf(
			"[SPRING PPO SEQ PLACE] mode=%d addr=%s shard=%d related=%s source=%s confidence=%.6f related_known=%v related_shard=%d chosen_shard=%d same_as_related=%v related_in_current_batch=%v totalPlaced=%d\n",
			params.SpringMode,
			key,
			sid,
			item.Related,
			source,
			confidence,
			relatedKnown,
			relatedShard,
			chosenShard,
			sameAsRelated,
			relatedInCurrentBatch,
			len(rthm.springAddrShard),
		)*/

	// 只有真正由 PPO 网络产生的动作，才进入在线训练。
	// heuristic fallback 只是保证系统能跑，不作为 PPO 经验。
	if source != "python_ppo" || result.BatchID == 0 {
		return SpringTrainAction{}, false
	}

	return SpringTrainAction{
		BatchID:               result.BatchID,
		Address:               key,
		Related:               item.Related,
		State:                 state,
		NextState:             nextState,
		Action:                chosenShard,
		LogProb:               result.LogProb,
		Value:                 result.Value,
		Entropy:               entropy,
		Confidence:            confidence,
		LocalReward:           localReward,
		RelatedKnown:          relatedKnown,
		RelatedShard:          relatedShard,
		RelatedWeight:         relatedWeight,
		RelatedCount:          relatedCount,
		SenderPos:             senderPos,
		ActionMask:            append([]int(nil), actionMask...),
		ChosenShard:           chosenShard,
		SameAsRelated:         sameAsRelated,
		RelatedInCurrentBatch: relatedInCurrentBatch,
		ShardLoadBefore:       shardLoadBefore,
		ShardLoadMeanBefore:   loadMeanBefore,
		LoadPenalty:           loadPenalty,
	}, true
}

func (rthm *RelayCommitteeModule) txSending(txlist []*core.Transaction) {
	useSpringPlacement := params.SpringMode != 0

	batchPlacement := make(map[string]uint64)

	// SpringMode = 1 或 2 时，才进行 SPRING 新地址放置
	// SpringMode = 0 时，直接走原始 Hash Relay
	if useSpringPlacement {
		batchPlacement = rthm.springPreparePlacement(txlist)
	}

	// the txs will be sent
	sendToShard := make(map[uint64][]*core.Transaction)

	for idx := 0; idx <= len(txlist); idx++ {
		if idx > 0 && (idx%params.InjectSpeed == 0 || idx == len(txlist)) {
			if useSpringPlacement {
				counts := make([]int, params.ShardNum)
				for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
					counts[sid] = len(sendToShard[sid])
				}

				rthm.sl.Slog.Printf(
					"[SPRING SEND] nowData=%d txlistLen=%d idx=%d counts=%v placementSize=%d\n",
					rthm.nowDataNum,
					len(txlist),
					idx,
					counts,
					len(batchPlacement),
				)
			}
			// send to shard
			for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
				it := message.InjectTxs{
					Txs:       sendToShard[sid],
					ToShardID: sid,
				}

				// 只有 SPRING-Heuristic / SPRING-PPO 才需要同步放置表
				// 原始 Hash Relay 不需要 PlacementMap
				if useSpringPlacement {
					it.PlacementMap = batchPlacement
				}

				itByte, err := json.Marshal(it)
				if err != nil {
					log.Panic(err)
				}

				sendMsg := message.MergeMessage(message.CInject, itByte)
				for _, ip := range rthm.IpNodeTable[sid] {
					go networks.TcpDial(sendMsg, ip)
				}
			}

			sendToShard = make(map[uint64][]*core.Transaction)
			time.Sleep(time.Second)
		}

		if idx == len(txlist) {
			break
		}

		tx := txlist[idx]

		var sendersid uint64

		if useSpringPlacement {
			// SPRING-Heuristic / SPRING-PPO：
			// 交易发送到 sender 在 SPRING 放置表中的分片
			sendersid = rthm.springAddrShard[string(tx.Sender)]
		} else {
			// 原始 Hash Relay：
			// 交易发送到 sender 哈希映射得到的分片
			sendersid = uint64(utils.Addr2Shard(tx.Sender))
		}

		sendToShard[sendersid] = append(sendToShard[sendersid], tx)
	}
}

// read transactions, the Number of the transactions is - batchDataNum
func (rthm *RelayCommitteeModule) MsgSendingControl() {
	txfile, err := os.Open(rthm.csvPath)
	if err != nil {
		log.Panic(err)
	}
	defer txfile.Close()
	reader := csv.NewReader(txfile)
	txlist := make([]*core.Transaction, 0) // save the txs in this epoch (round)

	for {
		data, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Panic(err)
		}
		if tx, ok := data2tx(data, uint64(rthm.nowDataNum)); ok {
			if springIOTEnabled() && !rthm.springApplyIOTTxIdentity(tx) {
				log.Panicf("IoT sidecar missing for tx_index=%d", tx.Nonce)
			}
			txlist = append(txlist, tx)
			rthm.nowDataNum++
		}

		// re-shard condition, enough edges
		if len(txlist) == int(rthm.batchDataNum) || rthm.nowDataNum == rthm.dataTotalNum {
			rthm.txSending(txlist)
			// reset the variants about tx sending
			txlist = make([]*core.Transaction, 0)
			rthm.Ss.StopGap_Reset()
		}

		if rthm.nowDataNum == rthm.dataTotalNum {
			break
		}
	}
}

func (rthm *RelayCommitteeModule) HandleBlockInfo(b *message.BlockInfoMsg) {

	rthm.springLock.Lock()
	defer rthm.springLock.Unlock()

	// 注意：
	// 1. CrossTx 第一版只用 Relay1Tx。
	// 2. Relay2Tx 是跨片交易第二阶段，不要和 Relay1 一起重复计入 crossRate。
	decisionBatchIDs := springCollectDecisionBatchIDs(
		b.InnerShardTxs,
		b.Relay1Txs,
	)

	crossTx := float64(len(b.Relay1Txs)+len(b.Relay2Txs)) / 2.0
	effectiveTx := float64(len(b.InnerShardTxs)) + crossTx
	communicationCostSum, communicationCostCount := 0.0, 0
	if params.SpringIOTMode == 1 {
		communicationCostSum, communicationCostCount = rthm.springIOTCommunicationCostForTxGroups(
			b.InnerShardTxs,
			b.Relay1Txs,
		)
	}

	stat := SpringBlockStat{
		NumTx:                  b.BlockBodyLength,
		InnerTx:                len(b.InnerShardTxs),
		Relay1Tx:               len(b.Relay1Txs),
		Relay2Tx:               len(b.Relay2Txs),
		CrossTx:                crossTx,
		EffectiveTx:            effectiveTx,
		CommunicationCostSum:   communicationCostSum,
		CommunicationCostCount: communicationCostCount,
		DecisionBatchIDs:       decisionBatchIDs,
	}

	rthm.sl.Slog.Printf(
		"[BLOCK INFO] shard=%d epoch=%d body=%d inner=%d relay1=%d relay2=%d commCostSum=%.6f commCostCount=%d decision_batches=%v\n",
		b.SenderShardID,
		b.Epoch,
		b.BlockBodyLength,
		len(b.InnerShardTxs),
		len(b.Relay1Txs),
		len(b.Relay2Txs),
		communicationCostSum,
		communicationCostCount,
		decisionBatchIDs,
	)

	// 更新最近 5 个区块窗口，供 springBuildState() 继续使用。
	// 这里不再跳过 body=0 的空块，因为空块也代表该 shard 当前负载为 0。
	arr := rthm.springStats[b.SenderShardID]
	arr = append(arr, stat)
	if len(arr) > 5 {
		arr = arr[len(arr)-5:]
	}
	rthm.springStats[b.SenderShardID] = arr

	// 按 epoch 收集所有 shard 的真实反馈。
	if _, ok := rthm.springEpochFeedback[b.Epoch]; !ok {
		rthm.springEpochFeedback[b.Epoch] = make(map[uint64]SpringBlockStat)
	}
	rthm.springEpochFeedback[b.Epoch][b.SenderShardID] = stat

	// 收齐一个 epoch 的所有 shard 反馈后，计算一次 reward。
	if len(rthm.springEpochFeedback[b.Epoch]) == params.ShardNum && !rthm.springRewardedEpoch[b.Epoch] {
		record, ok := rthm.springBuildFeedbackRewardRecord(b.Epoch, rthm.springEpochFeedback[b.Epoch])
		if ok {
			rthm.springRewardCount++
			rthm.springCrossRateSum += record.CrossRate
			rthm.springRewardSum += record.Reward
			rthm.springNormVarSum += record.NormalizedLoadVariance

			record.RewardedEpochCount = rthm.springRewardCount
			record.RunningAvgCrossRate = rthm.springCrossRateSum / float64(rthm.springRewardCount)
			record.RunningAvgReward = rthm.springRewardSum / float64(rthm.springRewardCount)
			record.RunningAvgNormVar = rthm.springNormVarSum / float64(rthm.springRewardCount)

			rthm.springAppendFeedbackRecord(record)

			rthm.sl.Slog.Printf(
				"[SPRING ONLINE REWARD] epoch=%d mode=%s total=%d effective=%.1f cross=%.1f inner=%d relay1=%d relay2=%d crossRate=%.6f runCross=%.6f rCSTR=%.6f rWLB=%.6f absDiff=%.6f normVar=%.6f commCost=%.6f maxShare=%.6f hotspot=%.6f reward=%.6f runReward=%.6f lambda=%.3f beta=%.3f loads=%v effectiveLoads=%v\n",
				record.Epoch,
				record.RewardMode,
				record.TotalTx,
				record.EffectiveTx,
				record.CrossTx,
				record.TotalInner,
				record.TotalRelay1,
				record.TotalRelay2,
				record.CrossRate,
				record.RunningAvgCrossRate,
				record.RCSTR,
				record.RWLB,
				record.AbsLoadDiff,
				record.NormalizedLoadVariance,
				record.CommunicationCost,
				record.MaxLoadShare,
				record.HotspotPenalty,
				record.Reward,
				record.RunningAvgReward,
				record.Lambda,
				record.Beta,
				record.Loads,
				record.EffectiveLoads,
			)

			if params.SpringMode == 2 && params.SpringOnlineTrain == 1 {
				onlineInput, updateOk := rthm.springBuildOnlineUpdateInputLocked(record)
				if updateOk {
					rthm.springWriteOnlineUpdateInput(onlineInput)

					// 在线训练阶段：生成 online_update 文件后，调用 Python 更新 PPO。
					// 不做显式负载保护，不改 PPO 动作，只用 reward 训练模型。
					rthm.springCallPythonOnlineUpdate(onlineInput)
				}
			} else if params.SpringMode == 2 && params.SpringOnlineTrain == 0 {
				rthm.sl.Slog.Printf(
					"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=SpringOnlineTrain_disabled reward=%.6f crossRate=%.6f normVar=%.6f\n",
					record.Epoch,
					record.Reward,
					record.CrossRate,
					record.NormalizedLoadVariance,
				)
			}
		} else {
			rthm.sl.Slog.Printf(
				"[SPRING ONLINE REWARD SKIP] epoch=%d reason=empty_or_relay2_only\n",
				b.Epoch,
			)
		}
		rthm.springRewardedEpoch[b.Epoch] = true

		// 简单清理旧 epoch，避免长时间运行 map 越来越大。
		for oldEpoch := range rthm.springEpochFeedback {
			if oldEpoch < b.Epoch-10 {
				delete(rthm.springEpochFeedback, oldEpoch)
			}
		}
		for oldEpoch := range rthm.springRewardedEpoch {
			if oldEpoch < b.Epoch-10 {
				delete(rthm.springRewardedEpoch, oldEpoch)
			}
		}
	}
}

func (rthm *RelayCommitteeModule) springGetStat(sid uint64, back int) SpringBlockStat {
	arr := rthm.springStats[sid]
	idx := len(arr) - 1 - back

	if idx < 0 {
		return SpringBlockStat{}
	}

	return arr[idx]
}

func springNormalizeStateCount(v int) float64 {
	return springNormalizeStateValue(float64(v))
}

func springNormalizeStateValue(v float64) float64 {
	denom := float64(params.MaxBlockSize_global)
	if params.TxBatchSize > params.MaxBlockSize_global {
		denom = float64(params.TxBatchSize)
	}
	if denom <= 0 {
		denom = 100.0
	}

	// Use a smooth log scale so relay-heavy blocks do not all collapse to 1.
	x := math.Log1p(v) / math.Log1p(denom*4.0)
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func springExtractRelatedShardFromState(state []float64) (bool, int) {
	base := 10 * params.ShardNum
	if len(state) < base+params.ShardNum {
		return false, -1
	}

	for sid := 0; sid < params.ShardNum; sid++ {
		if state[base+sid] > 0.5 {
			return true, sid
		}
	}
	return false, -1
}

func springMeanInt(values []int) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0
	for _, v := range values {
		sum += v
	}
	return float64(sum) / float64(len(values))
}

func springAddressFlagFromRelatedCount(relatedCount int) float64 {
	// The dataset does not provide a stable EOA/contract flag. As a light-weight
	// proxy, mark highly connected addresses inside the current block as 1.
	if relatedCount >= 4 {
		return 1.0
	}
	return 0.0
}

func springLocalActionReward(
	chosenShard int,
	relatedKnown bool,
	relatedShard int,
	relatedWeight float64,
	shardLoadBefore int,
	loadMeanBefore float64,
) (float64, float64) {
	reward := 0.0

	if relatedKnown {
		if chosenShard == relatedShard {
			reward += 1.2 * relatedWeight
		} else {
			reward -= 1.0 * relatedWeight
		}
	}

	loadPenalty := 0.0
	if loadMeanBefore > 1e-9 {
		loadPenalty = float64(shardLoadBefore) / loadMeanBefore
		if loadPenalty > 1.0 {
			overload := loadPenalty - 1.0
			reward -= 0.70 * overload
			if loadPenalty > 1.25 {
				reward -= 0.35 * (loadPenalty - 1.25)
			}
		} else {
			reward += 0.08 * (1.0 - loadPenalty)
		}
	}

	return reward, loadPenalty
}

func (rthm *RelayCommitteeModule) springBuildSenderPos(
	addr string,
	fallbackRelated utils.Address,
	batchPlacement map[string]uint64,
	batchRelated map[string]map[string]bool,
) ([]float64, string, bool, int, float64, bool, int) {
	senderPos := make([]float64, params.ShardNum)
	relatedSet := make(map[string]bool)

	if peers, ok := batchRelated[addr]; ok {
		for peer := range peers {
			relatedSet[peer] = true
		}
	}

	if fallbackRelated != "" && (params.SpringSenderPosMode != 1 || len(relatedSet) == 0) {
		relatedSet[string(fallbackRelated)] = true
	}

	relatedKeys := make([]string, 0, len(relatedSet))
	for peer := range relatedSet {
		relatedKeys = append(relatedKeys, peer)
	}
	sort.Strings(relatedKeys)

	knownCount := 0
	relatedInCurrentBatch := false
	shardCounts := make([]int, params.ShardNum)

	for _, peer := range relatedKeys {
		if batchPlacement != nil {
			if sid, ok := batchPlacement[peer]; ok && int(sid) < params.ShardNum {
				shardCounts[int(sid)]++
				knownCount++
				relatedInCurrentBatch = true
				continue
			}
		}

		if sid, ok := rthm.springAddrShard[peer]; ok && int(sid) < params.ShardNum {
			shardCounts[int(sid)]++
			knownCount++
		}
	}

	majorShard := -1
	majorCount := 0
	for sid, count := range shardCounts {
		if count > majorCount {
			majorCount = count
			majorShard = sid
		}
	}

	relatedKnown := knownCount > 0 && majorShard >= 0
	relatedWeight := 0.0
	if knownCount > 0 {
		for sid, count := range shardCounts {
			senderPos[sid] = float64(count) / float64(knownCount)
		}
		relatedWeight = float64(majorCount) / float64(knownCount)
	}

	summaryKeys := relatedKeys
	if len(summaryKeys) > 8 {
		summaryKeys = summaryKeys[:8]
	}
	relatedSummary := strings.Join(summaryKeys, ",")
	if len(relatedKeys) > len(summaryKeys) {
		relatedSummary = relatedSummary + ",+" + intToString(len(relatedKeys)-len(summaryKeys))
	}
	return senderPos, relatedSummary, relatedKnown, majorShard, relatedWeight, relatedInCurrentBatch, len(relatedKeys)
}

func (rthm *RelayCommitteeModule) springBuildStateFromSenderPos(
	senderPos []float64,
	flag float64,
	iotFeatures ...[]float64,
) []float64 {
	extraFeatures := []float64(nil)
	if len(iotFeatures) > 0 {
		extraFeatures = iotFeatures[0]
	}
	featureDim := len(extraFeatures)
	if params.SpringIOTMode == 1 && springConfiguredIOTFeatureDim() > 0 {
		featureDim = springConfiguredIOTFeatureDim()
	}
	currentLoadDim := 0
	if params.SpringIOTMode == 1 && featureDim > 0 {
		currentLoadDim = params.ShardNum
	}

	state := make([]float64, 0, 11*params.ShardNum+1+currentLoadDim+featureDim)

	for back := 4; back >= 0; back-- {
		for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
			st := rthm.springGetStat(sid, back)
			state = append(state, springNormalizeStateCount(st.NumTx))
		}
	}

	for back := 4; back >= 0; back-- {
		for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
			st := rthm.springGetStat(sid, back)
			state = append(state, springNormalizeStateValue(st.CrossTx))
		}
	}

	for sid := 0; sid < params.ShardNum; sid++ {
		v := 0.0
		if sid < len(senderPos) {
			v = senderPos[sid]
		}
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		state = append(state, v)
	}

	if flag != 0 {
		flag = 1
	}
	state = append(state, flag)

	if currentLoadDim > 0 {
		totalLoad := 0
		for _, load := range rthm.springShardLoad {
			totalLoad += load
		}
		for sid := 0; sid < params.ShardNum; sid++ {
			value := 0.0
			if totalLoad > 0 && sid < len(rthm.springShardLoad) {
				value = float64(rthm.springShardLoad[sid]) / float64(totalLoad)
			}
			state = append(state, springClamp(value, 0.0, 1.0))
		}
	}

	for idx := 0; idx < featureDim; idx++ {
		feature := 0.0
		if idx < len(extraFeatures) {
			feature = extraFeatures[idx]
		}
		state = append(state, springClamp(feature, 0.0, 1.0))
	}

	return state
}

func (rthm *RelayCommitteeModule) springBuildState(
	related utils.Address,
	batchPlacement map[string]uint64,
) []float64 {
	state := make([]float64, 0, 11*params.ShardNum+1)

	// 最近 5 个块的总交易数 num_tx。
	// 顺序保持为：更旧 -> 更新，和论文中滑动窗口时序信息一致。
	for back := 4; back >= 0; back-- {
		for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
			st := rthm.springGetStat(sid, back)
			state = append(state, springNormalizeStateCount(st.NumTx))
		}
	}

	// 最近 5 个块的跨片交易数 cross_tx。
	for back := 4; back >= 0; back-- {
		for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
			st := rthm.springGetStat(sid, back)
			state = append(state, springNormalizeStateValue(st.CrossTx))
		}
	}

	// sender_pos：
	// 优先使用当前 TxBatch 内已经产生的 placement，
	// 再回退到全局 address -> shard 放置表。
	relatedSid := -1
	relatedKey := string(related)

	if relatedKey != "" {
		if batchPlacement != nil {
			if sid, ok := batchPlacement[relatedKey]; ok {
				relatedSid = int(sid)
			}
		}

		if relatedSid < 0 {
			if sid, ok := rthm.springAddrShard[relatedKey]; ok {
				relatedSid = int(sid)
			}
		}
	}

	for sid := 0; sid < params.ShardNum; sid++ {
		if sid == relatedSid {
			state = append(state, 1.0)
		} else {
			state = append(state, 0.0)
		}
	}

	// flag F：
	// 当前 selectedTxs_300K.csv 没有稳定的合约账户/普通账户标签，
	// 先统一设为 0，代表未知或默认普通账户。
	state = append(state, 0.0)

	return state
}
