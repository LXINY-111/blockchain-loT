package committee

// 真实账户场景适配：沿用 203 维状态与原 PPO，不生成固定设备锚点。
import (
	"blockEmulator/core"
	"blockEmulator/params"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const springRealAccountLoadVersion = "real_accounts_execution_v2_1"

var springRealAddress = regexp.MustCompile(`^0x[0-9a-f]{40}$`)
var springRealAmount = regexp.MustCompile(`^[0-9]+$`)

func springRealAccountsEnabled() bool { return params.SpringIOTIdentityMode == 2 }

func springV2IndexOffset(path string) (uint64, error) {
	data, err := os.ReadFile(filepath.Join(filepath.Dir(path), "README_构造说明.md"))
	if err != nil || !strings.Contains(string(data), "<!-- iot-v2-metadata:start -->") {
		// 兼容生成器尚未精简的目录，与 Python 的元数据回退一致。
		if err != nil && !os.IsNotExist(err) {
			return 0, err
		}
		data, err = os.ReadFile(filepath.Join(filepath.Dir(path), "dataset_summary.json"))
		if err != nil {
			return 0, err
		}
		var summary map[string]json.RawMessage
		if err = json.Unmarshal(data, &summary); err != nil {
			return 0, err
		}
		var offset uint64
		err = json.Unmarshal(summary["index_offset"], &offset)
		return offset, err
	}
	// Windows 文本可能是 CRLF；Python read_text 会自动归一化换行。
	document := strings.ReplaceAll(string(data), "\r\n", "\n")
	parts := strings.SplitN(document, "<!-- iot-v2-metadata:start -->", 2)
	if len(parts) != 2 {
		return 0, fmt.Errorf("missing v2 construction metadata")
	}
	parts = strings.SplitN(parts[1], "<!-- iot-v2-metadata:end -->", 2)
	if len(parts) != 2 {
		return 0, fmt.Errorf("incomplete v2 construction metadata")
	}
	block := strings.TrimSpace(parts[0])
	if !strings.HasPrefix(block, "```json\n") || !strings.HasSuffix(block, "```") {
		return 0, fmt.Errorf("invalid v2 metadata block")
	}
	block = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(block, "```json\n"), "```"))
	var metadata map[string]json.RawMessage
	if err := json.Unmarshal([]byte(block), &metadata); err != nil {
		return 0, err
	}
	var summary map[string]json.RawMessage
	if err := json.Unmarshal(metadata["summary"], &summary); err != nil {
		return 0, err
	}
	var offset uint64
	err = json.Unmarshal(summary["index_offset"], &offset)
	return offset, err
}

func springRealFeatureFromRow(row map[string]string) (SpringIOTTxFeature, error) {
	if !springRealAddress.MatchString(row["from_address"]) || !springRealAddress.MatchString(row["to_address"]) || row["from_address"] == row["to_address"] {
		return SpringIOTTxFeature{}, fmt.Errorf("invalid v2 account addresses")
	}
	for _, name := range []string{"distance_m", "link_quality_forward", "link_quality_reverse", "flowDuration", "srcNumPackets", "dstNumPackets", "srcPayloadSize", "dstPayloadSize"} {
		value, err := strconv.ParseFloat(row[name], 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || (strings.HasPrefix(name, "link_quality") && value > 1) {
			return SpringIOTTxFeature{}, fmt.Errorf("invalid v2 field %s=%q", name, row[name])
		}
	}
	if row["from_actor_type"] != "device" || row["to_actor_type"] != "device" {
		return SpringIOTTxFeature{}, fmt.Errorf("v2 requires logical device accounts")
	}
	adapt := func(reverse bool) SpringIOTTxFeature {
		legacy := make(map[string]string, len(row)+4)
		for key, value := range row {
			legacy[key] = value
		}
		legacy["anchor_count"], legacy["anchor_weights"] = "1", "1"
		legacy["distance"] = row["distance_m"]
		legacy["link_quality"], legacy["anchor_types"] = row["link_quality_forward"], row["to_actor_type"]
		if reverse {
			legacy["link_quality"], legacy["anchor_types"] = row["link_quality_reverse"], row["from_actor_type"]
		}
		feature, _ := springIOTFeatureFromRow(legacy, 0)
		return feature
	}
	forward, reverse := adapt(false), adapt(true)
	forward.RealAccounts = true
	forward.StateObject = strings.TrimPrefix(row["from_address"], "0x")
	forward.AnchorObject = strings.TrimPrefix(row["to_address"], "0x")
	forward.AnchorObjects, forward.AnchorWeights = []string{forward.AnchorObject}, []float64{1}
	forward.RecipientFeatures = reverse.Features
	forward.RecipientCommunicationCostWeight = reverse.CommunicationCostWeight
	return forward, nil
}

type springAccountScene struct {
	Features []float64
	Cost     float64
	Count    int
	Peers    map[string]bool
}

func (rthm *RelayCommitteeModule) springRealAccountScenes(txs []*core.Transaction) map[string]*springAccountScene {
	result := make(map[string]*springAccountScene)
	add := func(address, peer string, features []float64, cost float64) {
		scene := result[address]
		if scene == nil {
			scene = &springAccountScene{Features: make([]float64, 10), Peers: make(map[string]bool)}
			result[address] = scene
		}
		for i, value := range features {
			if i == 2 {
				if scene.Count == 0 || value < scene.Features[i] {
					scene.Features[i] = value
				}
			} else {
				scene.Features[i] += value
			}
		}
		scene.Count++
		scene.Cost += cost
		scene.Peers[peer] = true
	}
	for _, tx := range txs {
		feature, ok := rthm.springIOTByTxIndex[tx.Nonce]
		if !ok || !feature.RealAccounts {
			panic("missing real-account scene")
		}
		add(string(tx.Sender), string(tx.Recipient), feature.Features, feature.CommunicationCostWeight)
		add(string(tx.Recipient), string(tx.Sender), feature.RecipientFeatures, feature.RecipientCommunicationCostWeight)
	}
	for _, scene := range result {
		for i := range scene.Features {
			if i != 2 {
				scene.Features[i] /= float64(scene.Count)
			}
		}
		scene.Features[0] = math.Min(1, float64(len(scene.Peers))/4)
		scene.Cost /= float64(scene.Count)
	}
	return result
}

// 此分支只改变新数据的身份与输入特征，不改变旧多锚点分支。
// 显式 v3 另增加逐分片成本输入和对应局部奖励；legacy 保持原 v2 语义。
// 模式 1 先放置所有发送方，再放置接收方；后续动作立即看到已有放置。
func (rthm *RelayCommitteeModule) springPrepareRealAccountPlacement(txs []*core.Transaction, placement map[string]uint64) {
	if params.SpringMode == 0 {
		return
	}
	if params.SpringSenderPosMode != 0 && params.SpringSenderPosMode != 1 {
		panic("v2 supports sender_pos_mode 0 or 1")
	}
	related := springBuildBatchRelatedMap(txs)
	scenes := rthm.springRealAccountScenes(txs)
	// 原始交易成本按两端关联累计，保持最终指标的同一个正向成本定义。
	costRelations := make(map[string]map[string]float64)
	addCost := func(a, b string, cost float64) {
		if a == b {
			return
		}
		if costRelations[a] == nil {
			costRelations[a] = make(map[string]float64)
		}
		costRelations[a][b] += cost
	}
	if params.SpringMechanismVersion == "v3" {
		for _, tx := range txs {
			cost := rthm.springIOTByTxIndex[tx.Nonce].CommunicationCostWeight
			addCost(string(tx.Sender), string(tx.Recipient), cost)
			addCost(string(tx.Recipient), string(tx.Sender), cost)
		}
	}
	// 加权基线按当前批次出现次数累计真实关系，只构造一次。
	weights := make(map[string]map[string]float64)
	addWeight := func(a, b string) {
		if weights[a] == nil {
			weights[a] = make(map[string]float64)
		}
		weights[a][b]++
	}
	if params.SpringMode == 6 {
		for _, tx := range txs {
			addWeight(string(tx.Recipient), string(tx.Sender))
			if params.SpringSenderPosMode != 1 {
				addWeight(string(tx.Sender), string(tx.Recipient))
			}
		}
	}
	rthm.springTxBatchSeq++
	batchID := rthm.springTxBatchSeq
	actions := make([]SpringTrainAction, 0)
	place := func(address, peer string) {
		scene := scenes[address]
		costVector := make([]float64, params.ShardNum)
		if params.SpringMechanismVersion == "v3" {
			keys := make([]string, 0, len(costRelations[address]))
			for p := range costRelations[address] {
				keys = append(keys, p)
			}
			sort.Strings(keys)
			total := 0.0
			for _, p := range keys {
				total += costRelations[address][p]
			}
			if total > 1e-8 {
				for _, p := range keys {
					if sid, ok := rthm.springAddrShard[p]; ok {
						costVector[sid] += costRelations[address][p] / total
					}
				}
			}
		}
		features := []float64(nil)
		if springIOTEnabled() {
			features = append([]float64(nil), scene.Features...)
			if params.SpringMechanismVersion == "v3" {
				features = append(features, costVector...)
			}
		}
		switch params.SpringMode {
		case 2:
			if action, ok := rthm.springPlaceAddressPPOSequential(address, peer, placement, related, features, append([]float64{scene.Cost}, costVector...)); ok {
				actions = append(actions, action)
			}
		case 7:
			rthm.springPlaceAddressCandidateOnlySequential(batchID, address, peer, placement, related, features)
		case 1:
			rthm.springEnsurePlacedWithBatchRelated(address, peer, placement, related)
		case 5:
			rthm.springEnsurePlacedAnchorOnlyWithBatchRelated(address, peer, placement, related)
		case 6:
			rthm.springEnsurePlacedNSShardAdaptedWithBatchWeights(address, peer, placement, weights)
		default:
			rthm.springEnsurePlaced(address, peer, placement)
		}
	}
	if params.SpringSenderPosMode == 1 {
		for _, tx := range txs {
			place(string(tx.Sender), "")
		}
		for _, tx := range txs {
			place(string(tx.Recipient), string(tx.Sender))
		}
	} else {
		for _, tx := range txs {
			place(string(tx.Sender), string(tx.Recipient))
			place(string(tx.Recipient), string(tx.Sender))
		}
	}
	if params.SpringMode == 2 && params.SpringOnlineTrain == 1 && len(actions) > 0 {
		rthm.springEnqueueTrainActionsLocked(batchID, actions, txs[0].Nonce, txs[len(txs)-1].Nonce, len(txs), rthm.nowDataNum >= rthm.dataTotalNum)
	}
	rthm.springFillTouchedPlacement(txs, placement)
}

// 与 Python iot_dense_action_reward 的系数、裁剪和低负载奖励逐项一致。
func springIOTDenseActionReward(chosen int, senderPos []float64, cost float64, load int, mean float64) (float64, float64) {
	mass, total := 0.0, 0.0
	for _, value := range senderPos {
		total += value
	}
	if chosen >= 0 && chosen < len(senderPos) {
		mass = springClamp(senderPos[chosen], 0, 1)
	}
	cross := 0.0
	if total > 1e-8 {
		cross = 1 - mass
	}
	reward := 1.4*mass - 0.8*cross - 0.45*springClamp(cost, 0, 1)*cross
	pressure := 0.0
	if mean > 1e-9 {
		pressure = float64(load) / mean
		if pressure > 1 {
			reward -= 0.55 * (pressure - 1)
			if pressure > 1.5 {
				reward -= 0.35 * (pressure - 1.5)
			}
		} else {
			gap := 1 - pressure
			reward += 0.10*gap + math.Min(0.18, 0.25*gap)
		}
	}
	return reward, pressure
}
