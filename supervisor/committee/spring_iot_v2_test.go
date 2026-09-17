package committee

import (
	"blockEmulator/core"
	"blockEmulator/params"
	"blockEmulator/supervisor/supervisor_log"
	"bytes"
	"encoding/json"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestV3StateRemovesOnlyAddressFlag(t *testing.T) {
	oldShards, oldIOT, oldDim := params.ShardNum, params.SpringIOTMode, params.SpringIOTFeatureDim
	oldVersion, oldFeature := params.SpringMechanismVersion, params.SpringFeatureMode
	defer func() {
		params.ShardNum, params.SpringIOTMode, params.SpringIOTFeatureDim = oldShards, oldIOT, oldDim
		params.SpringMechanismVersion, params.SpringFeatureMode = oldVersion, oldFeature
	}()
	params.ShardNum, params.SpringIOTMode, params.SpringIOTFeatureDim = 16, 1, 10
	params.SpringFeatureMode = "full"
	r := &RelayCommitteeModule{springShardLoad: make([]int, 16)}
	features := []float64{.25, .2, .1, .8, .3, 1, 0, 0, 0, .75}
	costs := make([]float64, 16)
	costs[0], costs[1] = .4, .6
	params.SpringMechanismVersion = "legacy"
	old := r.springBuildStateFromSenderPos(make([]float64, 16), 1, features)
	params.SpringMechanismVersion = "v3"
	extra := append(append([]float64(nil), features...), costs...)
	current := r.springBuildStateFromSenderPos(make([]float64, 16), 1, extra)
	zeroFlag := r.springBuildStateFromSenderPos(make([]float64, 16), 0, extra)
	want := append(append([]float64(nil), old[:176]...), old[177:]...)
	want = append(want, costs...)
	if len(old) != 203 || len(current) != 218 || !reflect.DeepEqual(current, want) || !reflect.DeepEqual(current, zeroFlag) {
		t.Fatal("state layout changed beyond removing address flag")
	}
	assertClose(t, current[201], .75) // 控制协议占比保留在第 202 维。
}

func TestV3SceneCostCountsOnlyFirstCrossPhase(t *testing.T) {
	oldVersion, oldCost := params.SpringMechanismVersion, params.SpringSceneCostMode
	defer func() { params.SpringMechanismVersion, params.SpringSceneCostMode = oldVersion, oldCost }()
	r := &RelayCommitteeModule{springIOTByTxIndex: map[uint64]SpringIOTTxFeature{
		1: {CommunicationCostWeight: .2}, 2: {CommunicationCostWeight: .8},
	}}
	inner, relay := &core.Transaction{Nonce: 1}, &core.Transaction{Nonce: 2}
	for _, tc := range []struct {
		version, cost string
		want          float64
	}{
		{"legacy", "legacy", 1}, {"v3", "cross", .8}, {"v3", "off", 0},
	} {
		params.SpringMechanismVersion, params.SpringSceneCostMode = tc.version, tc.cost
		// 同片一次、Relay1 一次；重复记录和 Relay2 不重复贡献。
		sum, count := r.springIOTCommunicationCostForTxGroups(
			[]*core.Transaction{inner}, []*core.Transaction{relay, relay}, []*core.Transaction{relay})
		assertClose(t, sum, tc.want)
		if count != 2 {
			t.Fatalf("%s/%s count=%d", tc.version, tc.cost, count)
		}
	}
}

// Python 独立计算期望值，Go 运行真实加载器与顺序放置入口后比较。
// 不把异步链上历史假设为离线批次历史；本测试给两端相同的完整阶段反馈。
func TestIOTV2CrossLanguage(t *testing.T) {
	path := os.Getenv("IOT_V2_PARITY_INPUT")
	if path == "" {
		t.Skip("run spring_lite/check_iot_v2_alignment.py for real dataset parity")
	}
	var fixture struct {
		Model, PythonExecutable, PythonDirectory     string
		InferenceItems                               []SpringBatchInferItem
		InferenceExpected                            []SpringBatchInferResult
		ScenePath                                    string
		MechanismVersion, FeatureMode, SceneCostMode string
		StartTx, Count, SenderPosMode, BatchSize     int
		Batches                                      []struct {
			Rows             [][]string
			Forward, Reverse [][]float64
			Costs            []float64
			Reward           float64
			Actions          []struct {
				Address, Related  string
				State             []float64
				ActionMask        []int
				Shard             int
				LocalReward, Cost float64
				CostByShard       []float64
			}
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	oldShards, oldMode, oldIOT, oldID, oldPos, oldBatch, oldBlock := params.ShardNum, params.SpringMode, params.SpringIOTMode, params.SpringIOTIdentityMode, params.SpringSenderPosMode, params.TxBatchSize, params.MaxBlockSize_global
	oldMechanism, oldFeature, oldCost := params.SpringMechanismVersion, params.SpringFeatureMode, params.SpringSceneCostMode
	params.SpringMechanismVersion, params.SpringFeatureMode, params.SpringSceneCostMode = fixture.MechanismVersion, fixture.FeatureMode, fixture.SceneCostMode
	defer func() {
		params.SpringMechanismVersion, params.SpringFeatureMode, params.SpringSceneCostMode = oldMechanism, oldFeature, oldCost
		params.ShardNum, params.SpringMode, params.SpringIOTMode, params.SpringIOTIdentityMode, params.SpringSenderPosMode, params.TxBatchSize, params.MaxBlockSize_global = oldShards, oldMode, oldIOT, oldID, oldPos, oldBatch, oldBlock
	}()
	params.ShardNum, params.SpringMode, params.SpringIOTMode, params.SpringIOTIdentityMode = 16, 7, 1, 2
	params.SpringSenderPosMode, params.TxBatchSize, params.MaxBlockSize_global = fixture.SenderPosMode, fixture.BatchSize, 1000
	features, err := springLoadIOTSidecarWindow(fixture.ScenePath, fixture.StartTx, fixture.Count)
	if err != nil {
		t.Fatal(err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Chdir(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(cwd)
	r := &RelayCommitteeModule{springIOTByTxIndex: features, springAddrShard: make(map[string]uint64), springShardLoad: make([]int, 16), springStats: make(map[uint64][]SpringBlockStat)}
	for sid := uint64(0); sid < 16; sid++ {
		r.springStats[sid] = make([]SpringBlockStat, 5)
	}
	closeVector := func(got, want []float64) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("vector lengths %d != %d", len(got), len(want))
		}
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-9 {
				t.Fatalf("vector[%d] got %.16g want %.16g", i, got[i], want[i])
			}
		}
	}
	nonce, priorActions := 0, 0
	loads := make([]int, 16)
	for epoch, batch := range fixture.Batches {
		txs := make([]*core.Transaction, 0, len(batch.Rows))
		for i, row := range batch.Rows {
			tx, ok := data2tx(row, uint64(nonce))
			if !ok {
				t.Fatal("valid transaction rejected")
			}
			before := append([]byte(nil), tx.TxHash...)
			if !r.springApplyIOTTxIdentity(tx) || !bytes.Equal(before, tx.TxHash) || tx.Value.String() != row[8] {
				t.Fatal("transaction identity/value/hash changed")
			}
			f := features[uint64(nonce)]
			closeVector(f.Features, batch.Forward[i])
			closeVector(f.RecipientFeatures, batch.Reverse[i])
			assertClose(t, f.CommunicationCostWeight, batch.Costs[i])
			txs = append(txs, tx)
			nonce++
		}
		r.springPreparePlacement(txs)
		contents, err := os.ReadFile("spring_io/decision_records.jsonl")
		if err != nil {
			t.Fatal(err)
		}
		decoder := json.NewDecoder(bytes.NewReader(contents))
		records := []SpringDecisionRecord{}
		for {
			var record SpringDecisionRecord
			err := decoder.Decode(&record)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			records = append(records, record)
		}
		if len(records) != priorActions+len(batch.Actions) {
			t.Fatal("placement action count mismatch")
		}
		for i, expected := range batch.Actions {
			record := records[priorActions+i]
			if record.Address != expected.Address || record.Related != expected.Related || int(record.Shard) != expected.Shard || !reflect.DeepEqual(record.ActionMask, expected.ActionMask) {
				t.Fatalf("action mismatch batch=%d action=%d: got addr=%s related=%s shard=%d mask=%v; want addr=%s related=%s shard=%d mask=%v", epoch, i, record.Address, record.Related, record.Shard, record.ActionMask, expected.Address, expected.Related, expected.Shard, expected.ActionMask)
			}
			closeVector(record.State, expected.State)
			reward, _ := springIOTDenseActionReward(expected.Shard, record.State[160:176], expected.Cost, loads[expected.Shard], springMeanInt(loads))
			if params.SpringMechanismVersion == "v3" {
				reward, _ = springIOTDenseActionReward(expected.Shard, record.State[160:176], 0, loads[expected.Shard], springMeanInt(loads))
				if params.SpringSceneCostMode == "cross" {
					for sid, cost := range expected.CostByShard {
						if sid != expected.Shard {
							reward -= .45 * cost
						}
					}
				}
			}
			assertClose(t, reward, expected.LocalReward)
			loads[expected.Shard]++
		}
		priorActions = len(records)
		stats := make(map[uint64]SpringBlockStat)
		for _, tx := range txs {
			a, b := r.springAddrShard[string(tx.Sender)], r.springAddrShard[string(tx.Recipient)]
			s := stats[a]
			s.NumTx++
			if params.SpringMechanismVersion != "v3" || (params.SpringSceneCostMode == "cross" && a != b) {
				s.CommunicationCostSum += features[tx.Nonce].CommunicationCostWeight
			}
			s.CommunicationCostCount++
			if a == b {
				s.InnerTx++
				s.EffectiveTx++
			} else {
				s.Relay1Tx++
				s.EffectiveTx += 0.5
				s.CrossTx += 0.5
				d := stats[b]
				d.NumTx++
				d.Relay2Tx++
				d.EffectiveTx += 0.5
				d.CrossTx += 0.5
				stats[b] = d
			}
			stats[a] = s
		}
		feedback, ok := r.springBuildFeedbackRewardRecord(epoch, stats)
		if !ok {
			t.Fatal("missing feedback")
		}
		assertClose(t, feedback.Reward, batch.Reward)
		for sid := uint64(0); sid < 16; sid++ {
			r.springStats[sid] = append(r.springStats[sid][1:], stats[sid])
		}
	}
	if nonce != fixture.Count || len(r.springAddrShard) != priorActions {
		t.Fatal("unplaced or pre-seeded real account")
	}
	if fixture.Model != "" {
		// 使用隔离的临时工作目录，避免触碰用户已有运行的 spring_io。
		oldModel := params.SpringModelFile
		params.SpringModelFile = fixture.Model
		defer func() { params.SpringModelFile = oldModel }()
		t.Setenv("SPRING_PYTHON", fixture.PythonExecutable)
		t.Setenv("OMP_NUM_THREADS", "1")
		t.Setenv("MKL_NUM_THREADS", "1")
		// 验证没有本地源码副本时，Go 仍可直接调用项目中的公共 Python 代码。
		t.Setenv("SPRING_PYTHON_DIR", fixture.PythonDirectory)
		r.sl = &supervisor_log.SupervisorLog{Slog: log.New(os.Stderr, "parity: ", 0)}
		defer func() { springInferServerMu.Lock(); defer springInferServerMu.Unlock(); springStopInferServerLocked() }()
		results, _, ok := r.springCallPythonBatch(fixture.InferenceItems)
		if !ok || len(results) != len(fixture.InferenceExpected) {
			t.Fatal("Go/Python inference connection failed")
		}
		for i, result := range results {
			want := fixture.InferenceExpected[i]
			if result.Source != "python_ppo" || result.Shard != want.Shard {
				t.Fatalf("PPO result mismatch %d", i)
			}
			assertClose(t, result.Confidence, want.Confidence)
			assertClose(t, result.LogProb, want.LogProb)
			assertClose(t, result.Value, want.Value)
		}
		t.Logf("verified %d actual PPO inference results via Go stdin/stdout connection", len(results))
	}
	dim := 203
	if fixture.MechanismVersion == "v3" {
		dim += params.ShardNum - 1 // 增加成本向量，删除旧账户 flag。
	}
	t.Logf("verified %d transactions, %d placements, %d-d states, masks, local/global rewards", nonce, priorActions, dim)
}

// 构造说明有无元数据标记时，与 Python 读取同一目录的选择保持一致。
func TestIOTV2MetadataFormats(t *testing.T) {
	directory := t.TempDir()
	readme := filepath.Join(directory, "README_构造说明.md")
	scene := filepath.Join(directory, "transaction_scene.csv")
	for _, document := range []string{"", "# ordinary README\n", "<!-- iot-v2-metadata:start -->\r\n```json\r\n{\"summary\":{\"index_offset\":5}}\r\n```\r\n<!-- iot-v2-metadata:end -->"} {
		if err := os.WriteFile(readme, []byte(document), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, "dataset_summary.json"), []byte(`{"index_offset":5}`), 0644); err != nil {
			t.Fatal(err)
		}
		offset, err := springV2IndexOffset(scene)
		if err != nil || offset != 5 {
			t.Fatalf("metadata offset=%d err=%v", offset, err)
		}
	}
	if err := os.WriteFile(readme, []byte("<!-- iot-v2-metadata:start -->broken"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := springV2IndexOffset(scene); err == nil {
		t.Fatal("incomplete embedded metadata must be rejected")
	}
}
