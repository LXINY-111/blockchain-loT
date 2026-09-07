package committee

import (
	"blockEmulator/core"
	"blockEmulator/params"
	"blockEmulator/utils"
	"encoding/csv"
	"fmt"
	"math"
	"math/big"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func assertClose(t *testing.T, got float64, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %.12f, want %.12f", got, want)
	}
}

func TestSpringIOTFeatureFromRowMatchesOfflineMDP(t *testing.T) {
	row := map[string]string{
		"tx_index":        "9",
		"device_label":    "AugustDoorBell",
		"device_mac":      "E0:76:D0:3F:00:AE",
		"srcIp":           "192.168.1.216",
		"dstIp":           "54.249.82.177",
		"dstPort":         "443",
		"protocol":        "tls",
		"srcNumPackets":   "5",
		"dstNumPackets":   "5",
		"srcPayloadSize":  "100",
		"dstPayloadSize":  "200",
		"flowDuration":    "1000",
		"distance":        "50",
		"link_quality":    "0.2",
		"from_address":    "0x" + "1",
		"to_address":      "0x" + "2",
		"mapped_mote_id":  "1",
		"related_mote_id": "2",
	}

	feature, ok := springIOTFeatureFromRow(row, 3)
	if !ok {
		t.Fatal("expected IoT feature row to parse")
	}

	wantAnchor := "0x" + "1"
	wantState := "0x" + "2"

	if feature.TxIndex != 9 {
		t.Fatalf("TxIndex = %d, want 9", feature.TxIndex)
	}
	if feature.StateObject != wantState {
		t.Fatalf("StateObject = %q, want %q", feature.StateObject, wantState)
	}
	if feature.AnchorObject != wantAnchor {
		t.Fatalf("AnchorObject = %q, want %q", feature.AnchorObject, wantAnchor)
	}
	if len(feature.AnchorObjects) != 1 || feature.AnchorObjects[0] != wantAnchor {
		t.Fatalf("AnchorObjects = %v, want [%q]", feature.AnchorObjects, wantAnchor)
	}
	if len(feature.Features) != 10 {
		t.Fatalf("feature dim = %d, want 10", len(feature.Features))
	}

	assertClose(t, feature.Features[0], 0.25)
	assertClose(t, feature.Features[1], 50.0/60.0)
	assertClose(t, feature.Features[2], 50.0/60.0)
	assertClose(t, feature.Features[3], 0.2)
	assertClose(t, feature.Features[4], math.Log1p(300.0/1000.0)/math.Log1p(10000.0))
	assertClose(t, feature.Features[5], 1.0)
	assertClose(t, feature.Features[6], 0.0)
	assertClose(t, feature.Features[7], 0.0)
	assertClose(t, feature.Features[8], 0.0)
	assertClose(t, feature.Features[9], 0.0)
}

func TestSpringIOTFeatureFromRowParsesMultiAnchorSet(t *testing.T) {
	row := map[string]string{
		"tx_index":         "10",
		"device_label":     "AugustDoorBell",
		"device_mac":       "E0:76:D0:3F:00:AE",
		"protocol":         "dns",
		"srcNumPackets":    "1",
		"dstNumPackets":    "1",
		"srcPayloadSize":   "37",
		"dstPayloadSize":   "242",
		"distance":         "50",
		"link_quality":     "0.2",
		"from_address":     "0xprimary",
		"to_address":       "0xstate",
		"anchor_addresses": "0xprimary;0xpeer;0xprimary",
		"anchor_types":     "device;gateway;cloud_endpoint",
		"anchor_weights":   "0.2;0.3;0.5",
	}

	feature, ok := springIOTFeatureFromRow(row, 0)
	if !ok {
		t.Fatal("expected IoT feature row to parse")
	}

	if len(feature.AnchorObjects) != 2 {
		t.Fatalf("AnchorObjects len = %d, want 2: %v", len(feature.AnchorObjects), feature.AnchorObjects)
	}
	if feature.AnchorObjects[0] != "0xprimary" || feature.AnchorObjects[1] != "0xpeer" {
		t.Fatalf("AnchorObjects = %v, want [0xprimary 0xpeer]", feature.AnchorObjects)
	}
	if len(feature.AnchorWeights) != 2 {
		t.Fatalf("AnchorWeights len = %d, want 2: %v", len(feature.AnchorWeights), feature.AnchorWeights)
	}
	assertClose(t, feature.AnchorWeights[0], 0.7)
	assertClose(t, feature.AnchorWeights[1], 0.3)
	assertClose(t, feature.Features[5], 0.2)
	assertClose(t, feature.Features[6], 0.3)
	assertClose(t, feature.Features[7], 0.5)
}

func TestSpringApplyIOTTxIdentityUsesNonceSidecar(t *testing.T) {
	feature := SpringIOTTxFeature{
		TxIndex:       2,
		StateObject:   "0xstate",
		AnchorObject:  "0xdevice",
		AnchorObjects: []string{"0xdevice", "0xpeer"},
		Features:      make([]float64, 10),
	}
	rthm := &RelayCommitteeModule{
		springIOTByTxIndex: map[uint64]SpringIOTTxFeature{
			2: feature,
		},
	}

	tx := core.NewTransaction("raw_sender", "raw_recipient", big.NewInt(1), 2, time.Now())
	if !rthm.springApplyIOTTxIdentity(tx) {
		t.Fatal("expected IoT identity to be applied")
	}

	if string(tx.Sender) != feature.StateObject {
		t.Fatalf("sender = %q, want %q", tx.Sender, feature.StateObject)
	}
	if string(tx.Recipient) != feature.AnchorObject {
		t.Fatalf("recipient = %q, want %q", tx.Recipient, feature.AnchorObject)
	}
}

func TestSpringIOTIdentityModeLoadsSidecarWhenFeatureModeOff(t *testing.T) {
	oldIOTMode := params.SpringIOTMode
	oldIdentityMode := params.SpringIOTIdentityMode
	oldSidecarFile := params.SpringIOTSidecarFile
	oldDatasetStartTx := params.DatasetStartTx
	defer func() {
		params.SpringIOTMode = oldIOTMode
		params.SpringIOTIdentityMode = oldIdentityMode
		params.SpringIOTSidecarFile = oldSidecarFile
		params.DatasetStartTx = oldDatasetStartTx
	}()
	params.SpringIOTMode = 0
	params.SpringIOTIdentityMode = 1
	params.DatasetStartTx = 0
	params.SpringIOTSidecarFile = filepath.Join(t.TempDir(), "iot_sidecar_small.csv")
	sidecar := strings.Join([]string{
		"tx_index,device_label,device_mac,srcIp,dstIp,dstPort,protocol,srcNumPackets,dstNumPackets,srcPayloadSize,dstPayloadSize,flowDuration,distance,link_quality,from_address,to_address,mapped_mote_id,related_mote_id",
		"0,Camera,AA:BB:CC:DD:EE:01,192.168.1.2,10.0.0.8,443,tls,3,4,120,180,1000,12,0.8,0xanchor,0xstate,1,2",
	}, "\n")
	if err := os.WriteFile(params.SpringIOTSidecarFile, []byte(sidecar), 0644); err != nil {
		t.Fatal(err)
	}

	rthm := NewRelayCommitteeModule(nil, nil, nil, "", 0, 0)
	if len(rthm.springIOTByTxIndex) == 0 {
		t.Fatal("expected IoT sidecar to load when identity mode is enabled")
	}

	tx := core.NewTransaction("raw_sender", "raw_recipient", big.NewInt(1), 0, time.Now())
	if !rthm.springApplyIOTTxIdentity(tx) {
		t.Fatal("expected IoT identity to be available with SpringIOTMode disabled")
	}
	feature := rthm.springIOTByTxIndex[0]
	if string(tx.Sender) != feature.StateObject {
		t.Fatalf("sender = %q, want sidecar state object %q", tx.Sender, feature.StateObject)
	}
	if string(tx.Recipient) != feature.AnchorObject {
		t.Fatalf("recipient = %q, want sidecar anchor object %q", tx.Recipient, feature.AnchorObject)
	}
}

func TestSpringLoadIOTSidecarWindowRemapsSourceIndexToLocalNonce(t *testing.T) {
	sidecarPath := filepath.Join(t.TempDir(), "iot_sidecar_window.csv")
	header := "tx_index,device_label,device_mac,srcIp,dstIp,dstPort,protocol,srcNumPackets,dstNumPackets,srcPayloadSize,dstPayloadSize,flowDuration,distance,link_quality,from_address,to_address,mapped_mote_id,related_mote_id"
	rows := []string{header}
	for idx := 0; idx < 4; idx++ {
		rows = append(rows, strings.Join([]string{
			strconv.Itoa(idx),
			"Camera",
			"AA:BB:CC:DD:EE:01",
			"192.168.1.2",
			"10.0.0.8",
			"443",
			"tls",
			"3",
			"4",
			"120",
			"180",
			"1000",
			"12",
			"0.8",
			fmt.Sprintf("0xanchor%d", idx),
			fmt.Sprintf("0xstate%d", idx),
			"1",
			"2",
		}, ","))
	}
	if err := os.WriteFile(sidecarPath, []byte(strings.Join(rows, "\n")), 0644); err != nil {
		t.Fatal(err)
	}

	features, err := springLoadIOTSidecarWindow(sidecarPath, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(features) != 2 {
		t.Fatalf("features len = %d, want 2", len(features))
	}
	if features[0].TxIndex != 0 || features[0].StateObject != "0xstate2" {
		t.Fatalf("local feature 0 = %+v, want source row 2 remapped to nonce 0", features[0])
	}
	if features[1].TxIndex != 1 || features[1].StateObject != "0xstate3" {
		t.Fatalf("local feature 1 = %+v, want source row 3 remapped to nonce 1", features[1])
	}
}

func TestSkipValidTransactionsCountsFilteredTransactionsAndKeepsLocalNonce(t *testing.T) {
	makeRow := func(idx int) string {
		return strings.Join([]string{
			"",
			"",
			"",
			"0x" + strings.Repeat(strconv.Itoa(idx+1), 40),
			"0x" + strings.Repeat(string(rune('a'+idx)), 40),
			"",
			"0",
			"0",
			"1",
		}, ",")
	}
	raw := strings.Join([]string{
		strings.Join([]string{"", "", "", "short", "short", "", "1", "0", "1"}, ","),
		makeRow(0),
		makeRow(1),
		makeRow(2),
	}, "\n")
	reader := csv.NewReader(strings.NewReader(raw))
	skipped, err := skipValidTransactions(reader, 2)
	if err != nil {
		t.Fatal(err)
	}
	if skipped != 2 {
		t.Fatalf("skipped = %d, want 2", skipped)
	}
	next, err := reader.Read()
	if err != nil {
		t.Fatal(err)
	}
	tx, ok := data2tx(next, 0)
	if !ok {
		t.Fatal("expected first transaction after the window offset to be valid")
	}
	if tx.Nonce != 0 {
		t.Fatalf("nonce = %d, want local nonce 0", tx.Nonce)
	}
}

func TestSpringBuildStateFromSenderPosAppendsIOTFeatures(t *testing.T) {
	oldShardNum := params.ShardNum
	defer func() { params.ShardNum = oldShardNum }()
	params.ShardNum = 4

	rthm := &RelayCommitteeModule{
		springStats: map[uint64][]SpringBlockStat{
			0: {},
			1: {},
			2: {},
			3: {},
		},
	}

	oldIOTMode := params.SpringIOTMode
	oldIOTFeatureDim := params.SpringIOTFeatureDim
	defer func() {
		params.SpringIOTMode = oldIOTMode
		params.SpringIOTFeatureDim = oldIOTFeatureDim
	}()
	params.SpringIOTMode = 1
	params.SpringIOTFeatureDim = 10

	iotFeatures := make([]float64, 10)
	for idx := range iotFeatures {
		iotFeatures[idx] = float64(idx+1) / 100.0
	}
	state := rthm.springBuildStateFromSenderPos(
		[]float64{0.25, 0.25, 0.25, 0.25},
		1.0,
		iotFeatures,
	)

	if len(state) != 11*params.ShardNum+1+params.ShardNum+10 {
		t.Fatalf("state dim = %d, want %d", len(state), 11*params.ShardNum+1+params.ShardNum+10)
	}
	for idx, want := range iotFeatures {
		assertClose(t, state[len(state)-len(iotFeatures)+idx], want)
	}
}

func TestSpringSeedIOTAnchorShardsAcceptsDeviceAccount(t *testing.T) {
	oldShardNum := params.ShardNum
	defer func() { params.ShardNum = oldShardNum }()
	params.ShardNum = 4

	rthm := &RelayCommitteeModule{
		springAddrShard: make(map[string]uint64),
		springIOTByTxIndex: map[uint64]SpringIOTTxFeature{
			0: {
				TxIndex:       0,
				StateObject:   "0xstateobject",
				AnchorObject:  "0x" + "aabbcc",
				AnchorObjects: []string{"0x" + "aabbcc", "0x" + "ddeeff"},
			},
		},
	}
	anchor := "0x" + "aabbcc"
	peerAnchor := "0x" + "ddeeff"
	tx := core.NewTransaction("0xstateobject", anchor, big.NewInt(1), 0, time.Now())
	batchPlacement := make(map[string]uint64)

	rthm.springSeedIOTAnchorShards([]*core.Transaction{tx}, batchPlacement)

	sid, ok := rthm.springAddrShard[anchor]
	if !ok {
		t.Fatalf("anchor %q was not seeded", anchor)
	}
	if sid >= uint64(params.ShardNum) {
		t.Fatalf("seeded shard = %d, want < %d", sid, params.ShardNum)
	}
	if batchPlacement[anchor] != sid {
		t.Fatalf("batch placement = %d, want %d", batchPlacement[anchor], sid)
	}
	if _, ok := rthm.springAddrShard[peerAnchor]; !ok {
		t.Fatalf("multi-anchor peer %q was not seeded", peerAnchor)
	}
}

func TestSpringBuildIOTBatchRelatedMapUsesAllSidecarAnchors(t *testing.T) {
	rthm := &RelayCommitteeModule{
		springIOTByTxIndex: map[uint64]SpringIOTTxFeature{
			7: {
				TxIndex:       7,
				StateObject:   "0xstate",
				AnchorObject:  "0xprimary",
				AnchorObjects: []string{"0xprimary", "0xpeer"},
			},
		},
	}
	tx := core.NewTransaction("0xstate", "0xprimary", big.NewInt(1), 7, time.Now())

	related := rthm.springBuildIOTBatchRelatedMap([]*core.Transaction{tx})

	peers := related["0xstate"]
	if !peers["0xprimary"] || !peers["0xpeer"] {
		t.Fatalf("related anchors = %v, want primary and peer", peers)
	}
}

func TestSpringBuildIOTBatchRelatedWeightMapAggregatesAnchorWeights(t *testing.T) {
	rthm := &RelayCommitteeModule{
		springIOTByTxIndex: map[uint64]SpringIOTTxFeature{
			7: {
				TxIndex:       7,
				StateObject:   "0xstate",
				AnchorObject:  "0xprimary",
				AnchorObjects: []string{"0xprimary", "0xpeer"},
				AnchorWeights: []float64{0.7, 0.3},
			},
			8: {
				TxIndex:       8,
				StateObject:   "0xstate",
				AnchorObject:  "0xprimary",
				AnchorObjects: []string{"0xprimary", "0xpeer"},
				AnchorWeights: []float64{0.2, 0.8},
			},
		},
	}
	tx1 := core.NewTransaction("0xstate", "0xprimary", big.NewInt(1), 7, time.Now())
	tx2 := core.NewTransaction("0xstate", "0xprimary", big.NewInt(1), 8, time.Now())

	related := rthm.springBuildIOTBatchRelatedWeightMap([]*core.Transaction{tx1, tx2})

	weights := related["0xstate"]
	assertClose(t, weights["0xprimary"], 0.9)
	assertClose(t, weights["0xpeer"], 1.1)
}

func TestSpringIOTCommunicationCostForTxGroupsUsesUniqueNonce(t *testing.T) {
	rthm := &RelayCommitteeModule{
		springIOTByTxIndex: map[uint64]SpringIOTTxFeature{
			1: {CommunicationCostWeight: 0.25},
			2: {CommunicationCostWeight: 0.75},
		},
	}
	tx1 := core.NewTransaction("0xstate1", "0xanchor1", big.NewInt(1), 1, time.Now())
	tx2 := core.NewTransaction("0xstate2", "0xanchor2", big.NewInt(1), 2, time.Now())

	sum, count := rthm.springIOTCommunicationCostForTxGroups(
		[]*core.Transaction{tx1, tx2},
		[]*core.Transaction{tx1, nil},
	)

	assertClose(t, sum, 1.0)
	if count != 2 {
		t.Fatalf("communication cost count = %d, want 2", count)
	}
}

func TestSpringBuildFeedbackRewardRecordUsesIOTDenseBalancedReward(t *testing.T) {
	oldShardNum := params.ShardNum
	oldIOTMode := params.SpringIOTMode
	oldCSTRWeight := params.SpringIOTCSTRWeight
	oldBalanceWeight := params.SpringIOTBalanceWeight
	oldCommCostWeight := params.SpringIOTCommCostWeight
	oldHotspotWeight := params.SpringIOTHotspotWeight
	oldHotspotThreshold := params.SpringIOTHotspotThreshold
	oldBeta := params.SpringRewardBeta
	defer func() {
		params.ShardNum = oldShardNum
		params.SpringIOTMode = oldIOTMode
		params.SpringIOTCSTRWeight = oldCSTRWeight
		params.SpringIOTBalanceWeight = oldBalanceWeight
		params.SpringIOTCommCostWeight = oldCommCostWeight
		params.SpringIOTHotspotWeight = oldHotspotWeight
		params.SpringIOTHotspotThreshold = oldHotspotThreshold
		params.SpringRewardBeta = oldBeta
	}()
	params.ShardNum = 4
	params.SpringIOTMode = 1
	params.SpringIOTCSTRWeight = 0.55
	params.SpringIOTBalanceWeight = 0.30
	params.SpringIOTCommCostWeight = 0.10
	params.SpringIOTHotspotWeight = 0.05
	params.SpringIOTHotspotThreshold = 0.45
	params.SpringRewardBeta = 0.1

	rthm := &RelayCommitteeModule{}
	stats := map[uint64]SpringBlockStat{
		0: {
			NumTx:                  40,
			InnerTx:                35,
			Relay1Tx:               5,
			CrossTx:                5,
			EffectiveTx:            40,
			CommunicationCostSum:   0.7,
			CommunicationCostCount: 2,
			DecisionBatchIDs:       []uint64{2, 1},
		},
		1: {
			NumTx:                  10,
			InnerTx:                8,
			Relay1Tx:               2,
			CrossTx:                2,
			EffectiveTx:            10,
			CommunicationCostSum:   0.4,
			CommunicationCostCount: 1,
			DecisionBatchIDs:       []uint64{2},
		},
		2: {
			NumTx:                  10,
			InnerTx:                10,
			CrossTx:                0,
			EffectiveTx:            10,
			CommunicationCostSum:   0.3,
			CommunicationCostCount: 1,
		},
		3: {
			NumTx:                  10,
			InnerTx:                9,
			Relay1Tx:               1,
			CrossTx:                1,
			EffectiveTx:            10,
			CommunicationCostSum:   0.2,
			CommunicationCostCount: 1,
		},
	}

	record, ok := rthm.springBuildFeedbackRewardRecord(12, stats)
	if !ok {
		t.Fatal("expected feedback reward record")
	}

	effectiveLoads := []float64{40, 10, 10, 10}
	effectiveTx := 70.0
	crossTx := 8.0
	crossRate := crossTx / effectiveTx
	rCSTR := 1.0 - crossRate
	avgLoad := effectiveTx / 4.0
	rawAbsDiff := 0.0
	rawVar := 0.0
	for _, load := range effectiveLoads {
		diff := load - avgLoad
		rawAbsDiff += math.Abs(diff)
		rawVar += diff * diff
	}
	rawVar /= 4.0
	normVar := rawVar / (avgLoad*avgLoad + 1e-6)
	normVar = normVar / (1.0 + normVar)
	normalizedAbsDiff := rawAbsDiff / (avgLoad + 1e-6)
	rWLB := math.Exp(-0.1*normalizedAbsDiff) * (1.0 - normVar)
	communicationCost := 1.6 / 5.0
	maxLoadShare := 40.0 / (70.0 + 1e-6)
	hotspotPenalty := (maxLoadShare - 0.45) / (1.0 - 0.45 + 1e-6)
	wantReward := 0.55*rCSTR + 0.30*rWLB - 0.10*communicationCost - 0.05*hotspotPenalty

	assertClose(t, record.CrossRate, crossRate)
	assertClose(t, record.RCSTR, rCSTR)
	assertClose(t, record.RWLB, rWLB)
	assertClose(t, record.CommunicationCost, communicationCost)
	assertClose(t, record.MaxLoadShare, maxLoadShare)
	assertClose(t, record.HotspotPenalty, hotspotPenalty)
	assertClose(t, record.Reward, wantReward)
	if record.RewardMode != "iot_dense_balanced" {
		t.Fatalf("reward mode = %q, want iot_dense_balanced", record.RewardMode)
	}
	if got := strings.Join([]string{strconv.FormatUint(record.DecisionBatchIDs[0], 10), strconv.FormatUint(record.DecisionBatchIDs[1], 10)}, ","); got != "1,2" {
		t.Fatalf("decision batch ids = %v, want [1 2]", record.DecisionBatchIDs)
	}
}

func TestSpringBuildFeedbackRewardRecordUsesPaperRewardWhenIOTModeOff(t *testing.T) {
	oldShardNum := params.ShardNum
	oldIOTMode := params.SpringIOTMode
	oldLambda := params.SpringRewardLambda
	oldBeta := params.SpringRewardBeta
	defer func() {
		params.ShardNum = oldShardNum
		params.SpringIOTMode = oldIOTMode
		params.SpringRewardLambda = oldLambda
		params.SpringRewardBeta = oldBeta
	}()
	params.ShardNum = 4
	params.SpringIOTMode = 0
	params.SpringRewardLambda = 0.5
	params.SpringRewardBeta = 0.1

	rthm := &RelayCommitteeModule{}
	stats := map[uint64]SpringBlockStat{
		0: {NumTx: 40, InnerTx: 35, Relay1Tx: 5, CrossTx: 5, EffectiveTx: 40},
		1: {NumTx: 10, InnerTx: 8, Relay1Tx: 2, CrossTx: 2, EffectiveTx: 10},
		2: {NumTx: 10, InnerTx: 10, CrossTx: 0, EffectiveTx: 10},
		3: {NumTx: 10, InnerTx: 9, Relay1Tx: 1, CrossTx: 1, EffectiveTx: 10},
	}

	record, ok := rthm.springBuildFeedbackRewardRecord(13, stats)
	if !ok {
		t.Fatal("expected feedback reward record")
	}

	effectiveLoads := []float64{40, 10, 10, 10}
	effectiveTx := 70.0
	crossTx := 8.0
	crossRate := crossTx / effectiveTx
	rCSTR := 1.0 - crossRate
	avgLoad := effectiveTx / 4.0
	rawAbsDiff := 0.0
	rawVar := 0.0
	for _, load := range effectiveLoads {
		diff := load - avgLoad
		rawAbsDiff += math.Abs(diff)
		rawVar += diff * diff
	}
	rawVar /= 4.0
	normVar := rawVar / (avgLoad*avgLoad + 1e-6)
	normVar = normVar / (1.0 + normVar)
	normalizedAbsDiff := rawAbsDiff / (avgLoad + 1e-6)
	rWLB := math.Exp(-0.1*normalizedAbsDiff) * (1.0 - normVar)
	wantReward := 0.5*rCSTR + 0.5*rWLB

	assertClose(t, record.CrossRate, crossRate)
	assertClose(t, record.RCSTR, rCSTR)
	assertClose(t, record.RWLB, rWLB)
	assertClose(t, record.NormalizedLoadVariance, normVar)
	assertClose(t, record.Reward, wantReward)
	if record.RewardMode != "spring_legacy" {
		t.Fatalf("reward mode = %q, want spring_legacy", record.RewardMode)
	}
}

func TestSpringRandomPlacementModeIsSeeded(t *testing.T) {
	oldShardNum := params.ShardNum
	oldSpringMode := params.SpringMode
	oldRandomSeed := params.SpringRandomSeed
	defer func() {
		params.ShardNum = oldShardNum
		params.SpringMode = oldSpringMode
		params.SpringRandomSeed = oldRandomSeed
	}()
	params.ShardNum = 4
	params.SpringMode = 3
	params.SpringRandomSeed = 7

	newModule := func() *RelayCommitteeModule {
		return &RelayCommitteeModule{
			springAddrShard: make(map[string]uint64),
			springShardLoad: make([]int, params.ShardNum),
		}
	}
	first := newModule()
	second := newModule()

	firstActions := make([]uint64, 0, 6)
	secondActions := make([]uint64, 0, 6)
	for idx := 0; idx < 6; idx++ {
		addr := utils.Address("state-" + strconv.Itoa(idx))
		firstActions = append(firstActions, first.springEnsurePlaced(addr, "", make(map[string]uint64)))
		secondActions = append(secondActions, second.springEnsurePlaced(addr, "", make(map[string]uint64)))
	}

	for idx := range firstActions {
		if firstActions[idx] != secondActions[idx] {
			t.Fatalf("seeded random mismatch at %d: %v vs %v", idx, firstActions, secondActions)
		}
		if firstActions[idx] >= uint64(params.ShardNum) {
			t.Fatalf("random shard = %d, want < %d", firstActions[idx], params.ShardNum)
		}
	}
}

func TestSpringMinStatePlacementChoosesLeastLoadedShard(t *testing.T) {
	oldShardNum := params.ShardNum
	oldSpringMode := params.SpringMode
	defer func() {
		params.ShardNum = oldShardNum
		params.SpringMode = oldSpringMode
	}()
	params.ShardNum = 4
	params.SpringMode = 4

	rthm := &RelayCommitteeModule{
		springAddrShard: make(map[string]uint64),
		springShardLoad: []int{9, 8, 1, 7},
	}

	addr := utils.Address("minstate-candidate")
	for idx := 0; utils.Addr2Shard(addr) == 2 && idx < 100; idx++ {
		addr = utils.Address("minstate-candidate-" + strconv.Itoa(idx))
	}
	if utils.Addr2Shard(addr) == 2 {
		t.Fatal("could not find a test address whose hash shard differs from the least-loaded shard")
	}

	got := rthm.springEnsurePlaced(addr, "", make(map[string]uint64))
	if got != 2 {
		t.Fatalf("MinState shard = %d, want least-loaded shard 2", got)
	}
	if rthm.springShardLoad[2] != 2 {
		t.Fatalf("least-loaded shard count = %d, want 2 after placement", rthm.springShardLoad[2])
	}
}

func TestSpringAnchorOnlyPlacementIgnoresLoadPenalty(t *testing.T) {
	oldShardNum := params.ShardNum
	oldSpringMode := params.SpringMode
	defer func() {
		params.ShardNum = oldShardNum
		params.SpringMode = oldSpringMode
	}()
	params.ShardNum = 4
	params.SpringMode = 5

	related := utils.Address("anchor-on-shard-1")
	addr := utils.Address("anchoronly-candidate")
	for idx := 0; utils.Addr2Shard(addr) == 1 && idx < 100; idx++ {
		addr = utils.Address("anchoronly-candidate-" + strconv.Itoa(idx))
	}
	if utils.Addr2Shard(addr) == 1 {
		t.Fatal("could not find a test address whose hash shard differs from the anchor shard")
	}

	rthm := &RelayCommitteeModule{
		springAddrShard: map[string]uint64{string(related): 1},
		springShardLoad: []int{0, 100, 0, 0},
	}

	got := rthm.springEnsurePlaced(addr, related, make(map[string]uint64))
	if got != 1 {
		t.Fatalf("AnchorOnly shard = %d, want related anchor shard 1 despite load", got)
	}
	if rthm.springShardLoad[1] != 101 {
		t.Fatalf("anchor shard count = %d, want 101 after placement", rthm.springShardLoad[1])
	}
}

func TestSpringNSShardAdaptedPlacementUsesWeightedGraphAndCapacity(t *testing.T) {
	oldShardNum := params.ShardNum
	oldSpringMode := params.SpringMode
	oldCapacityFactor := params.SpringNSShardCapacityFactor
	defer func() {
		params.ShardNum = oldShardNum
		params.SpringMode = oldSpringMode
		params.SpringNSShardCapacityFactor = oldCapacityFactor
	}()
	params.ShardNum = 4
	params.SpringMode = 6
	params.SpringNSShardCapacityFactor = 1.0

	rthm := &RelayCommitteeModule{
		springAddrShard: map[string]uint64{
			"0xheavy-anchor": 1,
			"0xlight-anchor": 2,
		},
		springShardLoad: []int{1, 3, 0, 4},
	}
	batchRelatedWeights := map[string]map[string]float64{
		"0xstate": {
			"0xheavy-anchor": 1.0,
			"0xlight-anchor": 0.5,
		},
	}

	got := rthm.springEnsurePlacedNSShardAdaptedWithBatchWeights(
		utils.Address("0xstate"),
		"",
		make(map[string]uint64),
		batchRelatedWeights,
	)

	if got != 2 {
		t.Fatalf("NSshard-adapted shard = %d, want shard 2 because shard 1 reached capacity", got)
	}
	if rthm.springShardLoad[2] != 1 {
		t.Fatalf("chosen shard load = %d, want 1 after placement", rthm.springShardLoad[2])
	}
}

func TestSpringCandidateMaskCapacityGuardBlocksOverloadedShard(t *testing.T) {
	oldShardNum := params.ShardNum
	oldTopK := params.SpringCandidateTopK
	oldGuard := params.SpringCapacityGuard
	oldFactor := params.SpringCapacityGuardFactor
	oldLoadWeight := params.SpringCandidateLoadWeight
	defer func() {
		params.ShardNum = oldShardNum
		params.SpringCandidateTopK = oldTopK
		params.SpringCapacityGuard = oldGuard
		params.SpringCapacityGuardFactor = oldFactor
		params.SpringCandidateLoadWeight = oldLoadWeight
	}()

	params.ShardNum = 4
	params.SpringCandidateTopK = 2
	params.SpringCapacityGuard = 1
	params.SpringCapacityGuardFactor = 1.2
	params.SpringCandidateLoadWeight = 1.0

	rthm := &RelayCommitteeModule{
		springShardLoad: []int{30, 1, 1, 1},
	}
	mask := rthm.springBuildCandidateActionMask(
		utils.Address("0x"+strings.Repeat("b", 40)),
		[]float64{1.0, 0.0, 0.0, 0.0},
	)

	if mask[0] != 0 {
		t.Fatalf("mask = %v, want overloaded shard 0 blocked", mask)
	}
	if springActionMaskCount(mask) != 2 {
		t.Fatalf("mask = %v, want exactly top-2 candidates", mask)
	}

	guarded, changed := rthm.springApplyCandidateGuard(
		utils.Address("0x"+strings.Repeat("b", 40)),
		[]float64{1.0, 0.0, 0.0, 0.0},
		0,
	)
	if !changed {
		t.Fatalf("guard did not rewrite overloaded action")
	}
	if guarded == 0 {
		t.Fatalf("guarded shard = %d, want non-overloaded shard", guarded)
	}
}

func TestSpringCandidateLoadPressureIncludesRecentBlockStages(t *testing.T) {
	oldShardNum := params.ShardNum
	oldBlockSize := params.MaxBlockSize_global
	defer func() {
		params.ShardNum = oldShardNum
		params.MaxBlockSize_global = oldBlockSize
	}()

	params.ShardNum = 2
	params.MaxBlockSize_global = 1000
	rthm := &RelayCommitteeModule{
		springShardLoad: []int{1, 1},
		springStats: map[uint64][]SpringBlockStat{
			0: {
				{NumTx: 800},
				{NumTx: 900},
			},
			1: {
				{NumTx: 100},
				{NumTx: 200},
			},
		},
	}

	pressures := rthm.springCandidateLoadPressures()
	if len(pressures) != 2 {
		t.Fatalf("pressures = %v, want two shards", pressures)
	}
	if pressures[0] <= pressures[1] {
		t.Fatalf("pressures = %v, want recent hot shard 0 to be higher", pressures)
	}
}

func TestSpringCandidateOnlyChoosesHighestScoreInsideSameTopKMask(t *testing.T) {
	oldShardNum := params.ShardNum
	oldTopK := params.SpringCandidateTopK
	oldGuard := params.SpringCapacityGuard
	oldLoadWeight := params.SpringCandidateLoadWeight
	oldBlockSize := params.MaxBlockSize_global
	defer func() {
		params.ShardNum = oldShardNum
		params.SpringCandidateTopK = oldTopK
		params.SpringCapacityGuard = oldGuard
		params.SpringCandidateLoadWeight = oldLoadWeight
		params.MaxBlockSize_global = oldBlockSize
	}()

	params.ShardNum = 4
	params.SpringCandidateTopK = 2
	params.SpringCapacityGuard = 0
	params.SpringCandidateLoadWeight = 1.0
	params.MaxBlockSize_global = 1000

	rthm := &RelayCommitteeModule{
		springShardLoad: []int{2, 2, 2, 2},
		springStats: map[uint64][]SpringBlockStat{
			0: {},
			1: {},
			2: {},
			3: {},
		},
	}
	addr := utils.Address("candidate-only-test-address")
	senderPos := []float64{0.10, 0.80, 0.30, 0.0}

	got, mask := rthm.springChooseCandidateOnlyShard(addr, senderPos)
	if springActionMaskCount(mask) != 2 {
		t.Fatalf("candidate-only mask = %v, want exactly two candidates", mask)
	}
	if mask[1] != 1 || mask[2] != 1 {
		t.Fatalf("candidate-only mask = %v, want relation-ranked shards 1 and 2", mask)
	}
	if got != 1 {
		t.Fatalf("candidate-only shard = %d, want highest-score shard 1", got)
	}

	// 直接调用共享函数复核：matched baseline 没有第二套候选公式。
	want := rthm.springBestCandidateShard(
		addr,
		senderPos,
		rthm.springBuildCandidateActionMask(addr, senderPos),
	)
	if got != want {
		t.Fatalf("candidate-only shard = %d, shared candidate result = %d", got, want)
	}
}
