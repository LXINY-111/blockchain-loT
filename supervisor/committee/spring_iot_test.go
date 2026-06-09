package committee

import (
	"blockEmulator/core"
	"blockEmulator/params"
	"math"
	"math/big"
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
	if len(feature.Features) != 6 {
		t.Fatalf("feature dim = %d, want 6", len(feature.Features))
	}

	assertClose(t, feature.Features[0], math.Log1p(300.0)/math.Log1p(1_000_000.0))
	assertClose(t, feature.Features[1], math.Log1p(10.0)/math.Log1p(10_000.0))
	assertClose(t, feature.Features[2], 50.0/60.0)
	assertClose(t, feature.Features[3], 0.8)
	assertClose(t, feature.Features[4], math.Log1p(3.0)/math.Log1p(1000.0))
	assertClose(t, feature.Features[5], 1.0)
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
}

func TestSpringApplyIOTTxIdentityUsesNonceSidecar(t *testing.T) {
	feature := SpringIOTTxFeature{
		TxIndex:       2,
		StateObject:   "0xstate",
		AnchorObject:  "0xdevice",
		AnchorObjects: []string{"0xdevice", "0xpeer"},
		Features:      []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
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

	iotFeatures := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	state := rthm.springBuildStateFromSenderPos(
		[]float64{0.25, 0.25, 0.25, 0.25},
		1.0,
		iotFeatures,
	)

	if len(state) != 11*params.ShardNum+1+6 {
		t.Fatalf("state dim = %d, want %d", len(state), 11*params.ShardNum+1+6)
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
