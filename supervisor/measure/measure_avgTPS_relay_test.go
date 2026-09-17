package measure

import (
	"math"
	"testing"
	"time"
)

func TestRelayTPSIgnoresEmptyEpochTimestamps(t *testing.T) {
	start := time.Unix(1700000000, 0)
	metric := &TestModule_avgTPS_Relay{
		epochID:      4,
		excutedTxNum: []float64{0, 0, 3, 0, 5},
		startTime:    []time.Time{{}, {}, start, {}, start.Add(3 * time.Second)},
		endTime:      []time.Time{{}, {}, start.Add(2 * time.Second), {}, start.Add(5 * time.Second)},
	}
	perEpoch, total := metric.calculateTPS()
	if total != 1.6 || perEpoch[2] != 1.5 || perEpoch[4] != 2.5 {
		t.Fatalf("unexpected TPS: %v total=%g", perEpoch, total)
	}
	for _, value := range perEpoch {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("nonfinite per-epoch TPS: %v", perEpoch)
		}
	}
}

func TestRelayTPSWithoutTransactions(t *testing.T) {
	metric := NewTestModule_avgTPS_Relay()
	perEpoch, total := metric.calculateTPS()
	if len(perEpoch) != 0 || total != 0 {
		t.Fatalf("empty workload TPS=%v total=%g", perEpoch, total)
	}
}
