package committee

import (
	"testing"
	"time"
)

func TestInjectionScheduleOffsetDoesNotRoundAtBatchBoundary(t *testing.T) {
	tests := []struct {
		name         string
		cumulativeTx uint64
		targetTPS    int
		want         time.Duration
	}{
		{name: "200 TPS", cumulativeTx: 1000, targetTPS: 200, want: 5 * time.Second},
		{name: "250 TPS", cumulativeTx: 1000, targetTPS: 250, want: 4 * time.Second},
		{name: "300 TPS", cumulativeTx: 1000, targetTPS: 300, want: 10 * time.Second / 3},
		{name: "350 TPS", cumulativeTx: 1000, targetTPS: 350, want: 20 * time.Second / 7},
		{name: "400 TPS", cumulativeTx: 1000, targetTPS: 400, want: 5 * time.Second / 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := injectionScheduleOffset(tt.cumulativeTx, tt.targetTPS)
			delta := got - tt.want
			if delta < 0 {
				delta = -delta
			}
			if delta > time.Nanosecond {
				t.Fatalf("offset=%s, want=%s", got, tt.want)
			}
		})
	}
}

func TestInjectionScheduleOffsetUsesCumulativeTransactions(t *testing.T) {
	firstBatch := injectionScheduleOffset(1000, 300)
	secondBatch := injectionScheduleOffset(2000, 300)

	if secondBatch != 2*firstBatch {
		t.Fatalf(
			"second cumulative offset=%s, want twice first offset=%s",
			secondBatch,
			2*firstBatch,
		)
	}
}

func TestInjectionScheduleOffsetHandlesDisabledInput(t *testing.T) {
	if got := injectionScheduleOffset(1000, 0); got != 0 {
		t.Fatalf("offset with zero target=%s, want=0", got)
	}
	if got := injectionScheduleOffset(0, 300); got != 0 {
		t.Fatalf("offset with zero transactions=%s, want=0", got)
	}
}
