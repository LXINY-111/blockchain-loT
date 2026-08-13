package committee

import "time"

type injectionPaceSnapshot struct {
	CumulativeTx     uint64
	TargetTPS        int
	Elapsed          time.Duration
	ActualOfferedTPS float64
	ScheduleLag      time.Duration
}

// injectionPacer keeps one cumulative schedule across TxBatch boundaries.
// This avoids rounding InjectSpeed down when TxBatchSize is not divisible by it.
type injectionPacer struct {
	start time.Time
	sent  uint64
}

func injectionScheduleOffset(cumulativeTx uint64, targetTPS int) time.Duration {
	if cumulativeTx == 0 || targetTPS <= 0 {
		return 0
	}
	seconds := float64(cumulativeTx) / float64(targetTPS)
	return time.Duration(seconds * float64(time.Second))
}

func (p *injectionPacer) waitAfter(sentNow int, targetTPS int) injectionPaceSnapshot {
	if sentNow <= 0 || targetTPS <= 0 {
		return injectionPaceSnapshot{TargetTPS: targetTPS}
	}

	now := time.Now()
	if p.start.IsZero() {
		p.start = now
	}
	p.sent += uint64(sentNow)

	targetOffset := injectionScheduleOffset(p.sent, targetTPS)
	if wait := time.Until(p.start.Add(targetOffset)); wait > 0 {
		time.Sleep(wait)
	}

	elapsed := time.Since(p.start)
	actualTPS := 0.0
	if elapsed > 0 {
		actualTPS = float64(p.sent) / elapsed.Seconds()
	}
	return injectionPaceSnapshot{
		CumulativeTx:     p.sent,
		TargetTPS:        targetTPS,
		Elapsed:          elapsed,
		ActualOfferedTPS: actualTPS,
		ScheduleLag:      elapsed - targetOffset,
	}
}
