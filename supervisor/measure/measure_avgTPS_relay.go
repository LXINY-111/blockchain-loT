package measure

import (
	"blockEmulator/message"
	"strconv"
	"time"
)

// to test average TPS in this system
type TestModule_avgTPS_Relay struct {
	epochID      int
	excutedTxNum []float64 // record how many excuted txs in a epoch, maybe the cross shard tx will be calculated as a 0.5 tx

	normalTxNum []int
	relay1TxNum []int
	relay2TxNum []int

	startTime []time.Time // record when the epoch starts
	endTime   []time.Time // record when the epoch ends
}

func NewTestModule_avgTPS_Relay() *TestModule_avgTPS_Relay {
	return &TestModule_avgTPS_Relay{
		epochID:      -1,
		excutedTxNum: make([]float64, 0),
		startTime:    make([]time.Time, 0),
		endTime:      make([]time.Time, 0),

		normalTxNum: make([]int, 0),
		relay1TxNum: make([]int, 0),
		relay2TxNum: make([]int, 0),
	}
}

func (tat *TestModule_avgTPS_Relay) OutputMetricName() string {
	return "Average_TPS"
}

// add the number of excuted txs, and change the time records
func (tat *TestModule_avgTPS_Relay) UpdateMeasureRecord(b *message.BlockInfoMsg) {
	if b.BlockBodyLength == 0 { // empty block
		return
	}

	epochid := b.Epoch
	earliestTime := b.ProposeTime
	latestTime := b.CommitTime
	r1TxNum := len(b.Relay1Txs)
	r2TxNum := len(b.Relay2Txs)

	// extend
	for tat.epochID < epochid {
		tat.excutedTxNum = append(tat.excutedTxNum, 0)
		tat.startTime = append(tat.startTime, time.Time{})
		tat.endTime = append(tat.endTime, time.Time{})

		tat.relay1TxNum = append(tat.relay1TxNum, 0)
		tat.relay2TxNum = append(tat.relay2TxNum, 0)
		tat.normalTxNum = append(tat.normalTxNum, 0)

		tat.epochID++
	}

	// modify the local epoch data
	tat.excutedTxNum[epochid] += float64(r1TxNum+r2TxNum) / 2
	tat.excutedTxNum[epochid] += float64(len(b.InnerShardTxs))

	tat.normalTxNum[epochid] += len(b.InnerShardTxs)
	tat.relay1TxNum[epochid] += r1TxNum
	tat.relay2TxNum[epochid] += r2TxNum

	if tat.startTime[epochid].IsZero() || tat.startTime[epochid].After(earliestTime) {
		tat.startTime[epochid] = earliestTime
	}
	if tat.endTime[epochid].IsZero() || latestTime.After(tat.endTime[epochid]) {
		tat.endTime[epochid] = latestTime
	}
}

func (tat *TestModule_avgTPS_Relay) HandleExtraMessage([]byte) {}

// output the average TPS
func (tat *TestModule_avgTPS_Relay) OutputRecord() (perEpochTPS []float64, totalTPS float64) {
	tat.writeToCSV()
	return tat.calculateTPS()
}

func (tat *TestModule_avgTPS_Relay) calculateTPS() (perEpochTPS []float64, totalTPS float64) {
	// calculate the simple result
	perEpochTPS = make([]float64, tat.epochID+1)
	totalTxNum := 0.0
	eTime := time.Time{}
	lTime := time.Time{}
	for eid, exTxNum := range tat.excutedTxNum {
		// 开头的空轮次没有提议/提交时间，不能把公元 1 年当作实验起点。
		// 空轮次保留数组位置并返回 0；没有有效时间窗口时不做除法。
		if exTxNum <= 0 || tat.startTime[eid].IsZero() || tat.endTime[eid].IsZero() {
			continue
		}
		timeGap := tat.endTime[eid].Sub(tat.startTime[eid]).Seconds()
		if timeGap <= 0 {
			continue
		}
		perEpochTPS[eid] = exTxNum / timeGap
		totalTxNum += exTxNum
		if eTime.IsZero() || eTime.After(tat.startTime[eid]) {
			eTime = tat.startTime[eid]
		}
		if tat.endTime[eid].After(lTime) {
			lTime = tat.endTime[eid]
		}
	}
	if !eTime.IsZero() && lTime.After(eTime) {
		totalTPS = totalTxNum / (lTime.Sub(eTime).Seconds())
	}
	return
}

func (tat *TestModule_avgTPS_Relay) writeToCSV() {
	fileName := tat.OutputMetricName()
	measureName := []string{"EpochID", "Total tx # in this epoch", "Normal tx # in this epoch", "Relay1 tx # in this epoch", "Relay2 tx # in this epoch", "Epoch start time", "Epoch end time", "Avg. TPS of this epoch"}
	measureVals := make([][]string, 0)
	perEpochTPS, _ := tat.calculateTPS()

	for eid, exTxNum := range tat.excutedTxNum {
		start, end := "", ""
		if !tat.startTime[eid].IsZero() && !tat.endTime[eid].IsZero() {
			start = strconv.FormatInt(tat.startTime[eid].UnixMilli(), 10)
			end = strconv.FormatInt(tat.endTime[eid].UnixMilli(), 10)
		}
		csvLine := []string{
			strconv.Itoa(eid),
			strconv.FormatFloat(exTxNum, 'f', 8, 64),
			strconv.Itoa(tat.normalTxNum[eid]),
			strconv.Itoa(tat.relay1TxNum[eid]),
			strconv.Itoa(tat.relay2TxNum[eid]),
			start,
			end,
			strconv.FormatFloat(perEpochTPS[eid], 'f', 8, 64),
		}
		measureVals = append(measureVals, csvLine)
	}
	WriteMetricsToCSV(fileName, measureName, measureVals)
}
