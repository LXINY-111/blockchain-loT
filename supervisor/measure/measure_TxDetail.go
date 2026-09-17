package measure

import (
	"blockEmulator/core"
	"blockEmulator/message"
	"blockEmulator/params"
	"math/big"
	"strconv"
	"time"
)

type txMetricDetailTime struct {
	// normal tx time
	TxProposeTimestamp, BlockProposeTimestamp, TxCommitTimestamp time.Time

	// relay tx time
	Relay1CommitTimestamp, Relay2CommitTimestamp time.Time

	// broker tx time
	Broker1CommitTimestamp, Broker2CommitTimestamp time.Time

	// 仅补充测量关联键，不修改交易、分片决策或共识消息。
	// DatasetRowIndex 是当前输入文件中从零开始的行号；分块数据的全局
	// tx_index 由分析端查场景表获得，不能把文件行号误当成全局序号。
	MetadataKnown                         bool
	Nonce, DatasetRowIndex                uint64
	Sender, Recipient                     string
	SenderShard, RecipientShard           uint64
	SenderShardKnown, RecipientShardKnown bool
}

func (detail *txMetricDetailTime) captureIdentity(tx *core.Transaction) {
	if params.SpringIOTMode != 1 || params.SpringIOTIdentityMode != 2 {
		// 只有 v2 的严格输入契约能保证有效交易序号等于物理文件行号。
		return
	}
	detail.MetadataKnown = true
	detail.Nonce = tx.Nonce
	detail.DatasetRowIndex = uint64(params.DatasetStartTx) + tx.Nonce
	detail.Sender, detail.Recipient = string(tx.Sender), string(tx.Recipient)
}

func (detail *txMetricDetailTime) metadataCSV() []string {
	if !detail.MetadataKnown {
		// Broker 拆分交易不能冒充原交易的关联键；原有时间指标照常保留。
		return make([]string, 7)
	}
	senderShard, recipientShard, cross := "", "", ""
	if detail.SenderShardKnown {
		senderShard = strconv.FormatUint(detail.SenderShard, 10)
	}
	if detail.RecipientShardKnown {
		recipientShard = strconv.FormatUint(detail.RecipientShard, 10)
	}
	if detail.SenderShardKnown && detail.RecipientShardKnown {
		cross = "0"
		if detail.SenderShard != detail.RecipientShard {
			cross = "1"
		}
	}
	return []string{strconv.FormatUint(detail.Nonce, 10), strconv.FormatUint(detail.DatasetRowIndex, 10),
		detail.Sender, detail.Recipient, senderShard, recipientShard, cross}
}

// to test Tx detail
type TestTxDetail struct {
	txHash2DetailTime map[string]*txMetricDetailTime
}

func NewTestTxDetail() *TestTxDetail {
	return &TestTxDetail{
		txHash2DetailTime: make(map[string]*txMetricDetailTime),
	}
}

func (ttd *TestTxDetail) OutputMetricName() string {
	return "Tx_Details"
}

func (ttd *TestTxDetail) UpdateMeasureRecord(b *message.BlockInfoMsg) {
	if b.BlockBodyLength == 0 { // empty block
		return
	}

	for _, innertx := range b.InnerShardTxs {
		if _, ok := ttd.txHash2DetailTime[string(innertx.TxHash)]; !ok {
			ttd.txHash2DetailTime[string(innertx.TxHash)] = &txMetricDetailTime{}
		}
		ttd.txHash2DetailTime[string(innertx.TxHash)].TxProposeTimestamp = innertx.Time
		ttd.txHash2DetailTime[string(innertx.TxHash)].BlockProposeTimestamp = b.ProposeTime
		ttd.txHash2DetailTime[string(innertx.TxHash)].TxCommitTimestamp = b.CommitTime
		detail := ttd.txHash2DetailTime[string(innertx.TxHash)]
		detail.captureIdentity(innertx)
		detail.SenderShard, detail.RecipientShard = b.SenderShardID, b.SenderShardID
		detail.SenderShardKnown, detail.RecipientShardKnown = true, true
	}
	for _, r1tx := range b.Relay1Txs {
		if _, ok := ttd.txHash2DetailTime[string(r1tx.TxHash)]; !ok {
			ttd.txHash2DetailTime[string(r1tx.TxHash)] = &txMetricDetailTime{}
		}
		ttd.txHash2DetailTime[string(r1tx.TxHash)].TxProposeTimestamp = r1tx.Time
		ttd.txHash2DetailTime[string(r1tx.TxHash)].BlockProposeTimestamp = b.ProposeTime
		ttd.txHash2DetailTime[string(r1tx.TxHash)].Relay1CommitTimestamp = b.CommitTime
		detail := ttd.txHash2DetailTime[string(r1tx.TxHash)]
		detail.captureIdentity(r1tx)
		detail.SenderShard, detail.SenderShardKnown = b.SenderShardID, true
	}
	for _, r2tx := range b.Relay2Txs {
		if _, ok := ttd.txHash2DetailTime[string(r2tx.TxHash)]; !ok {
			ttd.txHash2DetailTime[string(r2tx.TxHash)] = &txMetricDetailTime{}
		}
		ttd.txHash2DetailTime[string(r2tx.TxHash)].Relay2CommitTimestamp = b.CommitTime
		ttd.txHash2DetailTime[string(r2tx.TxHash)].TxCommitTimestamp = b.CommitTime
		detail := ttd.txHash2DetailTime[string(r2tx.TxHash)]
		detail.captureIdentity(r2tx)
		detail.RecipientShard, detail.RecipientShardKnown = b.SenderShardID, true
	}
	for _, b1tx := range b.Broker1Txs {
		if _, ok := ttd.txHash2DetailTime[string(b1tx.RawTxHash)]; !ok {
			ttd.txHash2DetailTime[string(b1tx.RawTxHash)] = &txMetricDetailTime{}
		}
		ttd.txHash2DetailTime[string(b1tx.RawTxHash)].TxProposeTimestamp = b1tx.Time
		ttd.txHash2DetailTime[string(b1tx.RawTxHash)].BlockProposeTimestamp = b.ProposeTime
		ttd.txHash2DetailTime[string(b1tx.RawTxHash)].Broker1CommitTimestamp = b.CommitTime
	}
	for _, b2tx := range b.Broker2Txs {
		if _, ok := ttd.txHash2DetailTime[string(b2tx.RawTxHash)]; !ok {
			ttd.txHash2DetailTime[string(b2tx.RawTxHash)] = &txMetricDetailTime{}
		}
		ttd.txHash2DetailTime[string(b2tx.RawTxHash)].Broker2CommitTimestamp = b.CommitTime
		ttd.txHash2DetailTime[string(b2tx.RawTxHash)].TxCommitTimestamp = b.CommitTime
	}
}

func (ttd *TestTxDetail) HandleExtraMessage([]byte) {}

func (ttd *TestTxDetail) OutputRecord() (perEpochCTXs []float64, totTxNum float64) {
	ttd.writeToCSV()
	return []float64{}, 0
}

func (ttd *TestTxDetail) writeToCSV() {
	fileName := ttd.OutputMetricName()
	measureName := []string{
		"TxHash (Byte -> Big Int)",
		"Tx propose timestamp",
		"Block propose timestamp",
		"Tx finally commit timestamp",
		"Relay1 Tx commit timestamp (not a relay tx -> nil)",
		"Relay2 Tx commit timestamp (not a relay tx -> nil)",
		"Broker1 Tx commit timestamp (not a broker tx -> nil)",
		"Broker2 Tx commit timestamp (not a broker tx -> nil)",
		"Confirmed latency of this tx (ms)",
		// 追加字段，保持原九列名称和位置，兼容旧分析脚本。
		"Tx local nonce", "Dataset row index", "Sender address", "Recipient address",
		"Sender shard", "Recipient shard", "Cross shard",
	}
	measureVals := make([][]string, 0)

	for key, val := range ttd.txHash2DetailTime {
		latency := ""
		if !val.TxProposeTimestamp.IsZero() && !val.TxCommitTimestamp.IsZero() && !val.TxCommitTimestamp.Before(val.TxProposeTimestamp) {
			latency = strconv.FormatInt(val.TxCommitTimestamp.Sub(val.TxProposeTimestamp).Milliseconds(), 10)
		}
		csvLine := []string{
			new(big.Int).SetBytes([]byte(key)).String(),

			timestampToString(val.TxProposeTimestamp),
			timestampToString(val.BlockProposeTimestamp),
			timestampToString(val.TxCommitTimestamp),

			timestampToString(val.Relay1CommitTimestamp),
			timestampToString(val.Relay2CommitTimestamp),

			timestampToString(val.Broker1CommitTimestamp),
			timestampToString(val.Broker2CommitTimestamp),

			latency,
		}
		csvLine = append(csvLine, val.metadataCSV()...)
		measureVals = append(measureVals, csvLine)
	}

	WriteMetricsToCSV(fileName, measureName, measureVals)
}

// zero time to empty string
func timestampToString(thisTime time.Time) string {
	if thisTime.IsZero() {
		return ""
	}
	return strconv.FormatInt(thisTime.UnixMilli(), 10)
}
