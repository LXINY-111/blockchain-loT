package measure

import (
	"blockEmulator/core"
	"blockEmulator/message"
	"blockEmulator/params"
	"encoding/csv"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestTxDetailIdentityAndOutOfOrderRelay(t *testing.T) {
	oldStart, oldPath := params.DatasetStartTx, params.DataWrite_path
	oldMode, oldIdentity := params.SpringIOTMode, params.SpringIOTIdentityMode
	t.Cleanup(func() {
		params.DatasetStartTx, params.DataWrite_path = oldStart, oldPath
		params.SpringIOTMode, params.SpringIOTIdentityMode = oldMode, oldIdentity
	})
	params.SpringIOTMode, params.SpringIOTIdentityMode = 1, 2
	params.DatasetStartTx, params.DataWrite_path = 250000, t.TempDir()+string(os.PathSeparator)
	start := time.Unix(1700000000, 0)
	inner := &core.Transaction{Nonce: 0, TxHash: []byte{1}, Sender: "a", Recipient: "b", Time: start}
	relay := &core.Transaction{Nonce: 1, TxHash: []byte{2}, Sender: "c", Recipient: "d", Time: start}
	metric := NewTestTxDetail()
	metric.UpdateMeasureRecord(&message.BlockInfoMsg{BlockBodyLength: 1, SenderShardID: 0,
		InnerShardTxs: []*core.Transaction{inner}, ProposeTime: start, CommitTime: start.Add(time.Second)})
	// 监督端异步接收时 Relay2 可能先到；不能覆盖其已知的接收分片。
	metric.UpdateMeasureRecord(&message.BlockInfoMsg{BlockBodyLength: 1, SenderShardID: 3,
		Relay2Txs: []*core.Transaction{relay}, CommitTime: start.Add(3 * time.Second)})
	metric.UpdateMeasureRecord(&message.BlockInfoMsg{BlockBodyLength: 1, SenderShardID: 0,
		Relay1Txs: []*core.Transaction{relay}, ProposeTime: start, CommitTime: start.Add(time.Second)})
	metric.OutputRecord()
	file, err := os.Open(filepath.Join(params.DataWrite_path, "supervisor_measureOutput", "Tx_Details.csv"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	rows, err := csv.NewReader(file).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 || len(rows[0]) != 16 || rows[0][8] != "Confirmed latency of this tx (ms)" {
		t.Fatalf("unexpected CSV schema: %v", rows)
	}
	for _, row := range rows[1:] {
		if row[9] == "0" {
			if row[10] != "250000" || row[13] != "0" || row[14] != "0" || row[15] != "0" || row[8] != "1000" {
				t.Fatalf("inner metadata: %v", row)
			}
		} else if row[10] != "250001" || row[11] != "c" || row[12] != "d" || row[13] != "0" || row[14] != "3" || row[15] != "1" || row[8] != "3000" {
			t.Fatalf("relay metadata: %v", row)
		}
	}
}

func TestTxDetailIncompleteRelayAndBrokerDoNotInventMetadata(t *testing.T) {
	detail := &txMetricDetailTime{MetadataKnown: true, Nonce: 0, SenderShardKnown: true, SenderShard: 0}
	row := detail.metadataCSV()
	if row[4] != "0" || row[5] != "" || row[6] != "" {
		t.Fatalf("incomplete relay must not look confirmed: %v", row)
	}
	for _, cell := range (&txMetricDetailTime{}).metadataCSV() {
		if cell != "" {
			t.Fatalf("unknown original identity must remain blank")
		}
	}
}
