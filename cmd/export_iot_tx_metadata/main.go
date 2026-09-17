// export_iot_tx_metadata 从已结束实验的区块数据库恢复交易关联键。
// 只读复制原文件，在系统临时副本上沿节点 0 的主链读取，避免副本重复计数。
// 不调用 storage.NewStorage：该函数会建桶/回填索引，不能用于只读审计。
package main

import (
	"blockEmulator/core"
	"bytes"
	"encoding/csv"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/boltdb/bolt"
)

type identity struct {
	tx                          *core.Transaction
	senderShard, recipientShard int
	originalCount, relayCount   int
}

func accumulate(records map[uint64]*identity, tx *core.Transaction, sid, count int) error {
	if tx == nil || len(tx.TxHash) == 0 || tx.HasBroker || tx.Nonce >= uint64(count) {
		return fmt.Errorf("invalid or out-of-window transaction in shard %d", sid)
	}
	record := records[tx.Nonce]
	if record == nil {
		record = &identity{tx: tx, senderShard: -1, recipientShard: -1}
		records[tx.Nonce] = record
	} else if !bytes.Equal(record.tx.TxHash, tx.TxHash) || record.tx.Sender != tx.Sender || record.tx.Recipient != tx.Recipient {
		return fmt.Errorf("conflicting transaction identity for nonce %d", tx.Nonce)
	}
	if tx.Relayed {
		record.relayCount++
		record.recipientShard = sid
	} else {
		record.originalCount++
		record.senderShard = sid
	}
	return nil
}

func readShard(path string, sid, count int, records map[uint64]*identity) error {
	// Bolt 1.3.1 在 Windows 下即使 ReadOnly 也会创建 .lock，且其只读关闭
	// 路径不会释放锁。使用临时副本隔离该行为，原实验目录完全不产生写入。
	original, err := os.Open(path)
	if err != nil {
		return err
	}
	defer original.Close()
	snapshot, err := os.CreateTemp("", "iot-db-snapshot-*")
	if err != nil {
		return err
	}
	defer os.Remove(snapshot.Name()) // 仅删除本函数刚创建的临时文件。
	_, copyErr := io.Copy(snapshot, original)
	closeErr := snapshot.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	db, err := bolt.Open(snapshot.Name(), 0600, &bolt.Options{Timeout: time.Second})
	if err != nil {
		return err
	}
	defer db.Close()
	return db.View(func(tx *bolt.Tx) error {
		blocks, latest := tx.Bucket([]byte("block")), tx.Bucket([]byte("newestBlockHash"))
		if blocks == nil || latest == nil {
			return fmt.Errorf("missing chain buckets: %s", path)
		}
		key := latest.Get([]byte("OnlyNewestBlock"))
		if len(key) == 0 {
			return fmt.Errorf("missing canonical chain head: %s", path)
		}
		visited := make(map[string]bool)
		for {
			if visited[string(key)] {
				return fmt.Errorf("cycle in canonical chain: %s", path)
			}
			visited[string(key)] = true
			encoded := blocks.Get(key)
			if len(encoded) == 0 {
				return fmt.Errorf("missing canonical block: %s", path)
			}
			var block core.Block
			if err := gob.NewDecoder(bytes.NewReader(encoded)).Decode(&block); err != nil {
				return err
			}
			if block.Header == nil || !bytes.Equal(block.Hash, key) {
				return fmt.Errorf("invalid block identity: %s", path)
			}
			for _, transaction := range block.Body {
				if err := accumulate(records, transaction, sid, count); err != nil {
					return err
				}
			}
			if block.Header.Number == 0 {
				break
			}
			key = block.Header.ParentBlockHash
		}
		return nil
	})
}

func recoverRows(runDir string) ([][]string, error) {
	readJSON := func(name string, target interface{}) error {
		data, err := os.ReadFile(filepath.Join(runDir, name))
		if err != nil {
			return err
		}
		return json.Unmarshal(bytes.TrimPrefix(data, []byte{0xef, 0xbb, 0xbf}), target)
	}
	var config struct{ SpringIOTMode, SpringIOTIdentityMode, ConsensusMethod, DatasetStartTx, TotalDataSize int }
	var context struct{ Shards, Nodes int }
	var status struct{ State string }
	if err := readJSON("paramsConfig.json", &config); err != nil {
		return nil, err
	}
	if err := readJSON("run_context.json", &context); err != nil {
		return nil, err
	}
	if err := readJSON("run_status.json", &status); err != nil {
		return nil, err
	}
	if status.State != "completed" || config.SpringIOTMode != 1 || config.SpringIOTIdentityMode != 2 || config.ConsensusMethod != 3 {
		return nil, fmt.Errorf("require a completed IoT v2 relay run")
	}
	if context.Shards <= 0 || context.Nodes <= 0 || config.DatasetStartTx < 0 || config.TotalDataSize <= 0 {
		return nil, fmt.Errorf("invalid run dimensions or transaction window")
	}
	records := make(map[uint64]*identity)
	for sid := 0; sid < context.Shards; sid++ {
		// 使用当前运行目录中的数据库，不跟随配置里迁移前的绝对路径。
		path := filepath.Join(runDir, "expTest", "database", "chainDB", fmt.Sprintf("S%d_N0", sid))
		if err := readShard(path, sid, config.TotalDataSize, records); err != nil {
			return nil, err
		}
	}
	if len(records) != config.TotalDataSize {
		return nil, fmt.Errorf("incomplete archive: got %d of %d transactions", len(records), config.TotalDataSize)
	}
	rows := [][]string{{"TxHash (Byte -> Big Int)", "Tx local nonce", "Dataset row index", "Sender address",
		"Recipient address", "Sender shard", "Recipient shard", "Cross shard"}}
	hashes := make(map[string]bool)
	for nonce := 0; nonce < config.TotalDataSize; nonce++ {
		record := records[uint64(nonce)]
		if record == nil || record.originalCount != 1 || record.relayCount > 1 {
			return nil, fmt.Errorf("missing or repeated committed stage at nonce %d", nonce)
		}
		cross := "0"
		if record.relayCount == 0 {
			record.recipientShard = record.senderShard
		} else {
			if record.recipientShard == record.senderShard {
				return nil, fmt.Errorf("same-shard relay at nonce %d", nonce)
			}
			cross = "1"
		}
		hash := new(big.Int).SetBytes(record.tx.TxHash).String()
		if hashes[hash] {
			return nil, fmt.Errorf("transaction hash reused by different nonces")
		}
		hashes[hash] = true
		rows = append(rows, []string{hash, strconv.Itoa(nonce), strconv.Itoa(config.DatasetStartTx + nonce),
			string(record.tx.Sender), string(record.tx.Recipient), strconv.Itoa(record.senderShard),
			strconv.Itoa(record.recipientShard), cross})
	}
	return rows, nil
}

func run() error {
	runDir := flag.String("run-dir", "", "completed run directory (databases remain read-only)")
	output := flag.String("output", "", "new metadata CSV; omit to write to stdout")
	flag.Parse()
	if *runDir == "" {
		return fmt.Errorf("--run-dir is required")
	}
	rows, err := recoverRows(*runDir)
	if err != nil {
		return err
	}
	var writer io.Writer = os.Stdout
	if *output != "" {
		file, err := os.OpenFile(*output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			return err
		}
		defer file.Close()
		writer = file
	}
	csvWriter := csv.NewWriter(writer)
	return csvWriter.WriteAll(rows)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
