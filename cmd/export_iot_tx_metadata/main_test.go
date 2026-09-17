package main

import (
	"blockEmulator/core"
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/boltdb/bolt"
)

func writeFixtureDB(t *testing.T, path string, body []*core.Transaction) {
	t.Helper()
	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		t.Fatal(err)
	}
	err = db.Update(func(tx *bolt.Tx) error {
		blocks, _ := tx.CreateBucket([]byte("block"))
		latest, _ := tx.CreateBucket([]byte("newestBlockHash"))
		genesis := &core.Block{Header: &core.BlockHeader{Number: 0}, Hash: []byte{10}}
		head := &core.Block{Header: &core.BlockHeader{Number: 1, ParentBlockHash: genesis.Hash}, Body: body, Hash: []byte{11}}
		// 非主链旧块故意含越界交易；恢复过程必须沿链头读取而不是扫描整个桶。
		orphan := &core.Block{Header: &core.BlockHeader{Number: 1}, Hash: []byte{12},
			Body: []*core.Transaction{{Nonce: 999, TxHash: []byte{99}}}}
		for _, block := range []*core.Block{genesis, head, orphan} {
			if err := blocks.Put(block.Hash, block.Encode()); err != nil {
				return err
			}
		}
		return latest.Put([]byte("OnlyNewestBlock"), head.Hash)
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRecoverCanonicalMetadataWithoutModifyingArchive(t *testing.T) {
	root := t.TempDir()
	for name, value := range map[string]interface{}{
		"paramsConfig.json": map[string]int{"SpringIOTMode": 1, "SpringIOTIdentityMode": 2, "ConsensusMethod": 3, "DatasetStartTx": 250000, "TotalDataSize": 2},
		"run_context.json":  map[string]int{"shards": 2, "nodes": 4},
		"run_status.json":   map[string]string{"state": "completed"},
	} {
		data, _ := json.Marshal(value)
		if err := os.WriteFile(filepath.Join(root, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	directory := filepath.Join(root, "expTest", "database", "chainDB")
	if err := os.MkdirAll(directory, 0700); err != nil {
		t.Fatal(err)
	}
	original := &core.Transaction{Nonce: 0, TxHash: []byte{1}, Sender: "a", Recipient: "b"}
	inner := &core.Transaction{Nonce: 1, TxHash: []byte{2}, Sender: "c", Recipient: "d"}
	relay := *original
	relay.Relayed = true
	paths := []string{filepath.Join(directory, "S0_N0"), filepath.Join(directory, "S1_N0")}
	writeFixtureDB(t, paths[0], []*core.Transaction{original, inner})
	writeFixtureDB(t, paths[1], []*core.Transaction{&relay})
	before := make([][32]byte, len(paths))
	for i, path := range paths {
		data, _ := os.ReadFile(path)
		before[i] = sha256.Sum256(data)
	}
	rows, err := recoverRows(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 || rows[1][2] != "250000" || rows[1][5] != "0" || rows[1][6] != "1" || rows[1][7] != "1" || rows[2][7] != "0" {
		t.Fatalf("unexpected recovered rows: %v", rows)
	}
	for i, path := range paths {
		data, _ := os.ReadFile(path)
		if sha256.Sum256(data) != before[i] {
			t.Fatalf("archive modified: %s", path)
		}
	}
}

func TestRecoverRejectsConflictingIdentityAndMissingStage(t *testing.T) {
	records := make(map[uint64]*identity)
	first := &core.Transaction{Nonce: 0, TxHash: []byte{1}, Sender: "a", Recipient: "b"}
	if err := accumulate(records, first, 0, 1); err != nil {
		t.Fatal(err)
	}
	conflict := *first
	conflict.TxHash = []byte{2}
	if err := accumulate(records, &conflict, 1, 1); err == nil {
		t.Fatal("conflicting nonce was accepted")
	}
	if err := accumulate(records, &core.Transaction{Nonce: 1, TxHash: []byte{3}}, 0, 1); err == nil {
		t.Fatal("out-of-window nonce was accepted")
	}
	if err := readShard(filepath.Join(t.TempDir(), "missing"), 0, 1, records); err == nil {
		t.Fatal("missing database was accepted")
	}
}
