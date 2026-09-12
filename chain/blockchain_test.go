package chain

import (
	"blockEmulator/core"
	"blockEmulator/params"
	"fmt"
	"log"
	"math/big"
	"path/filepath"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/core/rawdb"
)

func TestBlockChain(t *testing.T) {
	useTempChainDataDir(t)
	accounts := []string{"000000000001", "00000000002", "00000000003", "00000000004", "00000000005", "00000000006"}
	as := make([]*core.AccountState, 0)
	for idx := range accounts {
		as = append(as, &core.AccountState{
			Balance: big.NewInt(int64(idx)),
		})
	}
	fp := params.DatabaseWrite_path + "mptDB/ldb/s0/N0"
	fmt.Println(fp)
	db, err := rawdb.NewLevelDBDatabase(fp, 0, 1, "accountState", false)
	if err != nil {
		log.Panic(err)
	}
	params.ShardNum = 1
	pcc := &params.ChainConfig{
		ChainID:        0,
		NodeID:         0,
		ShardID:        0,
		Nodes_perShard: 1,
		ShardNums:      1,
		BlockSize:      uint64(params.MaxBlockSize_global),
		BlockInterval:  uint64(params.Block_Interval),
		InjectSpeed:    uint64(params.InjectSpeed),
	}
	CurChain, _ := NewBlockChain(pcc, db)
	CurChain.PrintBlockChain()
	CurChain.AddAccounts(accounts, as, 0)
	CurChain.PrintBlockChain()

	// add the same account.
	CurChain.AddAccounts([]string{accounts[0]}, []*core.AccountState{as[0]}, 0)
	CurChain.PrintBlockChain()

	astates := CurChain.FetchAccounts(accounts)
	for idx, state := range astates {
		fmt.Println(accounts[idx], state.Balance)
	}
	CurChain.CloseBlockChain()

	// t.TempDir removes only this test's isolated data after the test returns.
	fmt.Printf("Done")
}

func TestBlockValidationRejectsWrongRootsAndReplay(t *testing.T) {
	useTempChainDataDir(t)
	params.ShardNum = 1
	pcc := &params.ChainConfig{
		ChainID:        0,
		NodeID:         0,
		ShardID:        0,
		Nodes_perShard: 1,
		ShardNums:      1,
		BlockSize:      10,
		BlockInterval:  1000,
		InjectSpeed:    10,
	}
	db, err := rawdb.NewLevelDBDatabase(
		params.DatabaseWrite_path+"mptDB/ldb/s0/n0",
		0,
		1,
		"accountState",
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	blockchain, err := NewBlockChain(pcc, db)
	if err != nil {
		t.Fatal(err)
	}
	defer blockchain.CloseBlockChain()

	tx := core.NewTransaction("sender", "recipient", big.NewInt(5), 1, time.Unix(1, 0))
	blockchain.Txpool.AddTx2Pool(tx)
	block := blockchain.GenerateBlock(0)
	// Simulate a follower, which still has the injected transaction locally.
	blockchain.Txpool.AddTx2Pool(tx)
	if err := blockchain.AddBlock(block); err != nil {
		t.Fatalf("valid block was rejected: %v", err)
	}
	if got := blockchain.Txpool.GetTxQueueLen(); got != 0 {
		t.Fatalf("committed transaction remained in follower pool: %d", got)
	}

	replayHeader := &core.BlockHeader{
		ParentBlockHash: blockchain.CurrentBlock.Hash,
		StateRoot:       blockchain.CurrentBlock.Header.StateRoot,
		TxRoot:          GetTxTreeRoot([]*core.Transaction{tx}),
		Bloom:           *GetBloomFilter([]*core.Transaction{tx}),
		Number:          blockchain.CurrentBlock.Header.Number + 1,
		Time:            time.Unix(2, 0),
	}
	replay := core.NewBlock(replayHeader, []*core.Transaction{tx})
	replay.Hash = replay.Header.Hash()
	if err := blockchain.IsValidBlock(replay); err == nil {
		t.Fatal("an already committed transaction was accepted again")
	}

	tx2 := core.NewTransaction("sender2", "recipient2", big.NewInt(7), 2, time.Unix(3, 0))
	blockchain.Txpool.AddTx2Pool(tx2)
	wrongRoot := blockchain.GenerateBlock(0)
	wrongRoot.Header.StateRoot = []byte("wrong-state-root")
	wrongRoot.Hash = wrongRoot.Header.Hash()
	if err := blockchain.IsValidBlock(wrongRoot); err == nil {
		t.Fatal("block with a wrong StateRoot was accepted")
	}

	emptyHeader := &core.BlockHeader{
		ParentBlockHash: blockchain.CurrentBlock.Hash,
		StateRoot:       blockchain.CurrentBlock.Header.StateRoot,
		TxRoot:          GetTxTreeRoot(nil),
		Bloom:           *GetBloomFilter(nil),
		Number:          blockchain.CurrentBlock.Header.Number + 1,
		Time:            time.Unix(4, 0),
	}
	badHash := core.NewBlock(emptyHeader, nil)
	badHash.Hash = []byte("wrong-header-hash")
	if err := blockchain.IsValidBlock(badHash); err == nil {
		t.Fatal("block with a wrong header hash was accepted")
	}
}

func useTempChainDataDir(t *testing.T) {
	t.Helper()
	oldExpDataRootDir := params.ExpDataRootDir
	oldDataWritePath := params.DataWrite_path
	oldLogWritePath := params.LogWrite_path
	oldDatabaseWritePath := params.DatabaseWrite_path
	oldShardNum := params.ShardNum

	root := filepath.ToSlash(t.TempDir())
	params.ExpDataRootDir = root
	params.DataWrite_path = root + "/result/"
	params.LogWrite_path = root + "/log"
	params.DatabaseWrite_path = root + "/database/"

	t.Cleanup(func() {
		params.ExpDataRootDir = oldExpDataRootDir
		params.DataWrite_path = oldDataWritePath
		params.LogWrite_path = oldLogWritePath
		params.DatabaseWrite_path = oldDatabaseWritePath
		params.ShardNum = oldShardNum
	})
}
