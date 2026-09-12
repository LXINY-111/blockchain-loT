package pbft_all

import (
	"blockEmulator/core"
	"blockEmulator/shard"
	"math/big"
	"testing"
	"time"
)

func TestPBFTVotesUseStableMemberIdentity(t *testing.T) {
	node := &PbftConsensusNode{
		ShardID:   0,
		node_nums: 4,
		ip_nodeTable: map[uint64]map[uint64]string{
			0: {0: "n0", 1: "n1", 2: "n2", 3: "n3"},
		},
		cntPrepareConfirm: make(map[string]map[uint64]bool),
		cntCommitConfirm:  make(map[string]map[uint64]bool),
	}

	// Separate decoded pointers with the same NodeID must still count once.
	for idx := 0; idx < 3; idx++ {
		if !node.set2DMap(true, "digest", &shard.Node{ShardID: 0, NodeID: 1, IPaddr: "n1"}) {
			t.Fatal("valid shard member was rejected")
		}
	}
	if got := len(node.cntPrepareConfirm["digest"]); got != 1 {
		t.Fatalf("duplicate sender counted as %d votes, want 1", got)
	}
	if node.set2DMap(false, "digest", &shard.Node{ShardID: 1, NodeID: 1, IPaddr: "n1"}) {
		t.Fatal("cross-shard sender was accepted")
	}
	if node.set2DMap(false, "digest", &shard.Node{ShardID: 0, NodeID: 1, IPaddr: "forged"}) {
		t.Fatal("sender with a forged address was accepted")
	}
}

func TestDeleteElementsInListKeepsAndRemovesExpectedTransactions(t *testing.T) {
	tx1 := core.NewTransaction("a", "b", big.NewInt(1), 1, time.Unix(1, 0))
	tx2 := core.NewTransaction("c", "d", big.NewInt(1), 2, time.Unix(2, 0))
	list := []*core.Transaction{tx1, tx2}

	kept := DeleteElementsInList(append([]*core.Transaction(nil), list...), nil)
	if len(kept) != 2 {
		t.Fatalf("deleting no elements kept %d transactions, want 2", len(kept))
	}
	kept = DeleteElementsInList(append([]*core.Transaction(nil), list...), []*core.Transaction{tx1})
	if len(kept) != 1 || kept[0] != tx2 {
		t.Fatal("deleting tx1 did not preserve only tx2")
	}
}
