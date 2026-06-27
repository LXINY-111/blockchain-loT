package build

import (
	"blockEmulator/params"
	"testing"
)

func TestIpTableContains16ShardLocalLayout(t *testing.T) {
	ipMap := readIpTable("../ipTable.json")

	for sid := uint64(0); sid < 16; sid++ {
		nodes, ok := ipMap[sid]
		if !ok {
			t.Fatalf("missing shard %d in ipTable.json", sid)
		}
		if len(nodes) < 4 {
			t.Fatalf("shard %d has %d nodes, want at least 4", sid, len(nodes))
		}
	}

	supervisor, ok := ipMap[params.SupervisorShard]
	if !ok || supervisor[0] == "" {
		t.Fatalf("missing supervisor address in ipTable.json")
	}
}
