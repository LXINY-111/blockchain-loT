package utils

import (
	"blockEmulator/params"
	"testing"
)

func TestAddr2ShardKeepsHexAddressMapping(t *testing.T) {
	oldShardNum := params.ShardNum
	defer func() { params.ShardNum = oldShardNum }()
	params.ShardNum = 4

	if got := Addr2Shard("000000000000000f"); got != 3 {
		t.Fatalf("Addr2Shard(hex) = %d, want 3", got)
	}
}

func TestAddr2ShardHashesNonHexIOTObject(t *testing.T) {
	oldShardNum := params.ShardNum
	defer func() { params.ShardNum = oldShardNum }()
	params.ShardNum = 4

	addr := Address("iot_peer:o:syslog|service:any|proto:udp")
	first := Addr2Shard(addr)
	second := Addr2Shard(addr)
	if first < 0 || first >= params.ShardNum {
		t.Fatalf("Addr2Shard(IoT) = %d, want in [0,%d)", first, params.ShardNum)
	}
	if first != second {
		t.Fatalf("Addr2Shard(IoT) is not stable: %d != %d", first, second)
	}
}

func TestAddr2ShardUsesFullIOTObjectNotOnlySuffix(t *testing.T) {
	oldShardNum := params.ShardNum
	defer func() { params.ShardNum = oldShardNum }()
	params.ShardNum = 4

	state := Address("iot_state:camera:aa|iot_peer:o:syslog|service:any|proto:udp|udp")
	peer := Address("iot_peer:o:syslog|service:any|proto:udp")
	if Addr2Shard(state) == Addr2Shard(peer) {
		t.Fatalf("state and peer with same protocol suffix should not be forced to same shard")
	}
}
