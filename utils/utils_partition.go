package utils

import (
	"blockEmulator/params"
	"crypto/sha256"
	"math/big"
	"strconv"
	"strings"
)

// the default method
func Addr2Shard(addr Address) int {
	return Addr2ShardWithShardNum(addr, params.ShardNum)
}

func Addr2ShardWithShardNum(addr Address, shardNum int) int {
	if shardNum <= 0 {
		return 0
	}
	text := string(addr)
	if !isHexAddressText(text) {
		// IoT MDP 会把 device/service/protocol 组合成 iot_state/iot_peer 对象，
		// 这类对象不是十六进制账户地址，使用 SHA-256 做稳定哈希分片。
		return stableTextShard(text, shardNum)
	}

	last8_addr := text
	if len(last8_addr) > 8 {
		last8_addr = last8_addr[len(last8_addr)-8:]
	}
	num, err := strconv.ParseUint(last8_addr, 16, 64)
	if err != nil {
		return stableTextShard(text, shardNum)
	}
	return int(num) % shardNum
}

func stableTextShard(text string, shardNum int) int {
	hash := sha256.Sum256([]byte(text))
	return int(ModBytes(hash[:], uint(shardNum)))
}

func isHexAddressText(text string) bool {
	if strings.HasPrefix(text, "0x") || strings.HasPrefix(text, "0X") {
		text = text[2:]
	}
	if text == "" {
		return false
	}
	for _, ch := range text {
		if (ch >= '0' && ch <= '9') ||
			(ch >= 'a' && ch <= 'f') ||
			(ch >= 'A' && ch <= 'F') {
			continue
		}
		return false
	}
	return true
}

// mod method
func ModBytes(data []byte, mod uint) uint {
	num := new(big.Int).SetBytes(data)
	result := new(big.Int).Mod(num, big.NewInt(int64(mod)))
	return uint(result.Int64())
}
