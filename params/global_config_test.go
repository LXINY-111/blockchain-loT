package params

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExp7DefaultConfigTargets16ShardMainline(t *testing.T) {
	if ShardNum != 16 {
		t.Fatalf("ShardNum = %d, want 16", ShardNum)
	}
	if SpringCandidateTopK != 8 {
		t.Fatalf("SpringCandidateTopK = %d, want 8", SpringCandidateTopK)
	}
	if SpringCapacityGuard != 1 {
		t.Fatalf("SpringCapacityGuard = %d, want 1", SpringCapacityGuard)
	}
	if SpringCapacityGuardFactor != 1.3 {
		t.Fatalf("SpringCapacityGuardFactor = %f, want 1.3", SpringCapacityGuardFactor)
	}
	if !strings.Contains(SpringModelFile, `实验结果7`) {
		t.Fatalf("SpringModelFile = %q, want experiment 7 model path", SpringModelFile)
	}
	if !strings.Contains(SpringModelFile, `ppo_top8_guard13_w4540_16s_seed7.pt`) {
		t.Fatalf("SpringModelFile = %q, want TopK8 Guard1.3 w45-40 model path", SpringModelFile)
	}
}

func TestParamsConfigJSONHasNoUTF8BOM(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "paramsConfig.json"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.HasPrefix(data, []byte{0xEF, 0xBB, 0xBF}) {
		t.Fatal("paramsConfig.json has UTF-8 BOM; Go json.Unmarshal will reject it")
	}

	var config globalConfig
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatalf("paramsConfig.json should parse with Go json.Unmarshal: %v", err)
	}
}
