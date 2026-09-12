package params

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResult9DefaultConfigTargets16ShardMainline(t *testing.T) {
	if ShardNum != 16 {
		t.Fatalf("ShardNum = %d, want 16", ShardNum)
	}
	if SpringCandidateTopK != 7 {
		t.Fatalf("SpringCandidateTopK = %d, want 7", SpringCandidateTopK)
	}
	if SpringCapacityGuard != 0 {
		t.Fatalf("SpringCapacityGuard = %d, want 0", SpringCapacityGuard)
	}
	if SpringCapacityGuardFactor != 1.5 {
		t.Fatalf("SpringCapacityGuardFactor = %f, want 1.5", SpringCapacityGuardFactor)
	}
	if SpringLoadSemanticsVersion != "primary_owner_execution_v1" {
		t.Fatalf("SpringLoadSemanticsVersion = %q, want primary-owner execution semantics", SpringLoadSemanticsVersion)
	}
	if !strings.Contains(SpringModelFile, `实验结果9`) {
		t.Fatalf("SpringModelFile = %q, want Result9 model path", SpringModelFile)
	}
	if !strings.Contains(SpringModelFile, `ppo_top7_w5530_pareto_16s_seed7.pt`) {
		t.Fatalf("SpringModelFile = %q, want Pareto-selected PPO TopK7 w55-30 model path", SpringModelFile)
	}
	if DatasetStartTx != 2_500_000 || TotalDataSize != 444_019 || InjectSpeed != 250 {
		t.Fatalf(
			"DatasetStartTx/TotalDataSize/InjectSpeed = %d/%d/%d, want 2500000/444019/250",
			DatasetStartTx,
			TotalDataSize,
			InjectSpeed,
		)
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

func TestParamsConfigJSONUsesSupportedExperimentWindow(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "paramsConfig.json"))
	if err != nil {
		t.Fatal(err)
	}

	var config globalConfig
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatalf("paramsConfig.json should parse with Go json.Unmarshal: %v", err)
	}

	if config.SpringMode < 0 || config.SpringMode > 7 {
		t.Fatalf("SpringMode = %d, want a supported mode in [0, 7]", config.SpringMode)
	}
	if config.SpringOnlineTrain != 0 {
		t.Fatalf("SpringOnlineTrain = %d, want 0 for fixed-model BlockEmulator evaluation", config.SpringOnlineTrain)
	}
	if config.SpringEvalSample != 0 {
		t.Fatalf("SpringEvalSample = %d, want deterministic evaluation", config.SpringEvalSample)
	}
	knownWindow := false
	for _, window := range []struct {
		start int
		total int
	}{
		{start: 2_500_000, total: 444_019},   // validation template
		{start: 2_944_019, total: 2_000_000}, // held-out test template
	} {
		if config.DatasetStartTx == window.start && config.TotalDataSize == window.total {
			knownWindow = true
			break
		}
	}
	if !knownWindow {
		t.Fatalf(
			"DatasetStartTx/TotalDataSize = %d/%d, want a declared validation or test window",
			config.DatasetStartTx,
			config.TotalDataSize,
		)
	}
	if config.TxBatchSize != 1000 || config.MaxBlockSizeGlobal != 1000 ||
		config.InjectSpeed <= 0 {
		t.Fatalf("TxBatchSize/BlockSize/InjectSpeed = %d/%d/%d, want 1000/1000/positive", config.TxBatchSize, config.MaxBlockSizeGlobal, config.InjectSpeed)
	}

	assertConfigWeight := func(name string, got *float64, want float64) {
		t.Helper()
		if got == nil {
			t.Fatalf("%s is missing from paramsConfig.json", name)
		}
		if math.Abs(*got-want) > 1e-9 {
			t.Fatalf("%s = %.12f, want %.12f", name, *got, want)
		}
	}

	// The proposed PPO uses the IoT feature vector and TopK7. Original SPRING-PPO
	// and non-learning comparators keep the same IoT identities without that vector.
	if config.SpringMode == 2 && config.SpringIOTMode == 1 {
		if config.SpringIOTIdentityMode != 1 || config.SpringIOTFeatureDim != 10 {
			t.Fatalf("proposed PPO IoT mode/identity/features = %d/%d/%d, want 1/1/10", config.SpringIOTMode, config.SpringIOTIdentityMode, config.SpringIOTFeatureDim)
		}
		if config.SpringCandidateTopK != 7 {
			t.Fatalf("proposed PPO SpringCandidateTopK = %d, want 7", config.SpringCandidateTopK)
		}
		assertConfigWeight("SpringIOTCSTRWeight", config.SpringIOTCSTRWeight, 0.55)
		assertConfigWeight("SpringIOTBalanceWeight", config.SpringIOTBalanceWeight, 0.30)
		assertConfigWeight("SpringIOTCommCostWeight", config.SpringIOTCommCostWeight, 0.10)
		assertConfigWeight("SpringIOTHotspotWeight", config.SpringIOTHotspotWeight, 0.05)
		wantModel := fmt.Sprintf("ppo_top7_w5530_pareto_16s_seed%d.pt", config.SpringRandomSeed)
		if filepath.Base(config.SpringModelFile) != wantModel ||
			filepath.Base(filepath.Dir(config.SpringModelFile)) != "models" {
			t.Fatalf("SpringModelFile = %q, want a models directory containing %q", config.SpringModelFile, wantModel)
		}
	} else if config.SpringMode == 7 {
		// Candidate-Only 必须保留 Proposed 的 IoT 状态和 TopK7 候选构造，
		// 唯一移除的是 PPO 模型推理，因此模型必须明确为 NOT_APPLICABLE。
		if config.SpringIOTMode != 1 || config.SpringIOTIdentityMode != 1 || config.SpringIOTFeatureDim != 10 {
			t.Fatalf("candidate-only IoT mode/identity/features = %d/%d/%d, want 1/1/10", config.SpringIOTMode, config.SpringIOTIdentityMode, config.SpringIOTFeatureDim)
		}
		if config.SpringCandidateTopK != 7 {
			t.Fatalf("candidate-only SpringCandidateTopK = %d, want 7", config.SpringCandidateTopK)
		}
		if config.SpringModelFile != "NOT_APPLICABLE" {
			t.Fatalf("candidate-only SpringModelFile = %q, want NOT_APPLICABLE", config.SpringModelFile)
		}
		assertConfigWeight("SpringIOTCSTRWeight", config.SpringIOTCSTRWeight, 0.55)
		assertConfigWeight("SpringIOTBalanceWeight", config.SpringIOTBalanceWeight, 0.30)
		assertConfigWeight("SpringIOTCommCostWeight", config.SpringIOTCommCostWeight, 0.10)
		assertConfigWeight("SpringIOTHotspotWeight", config.SpringIOTHotspotWeight, 0.05)
	} else {
		if config.SpringIOTMode != 0 || config.SpringIOTIdentityMode != 1 || config.SpringIOTFeatureDim != 0 {
			t.Fatalf("baseline IoT mode/identity/features = %d/%d/%d, want 0/1/0", config.SpringIOTMode, config.SpringIOTIdentityMode, config.SpringIOTFeatureDim)
		}
		if config.SpringCandidateTopK != 0 {
			t.Fatalf("baseline SpringCandidateTopK = %d, want 0", config.SpringCandidateTopK)
		}
		if config.SpringMode == 2 {
			wantModel := fmt.Sprintf("original_spring_ppo_16s_seed%d.pt", config.SpringRandomSeed)
			if filepath.Base(config.SpringModelFile) != wantModel ||
				filepath.Base(filepath.Dir(config.SpringModelFile)) != "models" {
				t.Fatalf("SpringModelFile = %q, want a models directory containing %q", config.SpringModelFile, wantModel)
			}
		} else if config.SpringModelFile != "NOT_APPLICABLE" {
			t.Fatalf("SpringModelFile = %q, want NOT_APPLICABLE for non-PPO baseline", config.SpringModelFile)
		}
	}

	if strings.Contains(config.SpringModelFile, `瀹為獙`) || strings.Contains(config.ExpDataRootDir, `瀹為獙`) {
		t.Fatalf("paramsConfig.json contains mojibake path: model=%q exp=%q", config.SpringModelFile, config.ExpDataRootDir)
	}
	wantRunKey := fmt.Sprintf(`i%d_seed%d_`, config.InjectSpeed, config.SpringRandomSeed)
	normalizedRoot := strings.ReplaceAll(config.ExpDataRootDir, "/", `\`)
	if !strings.Contains(normalizedRoot, `\block_eval\`) ||
		!strings.Contains(config.ExpDataRootDir, wantRunKey) ||
		!strings.HasSuffix(normalizedRoot, `\expTest`) {
		t.Fatalf("ExpDataRootDir = %q, want a coherent block_eval expTest path containing %q", config.ExpDataRootDir, wantRunKey)
	}
}
